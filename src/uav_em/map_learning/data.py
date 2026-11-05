"""不读取隐藏真值的多行为采集、机载地图张量与因果时序训练题。"""

from dataclasses import asdict, dataclass, replace
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ..belief import ElectromagneticBelief
from ..planning import plan, path_length
from ..sensing import scan_record
from ..simulation import advance_route
from ..environment import ElectromagneticWorld
from .scenarios import ScenarioSettings, generate_scenario, public_geometry, free_graph, reachable, geometry_route


CHANNELS = ('power', 'supported', 'uncertainty', 'age', 'blocked', 'building_height')
NORMALIZATION = {'power_low_dbm': -120.0, 'power_high_dbm': -30.0, 'uncertainty_scale_db': 30.0,
                 'age_scale_seconds': 35.0, 'height_scale_m': 80.0}
POLICIES = ('goal', 'coverage', 'revisit')


@dataclass(frozen=True)
class CollectionSettings:
    """step_seconds采样与执行步长s，dwell_seconds往返策略到点悬停s。

    flights_per_policy为每场景每行为独立噪声／目标种子的次数；budget为RRT*采样提议数。
    task_stride为训练题锚点间隔帧数，window_spacing_seconds为历史帧目标间隔s。
    """

    step_seconds: float = 2.0
    dwell_seconds: float = 6.0
    flights_per_policy: int = 1
    budget: int = 250
    task_stride: int = 5
    window_frames: int = 4
    window_spacing_seconds: float = 5.0
    roi_size: int = 8

    def validate(self):
        for name in ('step_seconds', 'dwell_seconds', 'window_spacing_seconds'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        for name in ('flights_per_policy', 'budget', 'task_stride', 'window_frames', 'roi_size'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.window_frames < 2:
            raise ValueError('temporal tasks require at least two frames')


def json_write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def snapshot_tensor(snapshot, static_geometry):
    """未知功率置零并由mask区分；不把背景填充值当作已知低干扰。"""
    blocked, height = static_geometry
    support = snapshot.supported & ~blocked
    power = np.clip((snapshot.estimate_dbm + 120) / 90, 0, 1)
    return np.stack((np.where(support, power, 0), support,
                     snapshot.uncertainty_db / 30, np.where(support, snapshot.age_seconds / 35, 0),
                     blocked, height / 80)).astype(np.float32)


def static_channels(geometry):
    points = geometry.grid()
    blocked = np.array([[not geometry.free(point) for point in row] for row in points])
    height = np.zeros(blocked.shape, dtype=np.float32)
    for building in geometry.buildings:
        height = np.maximum(height, np.where(building.contains(points), building.height_m + building.base_m, 0))
    return blocked, height


def collect_episode(config, policy, settings=CollectionSettings(), flight_seed=0):
    """脚本化离线采集，不是simulate实验，不声称探索策略智能或安全最优。"""
    settings.validate()
    if policy not in POLICIES:
        raise ValueError('unknown collection policy')
    config = replace(config, seed=flight_seed)
    world = ElectromagneticWorld(config)
    geometry, aircraft = world.geometry, config.aircraft
    graph = free_graph(geometry)
    nodes = list(reachable(graph, geometry.node(config.map.start_m)))
    random = np.random.default_rng(flight_seed)
    belief = ElectromagneticBelief(config, geometry)
    static = static_channels(geometry)
    position = np.array(config.map.start_m, dtype=float)
    candidates = np.array([geometry.position(node) for node in nodes])
    distances = np.linalg.norm(candidates - position, axis=1)
    eligible = candidates[(distances >= 3 * config.map.resolution_m) & (distances <= 10 * config.map.resolution_m)]
    if not len(eligible):
        eligible = candidates[distances > 0]
    anchor = eligible[int(random.integers(len(eligible)))]
    frames, times, positions, energies, observations, searches, transitions = [], [], [], [], [], [], []
    visited = [position.copy()]
    clock, energy, next_review, dwell_until = 0.0, aircraft.initial_energy_wh, 0.0, 0.0
    target, route, leg = None, [], 0
    flown, fallbacks = 0.0, 0
    while True:
        measurement = world.observe(position, clock)
        belief.update(measurement)
        snapshot = belief.snapshot(clock)
        frames.append(snapshot_tensor(snapshot, static))
        times.append(clock)
        positions.append(position.copy())
        energies.append(energy)
        observations.append({'time_seconds': clock, 'position_m': position.tolist(),
                             'interference_dbm': measurement.interference_dbm,
                             'directional': scan_record(measurement) if measurement.directional_scan else None})
        if (clock >= config.simulation.max_seconds - 1e-8 or energy <= aircraft.reserve_energy_wh + 1e-8
                or policy == 'goal' and np.linalg.norm(position - config.map.goal_m) < 1e-7):
            break
        if target is not None and np.linalg.norm(position - target) < 1e-7:
            target, route = None, []
            if policy == 'revisit':
                dwell_until = clock + settings.dwell_seconds
        if clock >= dwell_until and (target is None or policy == 'goal' and clock >= next_review):
            if target is None:
                if policy == 'goal':
                    target = np.asarray(config.map.goal_m)
                elif policy == 'revisit':
                    target = anchor.copy() if leg % 2 == 0 else np.asarray(config.map.start_m)
                    leg += 1
                else:
                    sample = candidates[random.choice(len(candidates), min(100, len(candidates)), replace=False)]
                    novelty = np.linalg.norm(sample[:, None] - np.asarray(visited)[None], axis=-1).min(axis=1)
                    target = sample[int(np.argmax(novelty))]
                    if np.linalg.norm(target - position) < 1e-7:
                        target = anchor.copy()
            profile = 'avoid_interference' if policy == 'goal' else 'balanced'
            result = plan(snapshot, position, target, 'rrt_star', profile,
                          config.electromagnetic.interference_reference_dbm, settings.budget,
                          flight_seed + len(searches))
            route = result.path
            fallback = not bool(route)
            if fallback:
                route = geometry_route(geometry, graph, position, target)
                fallbacks += 1
            searches.append({'time_seconds': clock, 'target_m': target.tolist(), 'profile': profile,
                             'rrt_found': bool(result.path), 'geometry_fallback': fallback})
            if not route:
                raise RuntimeError('connected geometry failed to produce collection route')
            visited.append(target.copy())
            next_review = clock + config.simulation.review_seconds
        duration = min(settings.step_seconds, config.simulation.max_seconds - clock)
        moving = len(route) > 1 and clock >= dwell_until
        power = aircraft.flight_power_w if moving else aircraft.hover_power_w
        duration = min(duration, (energy - aircraft.reserve_energy_wh) * 3600 / power)
        if duration <= 1e-8:
            break
        if moving:
            distance = min(aircraft.speed_mps * duration, path_length(route))
            duration = distance / aircraft.speed_mps
            route, traversed = advance_route(route, distance)
            if not all(geometry.edge_free(begin, end) for begin, end in zip(traversed, traversed[1:])):
                raise RuntimeError('collection violated geometry')
            position = np.asarray(traversed[-1])
            flown += distance
        else:
            traversed = [position.tolist()]
        transitions.append({'start_seconds': clock, 'end_seconds': clock + duration,
                            'traversed_m': traversed, 'moving': moving,
                            'energy_used_wh': power * duration / 3600})
        energy -= power * duration / 3600
        clock += duration
    return {'maps': np.stack(frames), 'times': np.asarray(times, dtype=np.float64),
            'positions': np.asarray(positions, dtype=np.float32), 'energy_wh': np.asarray(energies, dtype=np.float32)}, {
                'policy': policy, 'flight_seed': flight_seed, 'observations': observations, 'searches': searches,
                'transitions': transitions,
                'frames': len(frames), 'flown_m': flown, 'geometry_fallbacks': fallbacks,
                'supported_fraction_mean': float(np.mean(np.stack(frames)[:, 1])),
                'supported_high_cells': int(np.sum((np.stack(frames)[:, 0] > 40 / 90) & (np.stack(frames)[:, 1] > 0.5))),
                'is_llm_experiment': False}


def history_window(times, anchor, frames=4, spacing_seconds=5.0):
    """仅使用锚点之前的实际帧；缺失／重复采样位标为无效，不伪造历史。"""
    times = np.asarray(times)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or type(anchor) is not int or not 0 <= anchor < len(times) or type(frames) is not int or frames < 1
            or not math.isfinite(spacing_seconds) or spacing_seconds < 1e-6):
        raise ValueError('invalid history window')
    selected, used = [], set()
    for offset in reversed(range(frames)):
        desired = times[anchor] - offset * spacing_seconds
        index = int(np.searchsorted(times[:anchor + 1], desired + 1e-8, side='right') - 1)
        if index < 0 or index in used:
            selected.append(-1)
        else:
            selected.append(index)
            used.add(index)
    return {'indices': selected, 'valid': [index >= 0 for index in selected],
            'offset_seconds': [float(times[index] - times[anchor]) if index >= 0 else None for index in selected]}


def temporal_target(older, newer, box):
    """同一世界坐标区域的估计变化，不监督源运动；不以未知填充值算功率差。"""
    left, top, right, bottom = box
    first, last = older[:, top:bottom, left:right], newer[:, top:bottom, left:right]
    before, after = first[1] > 0.5, last[1] > 0.5
    common = before & after
    delta = (last[0] - first[0]) * 90
    return {'supported_before': before.astype(int).tolist(), 'supported_now': after.astype(int).tolist(),
            'newly_supported': (~before & after).astype(int).tolist(),
            'lost_support': (before & ~after).astype(int).tolist(),
            'power_delta_db': [[round(float(value), 3) if valid else None for value, valid in zip(row, mask)]
                               for row, mask in zip(delta, common)],
            'age_seconds_now': [[round(float(value * 35), 3) if valid else None
                                 for value, valid in zip(row, mask)] for row, mask in zip(last[3], after)],
            'interpretation': 'belief_change_not_source_motion'}


def temporal_tasks(arrays, settings, seed):
    random = np.random.default_rng(seed)
    maps, times, positions = arrays['maps'], arrays['times'], arrays['positions']
    height, width = maps.shape[-2:]
    for anchor in range(settings.task_stride, len(times), settings.task_stride):
        window = history_window(times, anchor, settings.window_frames, settings.window_spacing_seconds)
        valid = [index for index in window['indices'] if index >= 0]
        if len(valid) < 2:
            continue
        size = min(settings.roi_size, height, width)
        if random.random() < 0.7:
            point = positions[anchor if random.random() < 0.5 else valid[0]]
            center = np.rint(point / arrays['resolution_m']).astype(int)
        else:
            center = np.array([random.integers(width), random.integers(height)])
        left = int(np.clip(center[0] - size // 2, 0, width - size))
        top = int(np.clip(center[1] - size // 2, 0, height - size))
        box = [left, top, left + size, top + size]
        yield {'anchor': anchor, 'window': window, 'roi_grid_xyxy': box,
               'older_index': valid[0], 'task': 'temporal_region',
               'target': temporal_target(maps[valid[0]], maps[anchor], box)}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def generate_scene(job):
    root, index, seed, split, scenario, collection = job
    root = Path(root)
    config, report = generate_scenario(seed + index, scenario)
    scene_id = f'scene_{index:05d}'
    json_write(root / 'evaluation_only' / f'{scene_id}.json', config.to_dict())
    json_write(root / 'evaluation_only' / f'{scene_id}_report.json', report)
    episodes = []
    task_file = root / 'episodes' / f'{scene_id}_tasks.jsonl'
    with task_file.open('x', encoding='utf-8') as tasks:
        for policy_index, policy in enumerate(POLICIES):
            for flight in range(collection.flights_per_policy):
                flight_seed = seed + index * 10000 + policy_index * 1000 + flight
                arrays, metadata = collect_episode(config, policy, collection, flight_seed)
                arrays['resolution_m'] = np.array(config.map.resolution_m, dtype=np.float32)
                episode_id = f'{scene_id}_{policy}_{flight:03d}'
                relative = f'episodes/{episode_id}.npz'
                with (root / relative).open('xb') as stream:
                    np.savez_compressed(stream, **arrays)
                public = {'scenario_id': scene_id, 'split': split,
                          'map': config.to_dict()['map'], 'aircraft': asdict(config.aircraft),
                          'public_propagation': {key: value for key, value in asdict(config.electromagnetic).items()
                                                 if key not in ('fields', 'power_grid')}, **metadata}
                metadata_path = root / 'episodes' / f'{episode_id}.json'
                json_write(metadata_path, public)
                count = 0
                for task in temporal_tasks(arrays, collection, flight_seed):
                    tasks.write(json.dumps({'episode_id': episode_id, 'split': split, **task},
                                           ensure_ascii=False, allow_nan=False) + '\n')
                    count += 1
                episodes.append({'id': episode_id, 'path': relative, 'scenario_id': scene_id,
                                 'split': split, 'policy': policy, 'frames': metadata['frames'],
                                 'temporal_tasks': count, 'flown_m': metadata['flown_m'],
                                 'supported_fraction_mean': metadata['supported_fraction_mean'],
                                 'supported_high_cells': metadata['supported_high_cells'],
                                 'geometry_fallbacks': metadata['geometry_fallbacks'],
                                 'sha256': file_hash(root / relative), 'metadata_sha256': file_hash(metadata_path)})
    print(f'[场景完成] {scene_id} {split} 楼={report["building_count"]} '
          f'源={report["source_count"]} 移动源={report["moving_source_count"]}', flush=True)
    return {'id': scene_id, 'split': split}, episodes, str(task_file)


def generate_dataset(output, scene_count=24, seed=100, scenario=ScenarioSettings(), collection=CollectionSettings(), workers=1):
    """按场景划分；工作进程只写互不重叠的场景，主进程按索引确定性合并。"""
    scenario.validate()
    collection.validate()
    if type(scene_count) is not int or scene_count < 3 or type(seed) is not int or seed < 0:
        raise ValueError('need >=3 scenes and a nonnegative integer seed')
    if type(workers) is not int or not 1 <= workers <= 32 or collection.flights_per_policy > 1000:
        raise ValueError('workers must be in [1,32] and flights_per_policy <= 1000')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'episodes').mkdir()
    (root / 'evaluation_only').mkdir()
    random = np.random.default_rng(seed)
    order = random.permutation(scene_count)
    holdout = max(1, int(scene_count * 0.15))
    splits = {int(index): 'test' if rank < holdout else 'validation' if rank < 2 * holdout else 'train'
              for rank, index in enumerate(order)}
    source_root = Path(__file__).resolve().parents[1]
    manifest = {'schema_version': 1, 'quantity': 'onboard_belief_not_truth', 'seed': seed,
                'channels': list(CHANNELS), 'normalization': NORMALIZATION,
                'scenario_settings': asdict(scenario), 'collection_settings': asdict(collection),
                'split_unit': 'scenario', 'scenarios': [], 'episodes': [], 'workers': workers,
                'source_sha256': {str(path.relative_to(source_root)): file_hash(path)
                                  for path in sorted(source_root.rglob('*.py'))},
                'source_prediction_encoded': False,
                'training_stage': 'spatial_autoencoder_with_temporal_task_data', 'contains_llm_training': False}
    jobs = [(str(root.resolve()), index, seed, splits[index], scenario, collection) for index in range(scene_count)]
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        results = executor.map(generate_scene, jobs, chunksize=1) if executor else map(generate_scene, jobs)
        with (root / 'temporal_tasks.jsonl').open('x', encoding='utf-8') as tasks:
            for scene, episodes, task_path in results:
                manifest['scenarios'].append(scene)
                manifest['episodes'].extend(episodes)
                with Path(task_path).open(encoding='utf-8') as partial:
                    for line in partial:
                        tasks.write(line)
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    manifest['temporal_tasks_sha256'] = file_hash(root / 'temporal_tasks.jsonl')
    json_write(root / 'manifest.json', manifest)
    return manifest
