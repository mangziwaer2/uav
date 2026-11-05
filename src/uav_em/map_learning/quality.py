"""离线逐帧观测重放与覆盖审核；不生成任何actor可用的真值特征。"""

from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from ..belief import ElectromagneticBelief
from ..config import ProjectConfig, AircraftConfig, MapConfig
from ..environment import Measurement
from ..planning import path_length
from .data import (CHANNELS, NORMALIZATION, CollectionSettings, snapshot_tensor, static_channels,
                   temporal_tasks, file_hash, json_write)
from .scenarios import public_geometry


def checked_path(root, relative):
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError('dataset path escapes root')
    return path


def verify_episode(arrays, public, entry, replay=True):
    maps, times, positions, energies = (arrays[key] for key in ('maps', 'times', 'positions', 'energy_wh'))
    config = replace(ProjectConfig(), map=MapConfig(**public['map']), aircraft=AircraftConfig(**public['aircraft']))
    geometry = public_geometry(config)
    count = len(times)
    if (maps.shape != (count, len(CHANNELS), config.map.rows, config.map.columns)
            or positions.shape != (count, 2) or energies.shape != (count,) or count != entry['frames']):
        raise ValueError('invalid episode shapes')
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('nonfinite episode')
    if (count < 2 or times[0] != 0 or np.any(np.diff(times) <= 0)
            or np.any(np.diff(energies) > 1e-4) or energies[-1] < config.aircraft.reserve_energy_wh - 1e-4):
        raise ValueError('invalid timing or energy')
    if not np.isclose(float(arrays['resolution_m']), config.map.resolution_m):
        raise ValueError('resolution mismatch')
    for channel in (1, 4):
        if not np.isin(maps[:, channel], (0, 1)).all():
            raise ValueError('nonbinary support or geometry mask')
    if np.any(maps < -1e-6) or np.any(maps[:, 0] > 1 + 1e-6):
        raise ValueError('invalid normalized maps')
    support = maps[:, 1] > 0.5
    if np.any(maps[:, 0][~support] != 0) or np.any(maps[:, 3][~support] != 0):
        raise ValueError('unknown power or age populated')
    if np.any(support & (maps[:, 4] > 0.5)):
        raise ValueError('blocked cell marked supported')
    if not all(geometry.free(point) for point in positions):
        raise ValueError('vehicle outside free geometry')
    if len(public['observations']) != count or len(public['transitions']) != count - 1:
        raise ValueError('observation or transition count mismatch')
    static = static_channels(geometry)
    if not np.allclose(maps[:, 4], static[0]) or not np.allclose(maps[:, 5], static[1] / 80):
        raise ValueError('public geometry does not match map channels')
    belief = ElectromagneticBelief(config, geometry)
    for frame, observation in enumerate(public['observations']):
        if (not np.isclose(observation['time_seconds'], times[frame], atol=1e-7)
                or not np.allclose(observation['position_m'], positions[frame], atol=3e-5)):
            raise ValueError('measurement timestamp or position mismatch')
        if replay:
            belief.update(Measurement(tuple(observation['position_m']), observation['time_seconds'], observation['interference_dbm']))
            rebuilt = snapshot_tensor(belief.snapshot(times[frame]), static)
            if not np.allclose(rebuilt, maps[frame], atol=2e-6, rtol=1e-6):
                raise ValueError(f'frame {frame} differs from causal measurement replay')
    flown = 0.0
    for index, transition in enumerate(public['transitions']):
        route = transition['traversed_m']
        duration = times[index + 1] - times[index]
        distance = path_length(route)
        power = config.aircraft.flight_power_w if transition['moving'] else config.aircraft.hover_power_w
        if (not np.isclose(transition['start_seconds'], times[index])
                or not np.isclose(transition['end_seconds'], times[index + 1])
                or not np.allclose(route[0], positions[index], atol=3e-5)
                or not np.allclose(route[-1], positions[index + 1], atol=3e-5)
                or not all(geometry.edge_free(first, second) for first, second in zip(route, route[1:]))
                or not np.isclose(distance, config.aircraft.speed_mps * duration if transition['moving'] else 0, atol=1e-4)
                or not np.isclose(transition['energy_used_wh'], power * duration / 3600, atol=1e-6)
                or not np.isclose(energies[index] - energies[index + 1], transition['energy_used_wh'], atol=2e-5)):
            raise ValueError('invalid executed transition')
        flown += distance
    if not np.isclose(flown, public['flown_m'], atol=1e-4):
        raise ValueError('flight length mismatch')
    high = support & (maps[:, 0] > 40 / 90)
    return {'frames': count, 'supported_cells': int(support.sum()), 'high_cells': int(high.sum()),
            'weak_cells': int((support & ~high).sum()), 'unknown_cells': int((~support).sum()),
            'replayed_frames': count if replay else 0,
            'min_power_dbm': float(maps[:, 0][support].min() * 90 - 120) if support.any() else None,
            'max_power_dbm': float(maps[:, 0][support].max() * 90 - 120) if support.any() else None}


def audit_dataset(dataset, output, require_coverage=False):
    """完整审核后新建报告；覆盖只要求各split有基本类型，不声称分布充分。"""
    root = Path(dataset)
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('schema_version') != 1 or manifest.get('channels') != list(CHANNELS)
            or manifest.get('normalization') != NORMALIZATION or manifest.get('split_unit') != 'scenario'):
        raise ValueError('unsupported dataset protocol')
    collection = CollectionSettings(**manifest['collection_settings'])
    errors, summaries = [], []
    scenes = {item['id']: item['split'] for item in manifest['scenarios']}
    if len(scenes) != len(manifest['scenarios']):
        errors.append('duplicate scene id')
    source_counts = defaultdict(Counter)
    for scene_id, split in scenes.items():
        report = json.loads((root / 'evaluation_only' / f'{scene_id}_report.json').read_text(encoding='utf-8'))
        source_counts[split][f'sources_{report["source_count"]}'] += 1
        source_counts[split]['moving_scenes'] += int(report['moving_source_count'] > 0)
        source_counts[split]['scenes'] += 1
    seen_ids, seen_hashes, geometry_splits = set(), {}, {}
    aggregates = defaultdict(Counter)
    task_features = defaultdict(Counter)
    task_hash = file_hash(root / 'temporal_tasks.jsonl')
    if task_hash != manifest.get('temporal_tasks_sha256'):
        errors.append('temporal tasks hash mismatch')
    with (root / 'temporal_tasks.jsonl').open(encoding='utf-8') as task_stream:
        for episode in manifest['episodes']:
            episode_id = episode['id']
            try:
                if episode_id in seen_ids or scenes.get(episode['scenario_id']) != episode['split']:
                    raise ValueError('duplicate episode or split leakage')
                seen_ids.add(episode_id)
                path = checked_path(root, episode['path'])
                public_path = path.with_suffix('.json')
                if file_hash(path) != episode.get('sha256') or file_hash(public_path) != episode.get('metadata_sha256'):
                    raise ValueError('episode data or metadata hash mismatch')
                public = json.loads(public_path.read_text(encoding='utf-8'))
                if (public['scenario_id'] != episode['scenario_id'] or public['split'] != episode['split']
                        or 'fields' in public['public_propagation'] or 'power_grid' in public['public_propagation']):
                    raise ValueError('public metadata boundary mismatch')
                geometry_key = json.dumps([public['map']['buildings'], public['map']['obstacles_m']], sort_keys=True)
                if require_coverage and geometry_key in geometry_splits and geometry_splits[geometry_key] != episode['split']:
                    raise ValueError('identical building layout across splits')
                geometry_splits[geometry_key] = episode['split']
                if require_coverage and episode['sha256'] in seen_hashes:
                    raise ValueError('duplicate trajectory arrays')
                seen_hashes[episode['sha256']] = episode_id
                with np.load(path, allow_pickle=False) as stored:
                    arrays = {key: stored[key] for key in stored.files}
                summary = verify_episode(arrays, public, episode)
                summary.update({'id': episode_id, 'split': episode['split'], 'policy': episode['policy']})
                summaries.append(summary)
                split = episode['split']
                aggregates[split].update({key: value for key, value in summary.items() if isinstance(value, int)})
                aggregates[split][f'policy_{episode["policy"]}'] += 1
                task_count = 0
                for expected in temporal_tasks(arrays, collection, public['flight_seed']):
                    line = task_stream.readline()
                    if not line or json.loads(line) != {'episode_id': episode_id, 'split': split, **expected}:
                        raise ValueError('temporal task differs from causal ROI target')
                    target = expected['target']
                    features = task_features[split]
                    features['tasks'] += 1
                    features['new_support_tasks'] += int(np.any(target['newly_supported']))
                    features['lost_support_tasks'] += int(np.any(target['lost_support']))
                    deltas = [value for row in target['power_delta_db'] for value in row if value is not None]
                    features['power_change_tasks'] += int(any(abs(value) >= 1 for value in deltas))
                    features['all_unknown_tasks'] += int(not np.any(target['supported_before']) and not np.any(target['supported_now']))
                    task_count += 1
                if task_count != episode['temporal_tasks']:
                    raise ValueError('temporal task count mismatch')
            except (ValueError, KeyError, OSError, TypeError) as error:
                errors.append(f'{episode_id}: {error}')
        if task_stream.readline():
            errors.append('unexpected trailing temporal tasks')
    if require_coverage:
        for split in ('train', 'validation', 'test'):
            if source_counts[split]['scenes'] < 3:
                errors.append(f'{split}: fewer than 3 scenes')
            for key in ('sources_0', 'sources_1', 'sources_2', 'sources_3', 'moving_scenes'):
                if not source_counts[split][key]:
                    errors.append(f'{split}: missing {key}')
            for key in ('high_cells', 'weak_cells', 'unknown_cells', 'policy_goal', 'policy_coverage', 'policy_revisit'):
                if not aggregates[split][key]:
                    errors.append(f'{split}: missing {key}')
            for key in ('new_support_tasks', 'lost_support_tasks', 'power_change_tasks'):
                if not task_features[split][key]:
                    errors.append(f'{split}: missing {key}')
    result = {'schema_version': 1, 'passed': not errors, 'coverage_required': require_coverage,
              'manifest_sha256': file_hash(manifest_path), 'errors': errors,
              'episodes_checked': len(summaries), 'split_statistics': dict(aggregates),
              'temporal_statistics': dict(task_features), 'evaluation_only_source_coverage': dict(source_counts),
              'episode_statistics': summaries, 'replay': 'all_frames_from_public_measurements_no_truth',
              'note': 'integrity and basic coverage, not physical calibration or proof of sufficient training data'}
    json_write(output, result)
    return result
