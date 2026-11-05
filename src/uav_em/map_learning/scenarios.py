"""有界拒绝采样的合成2.5D场景，几何验收不是现实真实性证明。"""

from collections import deque
from dataclasses import dataclass, replace
import math

import numpy as np

from ..config import ProjectConfig, MapConfig
from ..environment import Geometry, ElectromagneticWorld


@dataclass(frozen=True)
class ScenarioSettings:
    """节点行列数，resolution_m单位m，horizon_seconds单位s。

    building_count为楼数，max_attempts为有界重采样次数；源数0至max_sources。
    min_free_fraction为可飞比例，min_connected_fraction为主分量占可飞格比例。
    建筑、材料、功率、速度取值为未标定研究先验，不是城市统计分布。
    """

    columns: int = 33
    rows: int = 25
    resolution_m: float = 20.0
    building_count: int = 8
    max_sources: int = 3
    horizon_seconds: float = 100.0
    max_attempts: int = 40
    min_free_fraction: float = 0.55
    min_connected_fraction: float = 0.9
    require_goal_reachable_in_time: bool = True
    min_building_count: int | None = None

    def validate(self):
        if type(self.require_goal_reachable_in_time) is not bool:
            raise ValueError('require_goal_reachable_in_time must be boolean')
        if self.min_building_count is not None and (type(self.min_building_count) is not int
                or not 0 <= self.min_building_count <= self.building_count):
            raise ValueError('min_building_count must be an integer in [0, building_count]')
        for name in ('columns', 'rows', 'building_count', 'max_sources', 'max_attempts'):
            value = getattr(self, name)
            minimum = 8 if name in ('columns', 'rows') else 1 if name == 'max_attempts' else 0
            if type(value) is not int or value < minimum:
                raise ValueError(f'invalid scenario setting: {name}')
        if self.max_sources > 3:
            raise ValueError('max_sources must be <= 3')
        for name in ('resolution_m', 'horizon_seconds'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for value in (self.min_free_fraction, self.min_connected_fraction):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError('geometry fractions must be in (0,1]')


def public_geometry(config):
    return Geometry(config.map, config.aircraft.clearance_m, config.aircraft.altitude_m,
                    config.aircraft.vertical_clearance_m)


def free_graph(geometry):
    """只读公开建筑；每条栅格边均通过连续碰撞检查。"""
    nodes = {(column, row) for row in range(geometry.config.rows)
             for column in range(geometry.config.columns) if geometry.free(geometry.position((column, row)))}
    graph = {node: [] for node in sorted(nodes)}
    for column, row in graph:
        for neighbor in ((column + 1, row), (column, row + 1)):
            if neighbor in nodes and geometry.edge_free(geometry.position((column, row)), geometry.position(neighbor)):
                graph[(column, row)].append(neighbor)
                graph[neighbor].append((column, row))
    return graph


def reachable(graph, start):
    parents = {start: None}
    pending = deque([start])
    while pending:
        current = pending.popleft()
        for neighbor in graph[current]:
            if neighbor not in parents:
                parents[neighbor] = current
                pending.append(neighbor)
    return parents


def geometry_route(geometry, graph, start, target):
    """采集专用公开几何BFS备选，不冒称RRT*成功或LLM决策。"""
    ordered = sorted(graph, key=lambda node: float(np.linalg.norm(geometry.position(node) - start)))
    connector = next((node for node in ordered if geometry.edge_free(start, geometry.position(node))), None)
    target_node = geometry.node(target)
    if connector is None or target_node not in graph:
        return []
    parents = reachable(graph, connector)
    if target_node not in parents or not geometry.edge_free(geometry.position(target_node), target):
        return []
    reverse, current = [], target_node
    while current is not None:
        reverse.append(geometry.position(current).tolist())
        current = parents[current]
    points = [list(start), *reversed(reverse), list(target)]
    return [point for index, point in enumerate(points)
            if index == 0 or np.linalg.norm(np.asarray(point) - points[index - 1]) > 1e-8]


def generate_scenario(seed, settings=ScenarioSettings()):
    """配置与仅供离线审核的报告；隐藏源不进入采集策略。"""
    settings.validate()
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    random = np.random.default_rng(seed)
    high = np.array([settings.columns - 1, settings.rows - 1]) * settings.resolution_m
    base = ProjectConfig(seed=seed)
    for attempt in range(settings.max_attempts):
        count = settings.building_count if settings.min_building_count is None else int(
            random.integers(settings.min_building_count, settings.building_count + 1))
        start = np.array([settings.resolution_m, random.integers(1, settings.rows - 1) * settings.resolution_m])
        goal = np.array([high[0] - settings.resolution_m,
                         random.integers(1, settings.rows - 1) * settings.resolution_m])
        buildings, rectangles = [], []
        for _ in range(count * 30):
            if len(buildings) == count:
                break
            size = random.uniform(1.5, 4.0, 2) * settings.resolution_m
            low = random.uniform(settings.resolution_m, high - size - settings.resolution_m)
            upper = low + size
            margin = base.aircraft.clearance_m + settings.resolution_m * 0.4
            if any(np.all(point >= low - margin) and np.all(point <= upper + margin) for point in (start, goal)):
                continue
            if any(np.all(upper + margin >= other_low) and np.all(low - margin <= other_high)
                   for other_low, other_high in rectangles):
                continue
            rectangles.append((low, upper))
            buildings.append({'id': f'building_{len(buildings)}', 'height_m': float(random.uniform(10, 55)),
                              'footprint_m': [low.tolist(), [float(upper[0]), float(low[1])],
                                              upper.tolist(), [float(low[0]), float(upper[1])]],
                              'loss_db': float(random.uniform(5, 16))})
        if len(buildings) != count:
            continue
        map_config = MapConfig(columns=settings.columns, rows=settings.rows, resolution_m=settings.resolution_m,
                               start_m=tuple(start), goal_m=tuple(goal), obstacles_m=(),
                               geometry_mode='fixed_altitude', buildings=tuple(buildings),
                               provenance={'source': 'procedural synthetic simulation', 'generator_seed': seed})
        electromagnetic = replace(base.electromagnetic, fields=(), propagation_mode='single_reflection',
                                  path_loss_exponent=2.0, carrier_frequency_hz=float(random.choice([9e8, 2.4e9, 5.8e9])),
                                  relative_permittivity=float(random.uniform(3, 7)),
                                  conductivity_s_per_m=float(random.uniform(0.001, 0.02)))
        config = replace(base, map=map_config, electromagnetic=electromagnetic,
                         simulation=replace(base.simulation, max_seconds=settings.horizon_seconds))
        geometry = public_geometry(config)
        graph = free_graph(geometry)
        if geometry.node(start) not in graph or geometry.node(goal) not in graph:
            continue
        connected = reachable(graph, geometry.node(start))
        free_fraction = len(graph) / (settings.rows * settings.columns)
        if (free_fraction < settings.min_free_fraction or len(connected) / len(graph) < settings.min_connected_fraction
                or geometry.node(goal) not in connected):
            continue
        route = geometry_route(geometry, graph, start, goal)
        distance = sum(float(np.linalg.norm(np.asarray(end) - begin)) for begin, end in zip(route, route[1:]))
        energy = distance / config.aircraft.speed_mps * config.aircraft.flight_power_w / 3600
        if energy > config.aircraft.initial_energy_wh - config.aircraft.reserve_energy_wh:
            continue
        route_seconds = distance / config.aircraft.speed_mps
        if settings.require_goal_reachable_in_time and route_seconds > settings.horizon_seconds:
            continue
        fields = []
        source_geometry = Geometry(map_config, 0.0, 1.5, 0.0)
        source_count = int(random.integers(settings.max_sources + 1))
        for source_index in range(source_count):
            moving = bool(random.integers(2))
            begin = 0.0 if source_index == 0 or random.random() < 0.5 else float(random.uniform(0.15, 0.5) * settings.horizon_seconds)
            end = settings.horizon_seconds + 1.0 if random.random() < 0.7 else float(random.uniform(0.65, 0.95) * settings.horizon_seconds)
            for _ in range(100):
                center = random.uniform(settings.resolution_m, high - settings.resolution_m)
                angle = random.uniform(0, 2 * math.pi)
                speed = float(random.uniform(0.2, 1.5)) if moving else 0.0
                velocity = np.array([math.cos(angle), math.sin(angle)]) * speed
                endpoint = center + velocity * (end - begin)
                if source_geometry.edge_free(center, endpoint):
                    fields.append({'center_m': center.tolist(), 'height_m': 1.5,
                                   'eirp_dbm': float(random.uniform(-12, 10)), 'velocity_mps': velocity.tolist(),
                                   'start_seconds': begin, 'end_seconds': end})
                    break
        if len(fields) != source_count:
            continue
        config = replace(config, electromagnetic=replace(electromagnetic, fields=tuple(fields)))
        config.validate()
        world = ElectromagneticWorld(config)
        probe_power = [world.truth(np.asarray([start, goal]), time) for time in (0.0, settings.horizon_seconds / 2)]
        if not np.isfinite(probe_power).all():
            continue
        return config, {'seed': seed, 'attempt': attempt + 1, 'free_fraction': free_fraction,
                        'connected_fraction': len(connected) / len(graph), 'geometry_route_m': distance,
                        'geometry_route_energy_wh': energy, 'building_count': len(buildings),
                        'geometry_route_seconds': route_seconds,
                        'fits_collection_horizon': route_seconds <= settings.horizon_seconds,
                        'source_count': source_count,
                        'moving_source_count': int(sum(np.linalg.norm(item['velocity_mps']) > 0 for item in fields)),
                        'finite_power_check': True, 'kind': 'synthetic_not_calibrated'}
    raise ValueError(f'no admissible scenario after {settings.max_attempts} attempts for seed {seed}')
