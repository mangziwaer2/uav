"""米、Hz和dBm的大尺度传播近似；遮挡用原始矩形／多边形棱柱，不用飞行净空膨胀体。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from .buildings import Building, ring_edges

import numpy as np


@dataclass(frozen=True)
class PropagationModel:
    """公开传播近似与已知几何；不保存源参数、活动脚本或真值环境。"""

    altitude_m: float
    carrier_frequency_hz: float
    reference_distance_m: float
    path_loss_exponent: float
    obstacle_height_m: float
    obstacle_loss_db: float
    obstacles_m: tuple
    buildings: tuple = ()
    propagation_mode: str = "log_distance"
    relative_permittivity: float = 4.0
    conductivity_s_per_m: float = 0.005

    @classmethod
    def from_config(cls, config):
        settings = config.electromagnetic
        return cls(config.aircraft.altitude_m, settings.carrier_frequency_hz, settings.reference_distance_m,
                   settings.path_loss_exponent, settings.obstacle_height_m, settings.obstacle_loss_db,
                   tuple(tuple(box) for box in config.map.obstacles_m),
                   tuple(Building.from_dict(item) for item in config.map.buildings),
                   settings.propagation_mode, settings.relative_permittivity, settings.conductivity_s_per_m)

    @cached_property
    def propagation_buildings(self):
        """只用原始楼体生成反射面；矩形的统一高度不改变飞行几何。"""
        rectangles = tuple(Building(f"rectangle_{index}", ((box[0], box[1]), (box[2], box[1]),
                           (box[2], box[3]), (box[0], box[3])), (), self.obstacle_height_m,
                           loss_db=self.obstacle_loss_db)
                           for index, box in enumerate(self.obstacles_m) if self.obstacle_height_m > 0)
        return (*rectangles, *self.buildings)

    @cached_property
    def reflection_surfaces(self):
        """平地z=0与有限竖直外墙／庭院墙；不包含屋顶反射。"""
        surfaces = [(np.zeros(3), np.array([0.0, 0.0, 1.0]), None)]
        for building in self.propagation_buildings:
            for ring in (building.footprint_m, *building.holes_m):
                for first, second in ring_edges(ring):
                    edge = np.asarray(second) - first
                    length = np.linalg.norm(edge)
                    normal = np.array([edge[1], -edge[0], 0.0]) / length
                    anchor = np.array([*first, building.base_m])
                    surfaces.append((anchor, normal, (edge / length, length, building.height_m)))
        return tuple(surfaces)


@dataclass(frozen=True)
class Arrival:
    """仿真器内部的路径功率dBm与来波方位°；不向策略暴露路径或反射点。"""

    power_dbm: np.ndarray
    bearing_deg: np.ndarray
    kind: str


def bearing_degrees(delta):
    """从接收位置指向来波的水平角；竖直来波记NaN，由阵列近似作无方位贡献。"""
    delta = np.asarray(delta)
    return np.where(np.linalg.norm(delta[..., :2], axis=-1) > 1e-6,
                    np.degrees(np.arctan2(delta[..., 1], delta[..., 0])) % 360, np.nan)


def source_received_dbm(points, center, source, config, propagation=None):
    """只供真值生成与传感器内部调用；高斯场仅作为显式合成基线。"""
    points = np.asarray(points, dtype=float)
    horizontal_squared = np.sum((points - center) ** 2, axis=-1)
    if "sigma_m" in source:
        return source["peak_dbm"] - 10 / np.log(10) * horizontal_squared / (2 * source["sigma_m"] ** 2)
    return propagation_received_dbm(points, center, source, propagation or PropagationModel.from_config(config))


def propagation_received_dbm(points, center, source, settings: PropagationModel):
    """同一源的有限路径非相干叠加；不含背景，不做源反演。"""
    power = np.zeros(np.asarray(points).shape[:-1])
    for arrival in propagation_arrivals(points, center, source, settings):
        power += 10 ** (arrival.power_dbm / 10)
    return 10 * np.log10(power)


def source_arrivals(points, center, source, config, propagation=None):
    """点测与测向共用同一组到达路径；高斯基线没有反射路径。"""
    if "sigma_m" in source:
        yield Arrival(source_received_dbm(points, center, source, config),
                      bearing_degrees(np.asarray(center) - points), "gaussian")
    else:
        yield from propagation_arrivals(points, center, source, propagation or PropagationModel.from_config(config))


def direct_received_dbm(points, center, source, settings):
    """直达／等效穿楼路径：距离衰减和每栋楼一次的等效穿透损耗。"""
    points = np.asarray(points, dtype=float)
    horizontal_squared = np.sum((points - center) ** 2, axis=-1)
    vertical = settings.altitude_m - source["height_m"]
    distance = np.maximum(np.sqrt(horizontal_squared + vertical ** 2), settings.reference_distance_m)
    reference_loss = 20 * np.log10(4 * np.pi * settings.reference_distance_m
                                  * settings.carrier_frequency_hz / 299792458)
    received = (source["eirp_dbm"] - reference_loss
                - 10 * settings.path_loss_exponent * np.log10(distance / settings.reference_distance_m))
    if settings.obstacle_loss_db:
        for box in settings.obstacles_m:
            blocked = prism_intersection(points, settings.altitude_m, center, source["height_m"],
                                         box, settings.obstacle_height_m)
            received = received - blocked * settings.obstacle_loss_db
    for building in settings.buildings:
        if building.loss_db:
            received = received - building.ray_intersection(points, settings.altitude_m, center,
                                                            source["height_m"]) * building.loss_db
    return received


def fresnel_power(cos_incidence, frequency_hz, relative_permittivity, conductivity_s_per_m):
    """空气入射非磁性有耗介质；TE／TM反射功率等权平均，不追踪极化和相位。"""
    cosine = np.clip(np.asarray(cos_incidence, dtype=float), 0, 1)
    permittivity = relative_permittivity - 1j * conductivity_s_per_m / (2 * np.pi * frequency_hz * 8.8541878128e-12)
    transmitted = np.sqrt(permittivity - (1 - cosine ** 2) + 0j)
    perpendicular = (cosine - transmitted) / np.where(abs(cosine + transmitted) > 1e-15,
                                                      cosine + transmitted, 1.0)
    parallel = (permittivity * cosine - transmitted) / np.where(
        abs(permittivity * cosine + transmitted) > 1e-15, permittivity * cosine + transmitted, 1.0)
    return np.clip((abs(perpendicular) ** 2 + abs(parallel) ** 2) / 2, 0, 1)


def segment_hits_building(start, end, building):
    """任意三维线段与原始棱柱相交；收缩端点以允许反射点接触墙面。"""
    direction = end - start
    trim = np.minimum(0.25, 1e-6 / np.maximum(np.linalg.norm(direction, axis=-1), 1e-12))
    start = start + direction * trim[..., None]
    end = end - direction * trim[..., None]
    vertical = end[..., 2] - start[..., 2]
    parallel = abs(vertical) < 1e-12
    denominator = np.where(parallel, 1.0, vertical)
    first = (building.base_m - start[..., 2]) / denominator
    second = (building.base_m + building.height_m - start[..., 2]) / denominator
    enter = np.where(parallel, 0.0, np.maximum(0, np.minimum(first, second)))
    leave = np.where(parallel, 1.0, np.minimum(1, np.maximum(first, second)))
    valid = ((enter <= leave) & (leave >= 0) & (enter <= 1)
             & (~parallel | ((start[..., 2] >= building.base_m)
                              & (start[..., 2] <= building.base_m + building.height_m))))
    clipped_start = start + (end - start) * np.clip(enter, 0, 1)[..., None]
    clipped_end = start + (end - start) * np.clip(leave, 0, 1)[..., None]
    return valid & building.segment_blocked(clipped_start[..., :2], clipped_end[..., :2])


def propagation_arrivals(points, center, source, settings):
    """镜像法求单次墙／地面反射；反射两段必须不穿楼，功率按总路径长衰减。"""
    points, center = np.broadcast_arrays(np.asarray(points, dtype=float), np.asarray(center, dtype=float))
    receiver = np.concatenate((points, np.full((*points.shape[:-1], 1), settings.altitude_m)), axis=-1)
    transmitter = np.concatenate((center, np.full((*center.shape[:-1], 1), source["height_m"])), axis=-1)
    yield Arrival(direct_received_dbm(points, center, source, settings),
                  bearing_degrees(transmitter - receiver), "direct")
    if settings.propagation_mode != "single_reflection":
        return
    wavelength = 299792458 / settings.carrier_frequency_hz
    for anchor, normal, bounds in settings.reflection_surfaces:
        receiver_side = np.sum((receiver - anchor) * normal, axis=-1)
        source_side = np.sum((transmitter - anchor) * normal, axis=-1)
        image = transmitter - 2 * source_side[..., None] * normal
        denominator = receiver_side + source_side
        fraction = receiver_side / np.where(abs(denominator) > 1e-12, denominator, 1.0)
        bounce = receiver + fraction[..., None] * (image - receiver)
        valid = (receiver_side * source_side > 1e-12) & (fraction > 0) & (fraction < 1)
        if bounds is not None:
            tangent, length, height = bounds
            along = np.sum((bounce[..., :2] - anchor[:2]) * tangent, axis=-1)
            valid &= ((along > 1e-7) & (along < length - 1e-7)
                      & (bounce[..., 2] > anchor[2] + 1e-7) & (bounce[..., 2] < anchor[2] + height - 1e-7))
        if not np.any(valid):
            continue
        for building in settings.propagation_buildings:
            valid &= ~(segment_hits_building(receiver, bounce, building)
                       | segment_hits_building(bounce, transmitter, building))
        if not np.any(valid):
            continue
        incoming_length = np.linalg.norm(transmitter - bounce, axis=-1)
        outgoing_length = np.linalg.norm(receiver - bounce, axis=-1)
        distance = np.maximum(incoming_length + outgoing_length, settings.reference_distance_m)
        cosine = abs(np.sum((transmitter - bounce) * normal, axis=-1)) / np.maximum(incoming_length, 1e-12)
        reflectivity = fresnel_power(cosine, settings.carrier_frequency_hz,
                                    settings.relative_permittivity, settings.conductivity_s_per_m)
        gain = (wavelength / (4 * np.pi * distance)) ** 2 * reflectivity
        reflected = source["eirp_dbm"] + 10 * np.log10(np.maximum(gain, np.finfo(float).tiny))
        yield Arrival(np.where(valid & (gain > 0), reflected, -np.inf), bearing_degrees(bounce - receiver),
                      "ground" if bounds is None else "wall")


def prism_intersection(points, altitude_m, center, height_m, box, obstacle_height_m):
    """接收点到发射点的三维线段是否穿过矩形棱柱；每个棱柱只计一次损耗。"""
    points = np.asarray(points, dtype=float)
    enter = np.zeros(points.shape[:-1])
    leave = np.ones(points.shape[:-1])
    starts = (points[..., 0], points[..., 1], np.full(points.shape[:-1], altitude_m))
    center = np.asarray(center, dtype=float)
    ends = (center[..., 0], center[..., 1], height_m)
    for start, end, low, high in zip(starts, ends, (*box[:2], 0), (*box[2:], obstacle_height_m)):
        direction = end - start
        parallel = np.abs(direction) < 1e-12
        denominator = np.where(parallel, 1.0, direction)
        first, second = (low - start) / denominator, (high - start) / denominator
        enter = np.maximum(enter, np.where(parallel, -np.inf, np.minimum(first, second)))
        leave = np.minimum(leave, np.where(parallel, np.inf, np.maximum(first, second)))
        leave = np.where(parallel & ((start < low) | (start > high)), -np.inf, leave)
    return (enter <= leave) & (leave > 0) & (enter < 1) & (obstacle_height_m > 0)
