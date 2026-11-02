"""米、Hz和dBm的窄带大尺度传播近似；遮挡用未膨胀矩形的三维棱柱。"""

from __future__ import annotations

from dataclasses import dataclass

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

    @classmethod
    def from_config(cls, config):
        settings = config.electromagnetic
        return cls(config.aircraft.altitude_m, settings.carrier_frequency_hz, settings.reference_distance_m,
                   settings.path_loss_exponent, settings.obstacle_height_m, settings.obstacle_loss_db,
                   tuple(tuple(box) for box in config.map.obstacles_m))


def source_received_dbm(points, center, source, config):
    """只供真值生成与传感器内部调用；高斯场仅作为显式合成基线。"""
    points = np.asarray(points, dtype=float)
    horizontal_squared = np.sum((points - center) ** 2, axis=-1)
    if "sigma_m" in source:
        return source["peak_dbm"] - 10 / np.log(10) * horizontal_squared / (2 * source["sigma_m"] ** 2)
    return propagation_received_dbm(points, center, source, PropagationModel.from_config(config))


def propagation_received_dbm(points, center, source, settings: PropagationModel):
    """由显式源参数与公开传播近似前向计算，支持逐点时变中心；不做源反演。"""
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
    return received


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
