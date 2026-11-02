"""几何与时变干扰真值；仅点接收功率及阵列波束观测进入策略。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import MapConfig, ProjectConfig
from .propagation import source_received_dbm
from .sensing import DirectionalReceiver, DirectionalScan


@dataclass(frozen=True)
class Measurement:
    """位置m、时间s、含噪总功率dBm及可选波束快照；不是远处场的直接测量。"""

    position_m: tuple[float, float]
    time_seconds: float
    interference_dbm: float
    directional_scan: DirectionalScan | None = None


class Geometry:
    """已知平面障碍；线段与膨胀矩形作精确几何相交检查。"""

    def __init__(self, config: MapConfig, clearance_m: float = 0.0):
        self.config = config
        self.high = np.array([(config.columns - 1) * config.resolution_m,
                              (config.rows - 1) * config.resolution_m])
        self.boxes = [(np.array(box[:2]) - clearance_m, np.array(box[2:]) + clearance_m)
                      for box in config.obstacles_m]

    def free(self, point) -> bool:
        point = np.asarray(point, dtype=float)
        return bool(point.shape == (2,) and np.all(np.isfinite(point))
                    and np.all(point >= 0) and np.all(point <= self.high)
                    and not any(np.all(point >= low) and np.all(point <= high) for low, high in self.boxes))

    def edge_free(self, start, end) -> bool:
        if not self.free(start) or not self.free(end):
            return False
        start, end = np.asarray(start), np.asarray(end)
        direction = end - start
        for low, high in self.boxes:
            enter, leave = 0.0, 1.0
            for axis in range(2):
                if abs(direction[axis]) < 1e-12:
                    if start[axis] < low[axis] or start[axis] > high[axis]:
                        enter, leave = 1.0, 0.0
                        break
                else:
                    first = (low[axis] - start[axis]) / direction[axis]
                    second = (high[axis] - start[axis]) / direction[axis]
                    enter = max(enter, min(first, second))
                    leave = min(leave, max(first, second))
            if enter <= leave:
                return False
        return True

    def node(self, point) -> tuple[int, int]:
        column, row = np.rint(np.asarray(point) / self.config.resolution_m).astype(int)
        return int(column), int(row)

    def position(self, node) -> np.ndarray:
        return np.asarray(node, dtype=float) * self.config.resolution_m

    def grid(self) -> np.ndarray:
        columns, rows = np.meshgrid(np.arange(self.config.columns), np.arange(self.config.rows))
        return np.stack((columns, rows), axis=-1) * self.config.resolution_m


class ElectromagneticWorld:
    """真值仅供传感器、执行评价和回放使用，规划器不接收本对象。"""

    def __init__(self, config: ProjectConfig):
        config.validate()
        self.config = config
        self.geometry = Geometry(config.map, config.aircraft.clearance_m)
        if not self.geometry.free(config.map.start_m) or not self.geometry.free(config.map.goal_m):
            raise ValueError("start or goal intersects geometry or is outside map")
        self.random = np.random.default_rng(config.seed)
        self.receiver = DirectionalReceiver(config)

    def _active_fields(self, time_seconds: float):
        for index, item in enumerate(self.config.electromagnetic.fields):
            if item["start_seconds"] <= time_seconds < item["end_seconds"]:
                center = np.asarray(item["center_m"]) + np.asarray(item["velocity_mps"]) * (
                    time_seconds - item["start_seconds"])
                yield index, center, item

    def source_positions_for_evaluation(self, time_seconds: float) -> list[dict]:
        """仅供研究者回放的真实源位置；不进入观测、地图估计或模型上下文。"""
        return [{"id": f"S{index + 1}", "position_m": center.tolist()}
                for index, center, _ in self._active_fields(time_seconds)]

    def truth(self, points, time_seconds: float):
        """合成总干扰功率，dBm；背景和各源在线性毫瓦域叠加，无服务站。"""
        points = np.asarray(points, dtype=float)
        electromagnetic = self.config.electromagnetic
        interference = np.full(points.shape[:-1], 10 ** (electromagnetic.background_dbm / 10))
        for _, center, item in self._active_fields(time_seconds):
            interference += 10 ** (source_received_dbm(points, center, item, self.config) / 10)
        result = 10 * np.log10(interference)
        return float(result) if result.ndim == 0 else result

    def observe(self, position, time_seconds: float) -> Measurement:
        """同步点测和360°波束接收；源数量、位置、距离及远处功率均不直接返回。"""
        if not self.geometry.free(position) or not math.isfinite(time_seconds) or time_seconds < 0:
            raise ValueError("invalid sensor position or time")
        measured = self.truth(position, time_seconds) + self.random.normal(
            0.0, self.config.electromagnetic.measurement_noise_db)
        if not math.isfinite(measured):
            raise ValueError("nonfinite observation")
        if self.config.sensor.mode == "power_only":
            return Measurement(tuple(float(value) for value in position), float(time_seconds), float(measured))
        contributions, bearings = [], []
        for _, center, source in self._active_fields(time_seconds):
            contributions.append(float(source_received_dbm(position, center, source, self.config)))
            delta = center - position
            bearings.append(float(np.degrees(np.arctan2(delta[1], delta[0])) % 360)
                            if np.linalg.norm(delta) > 1e-6 else None)
        scan = self.receiver.scan(contributions, bearings, time_seconds)
        return Measurement(tuple(float(value) for value in position), float(time_seconds), float(measured), scan)
