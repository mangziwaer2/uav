"""单点观测驱动的局部干扰功率估计；未知区域保留背景先验并明确标记。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import deque

import numpy as np

from .config import ProjectConfig
from .environment import Geometry, Measurement
from .source_model import SourceTracker
from .sensing import scan_record
from .propagation import PropagationModel
from .local_estimation import local_gp_prediction, local_support_evidence


@dataclass
class MapSnapshot:
    """同一时刻的策略可见估计，不包含任何未观测的干扰源真值。"""

    time_seconds: float
    # 以下栅格数组按[行,列]索引；无支持处的背景估计仅为数值先验。
    estimate_dbm: np.ndarray
    # 常值模式为启发式尺度，GP为模型条件下的预测标准差dB；均非校准风险概率。
    uncertainty_db: np.ndarray
    supported: np.ndarray
    # 常值为选中测量年龄；GP为最近训练点年龄s，并非全部GP证据的年龄；无支持为-1。
    age_seconds: np.ndarray
    observation_count: int
    geometry: Geometry
    source_prediction_dbm: np.ndarray | None = None
    source_hypothesis: dict | None = None
    directional_observations: tuple[dict, ...] = ()
    estimation_details: dict | None = None
    nearest_support_distance_m: np.ndarray | None = None
    local_support_count: np.ndarray | None = None
    recent_point_measurements: tuple[dict, ...] = ()

    def at(self, position) -> tuple[float, float, bool]:
        """取最近网格的估计功率dBm、误差dB及支持标记，不访问真值。"""
        column, row = self.geometry.node(position)
        return (float(self.estimate_dbm[row, column]), float(self.uncertainty_db[row, column]),
                bool(self.supported[row, column]))


class ElectromagneticBelief:
    """缓存每个访问网格最近一次测量；不缓存未来真值。"""

    def __init__(self, config: ProjectConfig, geometry: Geometry):
        self.config = config
        self.geometry = geometry
        self.measurements: dict[tuple[int, int], Measurement] = {}
        self.observation_count = 0
        self.latest_time_seconds = -np.inf
        self.directional_measurements = deque(maxlen=128)
        self.source_tracker = SourceTracker(geometry.high, config.electromagnetic.background_dbm,
                                           config.belief.max_age_seconds,
                                           config.belief.source_max_fit_rmse_db,
                                           config.belief.source_count_penalty_db2,
                                           PropagationModel.from_config(config), config.sensor,
                                           config.belief.source_bearing_weight_db2,
                                           config.belief.source_max_bearing_rmse_deg)

    def update(self, measurement: Measurement) -> dict:
        if not self.geometry.free(measurement.position_m):
            raise ValueError("observation outside traversable map")
        values = (*measurement.position_m, measurement.time_seconds, measurement.interference_dbm)
        if not all(np.isfinite(value) for value in values) or measurement.time_seconds < 0:
            raise ValueError("invalid observation")
        if measurement.time_seconds < self.latest_time_seconds:
            raise ValueError("out-of-order observation")
        node = self.geometry.node(measurement.position_m)
        previous = self.measurements.get(node)
        if previous is not None and measurement.time_seconds < previous.time_seconds:
            raise ValueError("out-of-order observation")
        if measurement.directional_scan is not None:
            measurement.directional_scan.validate()
            self.directional_measurements.append(measurement)
        self.measurements[node] = measurement
        self.latest_time_seconds = measurement.time_seconds
        self.observation_count += 1
        return self.source_tracker.observe(measurement)

    def snapshot(self, time_seconds: float) -> MapSnapshot:
        """按距离与年龄选观测支持；有有效等效源时用其预测加观测残差，不把预测变成已知。"""
        grid = self.geometry.grid()
        prior = np.full(grid.shape[:-1], self.config.electromagnetic.background_dbm, dtype=float)
        estimate = prior.copy()
        settings = self.config.belief
        error = np.full(prior.shape, settings.unknown_error_db, dtype=float)
        age = np.full(prior.shape, -1.0)
        supported = np.zeros(prior.shape, dtype=bool)
        score = np.full(prior.shape, np.inf)
        hypothesis = self.source_tracker.active_hypothesis(time_seconds)
        prediction = (self.source_tracker.predict(hypothesis, grid, time_seconds)
                      if hypothesis is not None else None)
        if prediction is not None:
            estimate = prediction.copy()
        details = None
        support_distance, support_count = None, None
        if settings.estimation_method == "local_gp":
            prior_at_measurements = (lambda m: self.source_tracker.predict(
                hypothesis, m.position_m, m.time_seconds)) if prediction is not None else None
            estimate, error, supported, age, count = local_gp_prediction(
                grid, time_seconds, self.measurements.values(), settings,
                prediction if prediction is not None else self.config.electromagnetic.background_dbm,
                prior_at_measurements)
            support_distance, support_count = local_support_evidence(
                grid, time_seconds, self.measurements.values(), settings)
            details = {"method": "local_gp", "kernel": "matern32_space_times_exponential_time",
                       "training_point_count": count, "maximum_training_points": settings.gp_max_points,
                       "spatial_length_scale_m": settings.gp_length_scale_m,
                       "temporal_length_scale_seconds": settings.gp_time_scale_seconds,
                       "local_prediction_radius_m": settings.support_radius_m,
                       "maximum_measurement_age_seconds": settings.max_age_seconds,
                       "uncertainty_kind": "GP_predictive_std_db_under_assumed_model_not_calibrated",
                       "support_is_not_direct_measurement": True,
                       "support_count_is_not_independent_sample_count": True,
                       "temporal_decay_is_not_evidence_of_source_shutdown": True,
                       "building_shadow_boundaries_are_not_modelled_by_kernel": True}
        for measurement in (() if details is not None else self.measurements.values()):
            elapsed = time_seconds - measurement.time_seconds
            if elapsed < 0 or elapsed > settings.max_age_seconds:
                continue
            distance = np.linalg.norm(grid - measurement.position_m, axis=-1)
            local_error = (settings.measured_error_db + distance * settings.error_per_meter_db
                           + elapsed * settings.error_per_second_db)
            use = (distance <= settings.support_radius_m) & (local_error < score)
            # 局部常值外推不代表直接测得整个圆盘或完整源边界。
            estimate[use] = measurement.interference_dbm
            if prediction is not None:
                expected_measurement = self.source_tracker.predict(hypothesis, measurement.position_m,
                                                                  measurement.time_seconds)
                estimate[use] = prediction[use] + measurement.interference_dbm - expected_measurement
            error[use] = local_error[use]
            age[use] = elapsed
            supported[use] = True
            score[use] = local_error[use]
        scans = [item for item in self.directional_measurements
                 if 0 <= time_seconds - item.time_seconds <= settings.max_age_seconds]
        indexes = np.unique(np.linspace(0, len(scans) - 1, min(6, len(scans))).astype(int)) if scans else []
        recent = [m for m in self.source_tracker.recent(time_seconds) if m.time_seconds <= time_seconds]
        point_indexes = np.unique(np.linspace(0, len(recent) - 1, min(8, len(recent))).astype(int)) if recent else []
        points = tuple({"position_m": list(recent[i].position_m), "time_seconds": recent[i].time_seconds,
                        "interference_dbm": recent[i].interference_dbm} for i in point_indexes)
        return MapSnapshot(float(time_seconds), estimate, error, supported, age,
                           self.observation_count, self.geometry, prediction,
                           asdict(hypothesis) if hypothesis is not None else None,
                           tuple(scan_record(scans[index]) for index in indexes), details,
                           support_distance, support_count, points)
