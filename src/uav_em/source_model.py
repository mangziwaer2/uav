"""LLM源假设的公开模型前向预测与观测检验；不读取真源或自动求解源位置。"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math

import numpy as np

from .environment import Measurement
from .propagation import PropagationModel, propagation_received_dbm, propagation_arrivals
from .sensing import angular_peak_indices, beam_power_dbm


MAX_SOURCES = 3
MIN_SIGMA_M = 5.0
MAX_SPEED_MPS = 20.0
MIN_PEAK_DBM = -150.0
MAX_PEAK_DBM = 0.0
MAX_EIRP_DBM = 80.0
MAX_HEIGHT_M = 100.0


@dataclass(frozen=True)
class EquivalentSource:
    """中心m、空间尺度m、峰值接收功率贡献dBm、平移速度m/s；非真实发射机参数。"""

    center_m: tuple[float, float]
    sigma_m: float
    peak_dbm: float
    velocity_mps: tuple[float, float]


@dataclass(frozen=True)
class PropagationSource:
    """候选发射源：中心m、频带内EIRP dBm、相对地面高度m和速度m/s，均为假设。"""

    center_m: tuple[float, float]
    eirp_dbm: float
    height_m: float
    velocity_mps: tuple[float, float]


@dataclass(frozen=True)
class SourceHypothesis:
    """reference_seconds是中心坐标的仿真参考时刻s，不是源出现时间。"""

    reference_seconds: float
    sources: tuple[EquivalentSource | PropagationSource, ...]


def parse_source_hypothesis(value, bounds) -> SourceHypothesis:
    """严格解析同一类型至多3个候选源；传播源为默认协议，高斯源保留为显式合成基线。"""
    def number(item):
        return not isinstance(item, bool) and isinstance(item, (int, float)) and math.isfinite(item)

    def vector(item):
        return isinstance(item, list) and len(item) == 2 and all(number(part) for part in item)

    if not isinstance(value, dict) or set(value) != {"reference_seconds", "sources"}:
        raise ValueError("source_hypothesis requires reference_seconds and sources")
    reference = value["reference_seconds"]
    if not number(reference) or reference < 0:
        raise ValueError("source reference time must be finite and nonnegative")
    sources = value["sources"]
    if not isinstance(sources, list) or len(sources) > MAX_SOURCES:
        raise ValueError("source hypothesis supports at most three sources")
    parsed = []
    for source in sources:
        gaussian_keys = {"center_m", "sigma_m", "peak_dbm", "velocity_mps"}
        propagation_keys = {"center_m", "eirp_dbm", "height_m", "velocity_mps"}
        if not isinstance(source, dict) or set(source) not in (gaussian_keys, propagation_keys):
            raise ValueError("invalid source schema; do not mix Gaussian and propagation parameters")
        if not vector(source["center_m"]) or not vector(source["velocity_mps"]):
            raise ValueError("source center and velocity must have two finite coordinates")
        if np.any(np.asarray(source["center_m"]) < 0) or np.any(np.asarray(source["center_m"]) > bounds):
            raise ValueError("source reference center outside map")
        if np.linalg.norm(source["velocity_mps"]) > MAX_SPEED_MPS:
            raise ValueError("source speed exceeds limit")
        center = tuple(float(part) for part in source["center_m"])
        velocity = tuple(float(part) for part in source["velocity_mps"])
        if set(source) == propagation_keys:
            if not number(source["eirp_dbm"]) or not MIN_PEAK_DBM <= source["eirp_dbm"] <= MAX_EIRP_DBM:
                raise ValueError("source EIRP outside limits")
            if not number(source["height_m"]) or not 0 <= source["height_m"] <= MAX_HEIGHT_M:
                raise ValueError("source height outside limits")
            parsed.append(PropagationSource(center, float(source["eirp_dbm"]), float(source["height_m"]), velocity))
        else:
            if not number(source["sigma_m"]) or not MIN_SIGMA_M <= source["sigma_m"] <= np.linalg.norm(bounds):
                raise ValueError("source spatial scale outside limits")
            if not number(source["peak_dbm"]) or not MIN_PEAK_DBM <= source["peak_dbm"] <= MAX_PEAK_DBM:
                raise ValueError("source peak power outside limits")
            parsed.append(EquivalentSource(center, float(source["sigma_m"]), float(source["peak_dbm"]), velocity))
    if len({type(source) for source in parsed}) > 1:
        raise ValueError("source models cannot be mixed in one hypothesis")
    return SourceHypothesis(float(reference), tuple(parsed))


def source_contributions(hypothesis, points, time_seconds, propagation=None):
    """只用候选源及公开近似，返回每个候选源的预测贡献和时变中心。"""
    points = np.asarray(points, dtype=float)
    times = np.asarray(time_seconds, dtype=float)
    for source in hypothesis.sources:
        center = (np.asarray(source.center_m)
                  + np.expand_dims(times - hypothesis.reference_seconds, -1) * np.asarray(source.velocity_mps))
        if isinstance(source, PropagationSource):
            if propagation is None:
                raise ValueError("propagation source requires an explicit public propagation model")
            contribution = propagation_received_dbm(points, center, asdict(source), propagation)
        else:
            distance_squared = np.sum((points - center) ** 2, axis=-1)
            contribution = source.peak_dbm - 10 / np.log(10) * distance_squared / (2 * source.sigma_m ** 2)
        yield contribution, center


def predict_power(hypothesis: SourceHypothesis, points, time_seconds, background_dbm: float,
                  propagation: PropagationModel | None = None):
    """在线性功率域叠加背景和所有源；支持逐点时间，绝不读取隐藏源。"""
    points = np.asarray(points, dtype=float)
    power = np.full(points.shape[:-1], 10 ** (background_dbm / 10), dtype=float)
    for contribution, _ in source_contributions(hypothesis, points, time_seconds, propagation):
        power += 10 ** (contribution / 10)
    result = 10 * np.log10(power)
    return float(result) if result.ndim == 0 else result


class SourceTracker:
    """缓存带时序的功率／方向观测并检验LLM提议，不把拟合等同于真实定位。"""

    def __init__(self, bounds, background_dbm: float, window_seconds: float,
                 max_fit_rmse_db: float, count_penalty_db2: float,
                 propagation: PropagationModel | None = None, sensor=None,
                 bearing_weight_db2: float = 9.0, max_bearing_rmse_deg: float = 45.0):
        self.bounds = np.asarray(bounds, dtype=float)
        self.background_dbm = background_dbm
        self.window_seconds = window_seconds
        self.max_fit_rmse_db = max_fit_rmse_db
        self.count_penalty_db2 = count_penalty_db2
        self.propagation = propagation
        self.sensor = sensor
        self.bearing_weight_db2 = bearing_weight_db2
        self.max_bearing_rmse_deg = max_bearing_rmse_deg
        self.measurements = deque(maxlen=128)
        self.prediction_checks = deque(maxlen=32)
        self.hypothesis: SourceHypothesis | None = None
        self.accepted_seconds: float | None = None
        self.revision = 0

    def predict(self, hypothesis, points, time_seconds):
        return predict_power(hypothesis, points, time_seconds, self.background_dbm, self.propagation)

    def bearing_check(self, hypothesis, measurement):
        """观测角峰与候选源合成后的预测角峰比较，不给真实源配对或泄漏标定偏差。"""
        scan = measurement.directional_scan
        if scan is None or not scan.peak_indices or self.sensor is None or self.sensor.mode == "power_only":
            return None
        contributions, bearings = [], []
        if hypothesis is not None:
            for source in hypothesis.sources:
                center = (np.asarray(source.center_m) + (measurement.time_seconds - hypothesis.reference_seconds)
                          * np.asarray(source.velocity_mps))
                if isinstance(source, PropagationSource):
                    for arrival in propagation_arrivals(measurement.position_m, center, asdict(source), self.propagation):
                        contributions.append(float(arrival.power_dbm))
                        bearings.append(float(arrival.bearing_deg) if np.isfinite(arrival.bearing_deg) else None)
                else:
                    delta = center - measurement.position_m
                    distance_squared = np.sum(delta ** 2)
                    contributions.append(float(source.peak_dbm - 10 / np.log(10)
                                               * distance_squared / (2 * source.sigma_m ** 2)))
                    bearings.append(float(np.degrees(np.arctan2(delta[1], delta[0])) % 360)
                                    if np.linalg.norm(delta) > 1e-6 else None)
        predicted = beam_power_dbm(contributions, bearings, scan.azimuth_deg, self.background_dbm,
                                   scan.beamwidth_deg, self.sensor.front_back_db)
        threshold = max(self.sensor.detection_threshold_dbm, self.background_dbm + self.sensor.minimum_excess_db,
                        float(np.median(predicted)) + self.sensor.minimum_contrast_db) - 2 * self.sensor.noise_db
        indexes = angular_peak_indices(scan.azimuth_deg, predicted, threshold, scan.beamwidth_deg)
        predicted_angles = [scan.azimuth_deg[index] for index in indexes]
        residuals = []
        for index in scan.peak_indices:
            observed = scan.azimuth_deg[index]
            nearest = min(predicted_angles, key=lambda angle: abs((observed - angle + 180) % 360 - 180),
                          default=None)
            residual = (observed - nearest + 180) % 360 - 180 if nearest is not None else 180.0
            residuals.append({"observed_deg": observed, "predicted_deg": nearest, "residual_deg": residual})
        scale = max(1.0, float(np.sqrt((scan.beamwidth_deg / 2) ** 2 + scan.bearing_error_deg ** 2
                                      + (360 / len(scan.azimuth_deg)) ** 2 / 12)))
        errors = np.array([item["residual_deg"] for item in residuals])
        normalized = np.abs(errors) / scale
        huber = np.where(normalized <= 1, normalized ** 2 / 2, normalized - 0.5)
        return {"time_seconds": measurement.time_seconds, "position_m": list(measurement.position_m),
                "predicted_peaks_deg": predicted_angles, "peak_residuals": residuals,
                "mean_square_deg2": float(np.mean(errors ** 2)), "loss": float(np.mean(huber)),
                "scale_deg": scale, "correlation_seconds": scan.calibration_correlation_seconds}

    def bearing_score(self, hypothesis, measurements, now):
        checks = [check for item in measurements if (check := self.bearing_check(hypothesis, item)) is not None]
        if not checks:
            return {"scan_count": 0, "block_count": 0, "rmse_deg": None, "loss": 0.0,
                    "worst_residuals": []}
        correlation = max(item["correlation_seconds"] for item in checks)
        blocks = {}
        for index, check in enumerate(checks):
            key = int((now - check["time_seconds"]) / correlation) if correlation > 0 else index
            blocks.setdefault(key, []).append(check)
        block_errors, block_losses, weights = [], [], []
        for block in blocks.values():
            ages = np.array([now - item["time_seconds"] for item in block])
            local_weights = np.exp(-ages / self.window_seconds)
            block_errors.append(np.average([item["mean_square_deg2"] for item in block], weights=local_weights))
            block_losses.append(np.average([item["loss"] for item in block], weights=local_weights))
            weights.append(float(np.mean(local_weights)))
        worst = sorted(checks, key=lambda item: -item["mean_square_deg2"])[:3]
        return {"scan_count": len(checks), "block_count": len(blocks),
                "rmse_deg": float(np.sqrt(np.average(block_errors, weights=weights))),
                "loss": float(np.average(block_losses, weights=weights)),
                "correlation_block_seconds": correlation,
                "worst_residuals": [{key: item[key] for key in (
                    "time_seconds", "position_m", "predicted_peaks_deg", "peak_residuals", "scale_deg")}
                                    for item in worst]}

    def observe(self, measurement: Measurement) -> dict:
        """先检验旧假设对新点的预测，再加入测量；不以更新后拟合误差冒充创新量。"""
        check = {"source_prediction_dbm": None, "source_residual_db": None,
                 "source_revision": self.revision}
        if (self.hypothesis is not None
                and 0 <= measurement.time_seconds - self.accepted_seconds <= self.window_seconds):
            prediction = self.predict(self.hypothesis, measurement.position_m, measurement.time_seconds)
            check.update(source_prediction_dbm=prediction,
                         source_residual_db=measurement.interference_dbm - prediction)
            bearing = self.bearing_check(self.hypothesis, measurement)
            if bearing is not None:
                check["source_bearing_check"] = bearing
            self.prediction_checks.append({"position_m": list(measurement.position_m),
                                           "time_seconds": measurement.time_seconds, **check})
        self.measurements.append(measurement)
        return check

    def recent(self, now: float) -> list[Measurement]:
        return [item for item in self.measurements if -1e-6 <= now - item.time_seconds <= self.window_seconds]

    def score(self, hypothesis: SourceHypothesis | None, now: float) -> dict:
        """年龄加权Huber误差dB²加每源复杂度惩罚；RMSE只是在当前窗口拟合的误差dB。"""
        measurements = self.recent(now)
        if not measurements:
            return {"count": 0, "rmse_db": None, "objective_db2": None, "worst_residuals": [],
                    "bearing_fit": self.bearing_score(hypothesis, [], now)}
        points = [item.position_m for item in measurements]
        times = np.array([item.time_seconds for item in measurements])
        prediction = (self.predict(hypothesis, points, times)
                      if hypothesis is not None else np.full(len(points), self.background_dbm))
        residual = np.array([item.interference_dbm for item in measurements]) - prediction
        absolute = np.abs(residual)
        weights = np.exp(-(now - times) / self.window_seconds)
        huber = np.where(absolute <= 5, residual ** 2 / 2, 5 * (absolute - 2.5))
        worst = np.argsort(-absolute, kind="stable")[:3]
        bearing = self.bearing_score(hypothesis, measurements, now)
        return {"count": len(measurements), "rmse_db": float(np.sqrt(np.average(residual ** 2, weights=weights))),
                "objective_db2": float(np.average(huber, weights=weights)
                                       + self.bearing_weight_db2 * bearing["loss"]
                                       + self.count_penalty_db2 * (len(hypothesis.sources) if hypothesis else 0)),
                "power_loss_db2": float(np.average(huber, weights=weights)),
                "bearing_loss_db2": self.bearing_weight_db2 * bearing["loss"], "bearing_fit": bearing,
                "worst_residuals": [{"position_m": list(measurements[index].position_m),
                                     "time_seconds": float(times[index]),
                                     "measured_dbm": measurements[index].interference_dbm,
                                     "predicted_dbm": float(prediction[index]),
                                     "residual_db": float(residual[index])} for index in worst]}

    def status(self, now: float) -> str:
        if self.hypothesis is None or now < self.accepted_seconds:
            return "unavailable"
        if now - self.accepted_seconds > self.window_seconds:
            return "expired"
        fit = self.score(self.hypothesis, now)
        if not fit["count"] or fit["rmse_db"] > self.max_fit_rmse_db:
            return "suspended_bad_fit"
        if fit["bearing_fit"]["rmse_deg"] is not None and fit["bearing_fit"]["rmse_deg"] > self.max_bearing_rmse_deg:
            return "suspended_bad_bearing"
        return "active"

    def active_hypothesis(self, now: float) -> SourceHypothesis | None:
        return self.hypothesis if self.status(now) == "active" else None

    def consider(self, hypothesis: SourceHypothesis, now: float) -> dict:
        """同一观测窗口比较新旧假设；坏提议不覆盖旧假设，不保证源的可辨识性。"""
        if abs(hypothesis.reference_seconds - now) > 0.002:
            return {"status": "rejected_reference_time", "candidate_fit": None,
                    "incumbent_fit": self.score(self.hypothesis, now),
                    "background_fit": self.score(None, now), "revision_after": self.revision,
                    "fit_is_not_future_validation": True}
        candidate = self.score(hypothesis, now)
        incumbent = self.score(self.hypothesis, now)
        baseline = self.score(None, now)
        status = "accepted"
        comparison = incumbent if self.status(now) != "expired" else baseline
        if not candidate["count"]:
            status = "rejected_no_observations"
        elif candidate["rmse_db"] > self.max_fit_rmse_db:
            status = "rejected_bad_fit"
        elif (candidate["bearing_fit"]["rmse_deg"] is not None
              and candidate["bearing_fit"]["rmse_deg"] > self.max_bearing_rmse_deg):
            status = "rejected_bad_bearing"
        elif candidate["objective_db2"] >= (min(comparison["objective_db2"], baseline["objective_db2"])
                                           if hypothesis.sources else incumbent["objective_db2"]) - 1e-9:
            status = "rejected_no_improvement"
        if status == "accepted":
            self.hypothesis = hypothesis
            self.accepted_seconds = now
            self.revision += 1
        return {"status": status, "candidate_fit": candidate, "incumbent_fit": incumbent,
                "background_fit": baseline, "revision_after": self.revision,
                "fit_is_not_future_validation": True}

    def summary(self, now: float) -> dict:
        """固定上限的原始测量和新点预测残差；不暴露环境配置或未来脚本。"""
        recent = self.recent(now)
        indexes = np.unique(np.linspace(0, len(recent) - 1, min(32, len(recent))).astype(int)) if recent else []
        return {"status": self.status(now), "revision": self.revision,
                "background_prior_dbm": self.background_dbm,
                "hypothesis": asdict(self.hypothesis) if self.hypothesis else None,
                "current_fit": self.score(self.hypothesis, now),
                "measurement_columns": ["x_m", "y_m", "time_seconds", "interference_dbm"],
                "measurements": [[*recent[index].position_m, recent[index].time_seconds,
                                   recent[index].interference_dbm] for index in indexes],
                "measurement_count": len(recent), "window_seconds": self.window_seconds,
                "recommended_source_model": self.propagation.propagation_mode if self.propagation is not None else "gaussian",
                "public_propagation_model": asdict(self.propagation) if self.propagation is not None else None,
                "source_fields": ["center_m", "eirp_dbm", "height_m", "velocity_mps"]
                                 if self.propagation is not None else ["center_m", "sigma_m", "peak_dbm", "velocity_mps"],
                "new_point_checks": [item for item in self.prediction_checks
                                     if 0 <= now - item["time_seconds"] <= self.window_seconds][-6:],
                "limits": {"max_sources": MAX_SOURCES, "sigma_range_m": [MIN_SIGMA_M, float(np.linalg.norm(self.bounds))],
                           "peak_range_dbm": [MIN_PEAK_DBM, MAX_PEAK_DBM], "max_speed_mps": MAX_SPEED_MPS,
                           "eirp_range_dbm": [MIN_PEAK_DBM, MAX_EIRP_DBM], "height_range_m": [0, MAX_HEIGHT_M],
                           "max_bearing_rmse_deg": self.max_bearing_rmse_deg,
                           "bearing_weight_db2": self.bearing_weight_db2,
                           "max_fit_rmse_db": self.max_fit_rmse_db, "count_penalty_db2": self.count_penalty_db2},
                "bearing_validation": "observed peaks to nearest predicted blended peak; no source IDs; "
                                      "many-to-one matches allowed; absence of peaks is not negative evidence; "
                                      "correlated scans averaged in time blocks, not a calibrated likelihood",
                "meaning": "source hypotheses only; fit is nonunique, model predictions are not observations; "
                           "timestamps must be used; no inferred source birth/death or physical safety guarantee"}
