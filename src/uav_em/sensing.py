"""阵列波束功率的可见观测；多个辐射源先叠加，绝不返回真源标识或距离。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DirectionalScan:
    """方向以东为0°、逆时针为正；功率dBm，只有局部角峰才作为检测方向。"""

    azimuth_deg: tuple[float, ...]
    power_dbm: tuple[float, ...]
    peak_indices: tuple[int, ...]
    detection_threshold_dbm: float
    beamwidth_deg: float
    bearing_error_deg: float
    calibration_correlation_seconds: float = 0.0

    def peaks(self) -> list[list[float]]:
        return [[self.azimuth_deg[index], self.power_dbm[index]] for index in self.peak_indices]

    def validate(self):
        if (not 4 <= len(self.azimuth_deg) <= 360 or len(self.azimuth_deg) != len(self.power_dbm)
                or not np.isfinite((*self.azimuth_deg, *self.power_dbm, self.detection_threshold_dbm,
                                     self.beamwidth_deg, self.bearing_error_deg,
                                     self.calibration_correlation_seconds)).all()
                or any(not 0 <= angle < 360 for angle in self.azimuth_deg)
                or len(set(self.azimuth_deg)) != len(self.azimuth_deg)
                or not 0 < self.beamwidth_deg <= 180 or not 0 <= self.bearing_error_deg <= 45
                or self.calibration_correlation_seconds < 0
                or len(set(self.peak_indices)) != len(self.peak_indices)
                or any(type(index) is not int or not 0 <= index < len(self.power_dbm)
                       for index in self.peak_indices)):
            raise ValueError("invalid directional observation")
        if any(self.power_dbm[index] < self.detection_threshold_dbm for index in self.peak_indices):
            raise ValueError("directional detection below threshold")


def scan_record(measurement):
    scan = measurement.directional_scan
    return {"time_seconds": measurement.time_seconds, "position_m": list(measurement.position_m),
            "peaks_deg_dbm": scan.peaks(), "beamwidth_deg": scan.beamwidth_deg,
            "bearing_error_std_deg": scan.bearing_error_deg,
            "calibration_correlation_seconds": scan.calibration_correlation_seconds,
            "minimum_peak_separation_deg": scan.beamwidth_deg,
            "detection_threshold_dbm": scan.detection_threshold_dbm}


def beam_power_dbm(contributions_dbm, bearings_deg, azimuth_deg, background_dbm,
                   beamwidth_deg, front_back_db, offset_deg=0.0):
    """无噪声接收方向响应；可对候选源计算预测，不能作为观测返回。"""
    azimuth = np.asarray(azimuth_deg, dtype=float)
    power = np.full(azimuth.shape, 10 ** (background_dbm / 10))
    for contribution, bearing in zip(contributions_dbm, bearings_deg):
        delta = (azimuth + offset_deg - bearing + 180) % 360 - 180 if bearing is not None else np.zeros_like(azimuth)
        attenuation = np.minimum(front_back_db, 12 * (delta / beamwidth_deg) ** 2)
        power += 10 ** ((contribution - attenuation) / 10)
    return 10 * np.log10(power)


def angular_peak_indices(azimuth_deg, measured_dbm, threshold_dbm, beamwidth_deg):
    """环形局部峰与主瓣内去重；峰数不是发射源数量。"""
    azimuth, measured = np.asarray(azimuth_deg), np.asarray(measured_dbm)
    peaks = np.flatnonzero((measured >= threshold_dbm) & (measured > np.roll(measured, 1))
                           & (measured >= np.roll(measured, -1)))
    resolved = []
    for index in sorted(peaks.tolist(), key=lambda candidate: (-measured[candidate], candidate)):
        if all(abs((azimuth[index] - azimuth[other] + 180) % 360 - 180) >= beamwidth_deg
               for other in resolved):
            resolved.append(index)
    return tuple(sorted(resolved))


class DirectionalReceiver:
    """同一时刻并行形成全部波束；独立随机流不改变原有点测噪声序列。"""

    def __init__(self, config):
        self.settings = config.sensor
        self.background_dbm = config.electromagnetic.background_dbm
        self.random = np.random.default_rng(np.random.SeedSequence([config.seed, 1701]))
        self.calibration_random = np.random.default_rng(np.random.SeedSequence([config.seed, 1702]))
        self.last_scan_seconds = None
        self.calibration_error_deg = 0.0

    def _calibration_error(self, time_seconds):
        if not np.isfinite(time_seconds) or time_seconds < 0:
            raise ValueError("invalid sensor time")
        if self.last_scan_seconds is not None and time_seconds < self.last_scan_seconds:
            raise ValueError("out-of-order sensor time")
        settings = self.settings
        if self.last_scan_seconds is None or settings.calibration_correlation_seconds == 0:
            self.calibration_error_deg = self.calibration_random.normal(0.0, settings.bearing_error_deg)
        elif time_seconds > self.last_scan_seconds:
            elapsed = time_seconds - self.last_scan_seconds
            correlation = np.exp(-elapsed / settings.calibration_correlation_seconds)
            innovation_std = settings.bearing_error_deg * np.sqrt(
                -np.expm1(-2 * elapsed / settings.calibration_correlation_seconds))
            self.calibration_error_deg = correlation * self.calibration_error_deg + self.calibration_random.normal(
                0.0, innovation_std)
        self.last_scan_seconds = time_seconds
        return self.calibration_error_deg

    def scan(self, contributions_dbm, bearings_deg, time_seconds=0.0) -> DirectionalScan:
        if len(contributions_dbm) != len(bearings_deg):
            raise ValueError("source powers and bearings must have equal lengths")
        settings = self.settings
        azimuth = np.arange(settings.azimuth_bins) * 360.0 / settings.azimuth_bins
        offset = self._calibration_error(time_seconds)
        measured = beam_power_dbm(contributions_dbm, bearings_deg, azimuth, self.background_dbm,
                                  settings.beamwidth_deg, settings.front_back_db, offset)
        measured += self.random.normal(0.0, settings.noise_db, settings.azimuth_bins)
        threshold = max(settings.detection_threshold_dbm, self.background_dbm + settings.minimum_excess_db,
                        float(np.median(measured)) + settings.minimum_contrast_db)
        peaks = angular_peak_indices(azimuth, measured, threshold, settings.beamwidth_deg)
        return DirectionalScan(tuple(azimuth.tolist()), tuple(measured.tolist()), peaks,
                               threshold, settings.beamwidth_deg, settings.bearing_error_deg,
                               settings.calibration_correlation_seconds)
