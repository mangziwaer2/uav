"""仅依赖接收观测的提前复核事件，不使用隐藏干扰区或未来脚本。"""

from collections import deque


class ObservationMonitor:
    """将持续增强和首次测向发现与单次innovation区分；仍受最小复核间隔约束。"""

    def __init__(self, config):
        self.config = config
        self.recent = deque(maxlen=128)
        self.review_power_dbm = None
        self.review_directional_peak_dbm = None

    def observe(self, measurement) -> dict:
        settings = self.config.simulation
        self.recent.append(measurement)
        while self.recent and measurement.time_seconds - self.recent[0].time_seconds > settings.warning_window_seconds:
            self.recent.popleft()
        rise = measurement.interference_dbm - min(item.interference_dbm for item in self.recent)
        scan = measurement.directional_scan
        peak = max((power for _, power in scan.peaks()), default=None) if scan else None
        reasons = []
        if self.review_power_dbm is not None:
            if peak is not None and self.review_directional_peak_dbm is None:
                reasons.append("direction_detected")
            if (peak is not None and self.review_directional_peak_dbm is not None
                    and peak - self.review_directional_peak_dbm >= settings.warning_rise_db):
                reasons.append("directional_power_rising")
            if (len(self.recent) >= 3 and rise >= settings.warning_rise_db
                    and measurement.interference_dbm >= self.config.sensor.detection_threshold_dbm
                    and measurement.interference_dbm - self.review_power_dbm >= settings.warning_rise_db):
                reasons.append("power_rising")
            reference = self.config.electromagnetic.interference_reference_dbm
            if self.review_power_dbm < reference <= measurement.interference_dbm:
                reasons.append("reference_crossed")
        return {"reasons": reasons, "window_seconds": settings.warning_window_seconds,
                "power_rise_db": round(rise, 3), "detected_peak_dbm": peak,
                "review_power_dbm": self.review_power_dbm,
                "meaning": "observations only; warning requests review, not a safety verdict"}

    def reviewed(self, measurement):
        self.review_power_dbm = measurement.interference_dbm
        scan = measurement.directional_scan
        self.review_directional_peak_dbm = max((power for _, power in scan.peaks()), default=None) if scan else None
