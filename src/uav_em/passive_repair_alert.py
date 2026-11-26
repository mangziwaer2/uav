"""Causal repair reviews from the forward GP and ordinary point feedback."""

from dataclasses import dataclass
import math

from .passive_event_choice import ALERT_DBM
from .passive_map import SOURCES


@dataclass(frozen=True)
class RepairAlertSettings:
    early_dbm: float
    early_z: float
    worsening_db: float
    clear_samples: int

    def __post_init__(self):
        thresholds = (self.early_dbm, self.early_z, self.worsening_db)
        if (not all(type(value) in (int, float) and math.isfinite(value) for value in thresholds)
                or self.early_dbm >= ALERT_DBM or self.early_z <= 0 or self.worsening_db <= 0
                or type(self.clear_samples) is not int or self.clear_samples < 1):
            raise ValueError("invalid repair alert settings")


def _residual(frame):
    forecast = frame.get("forecast")
    if (forecast is None or forecast["support"] not in SOURCES[:-1]
            or not forecast["issued_s"] < frame["t"] == forecast["arrival_s"]
            or tuple(forecast["xy"]) != tuple(frame["xy"])):
        return None
    mean, std = forecast["dbm"], forecast["observation_std_db"]
    if not (math.isfinite(mean) and math.isfinite(std) and std > 0):
        return None
    value = (frame["measured_dbm"] - mean) / std
    return value if math.isfinite(value) else None


class RepairAlert:
    def __init__(self, settings):
        self.settings = settings
        self.last_review = None
        self.previous_early = False
        self.clear_count = 0
        self.last_sample_s = -math.inf

    def check(self, frame, peak_dbm):
        """Consume one ordinary sample after checking its frozen prediction."""
        clock, measured = frame["t"], frame["measured_dbm"]
        if (not math.isfinite(clock) or clock < 0 or clock <= self.last_sample_s or not math.isfinite(measured)
                or peak_dbm is not None and not math.isfinite(peak_dbm)):
            raise ValueError("expected a new finite sample and a supported peak or None")
        self.last_sample_s = clock
        settings = self.settings
        residual_z = _residual(frame)
        early = (residual_z is not None and residual_z >= settings.early_z and measured >= settings.early_dbm)
        high = peak_dbm is not None and peak_dbm >= ALERT_DBM
        clear = not early and measured < ALERT_DBM and peak_dbm is not None and peak_dbm < ALERT_DBM
        self.clear_count = self.clear_count + 1 if clear else 0
        if self.clear_count >= settings.clear_samples:
            self.last_review = None

        reason = "waiting"
        review = False
        if high or early:
            reason = "persistent"
            if self.last_review is None:
                reason = "lookahead" if high else "positive_residual"
            else:
                old_measured, old_peak, old_high = self.last_review
                if high and not old_high:
                    reason = "lookahead"
                elif early and not self.previous_early:
                    reason = "positive_residual"
                elif (measured - old_measured >= settings.worsening_db
                      or peak_dbm is not None and old_peak is not None and peak_dbm - old_peak >= settings.worsening_db):
                    reason = "worsened"
            review = reason != "persistent"
            if review:
                self.last_review = measured, peak_dbm, high
        self.previous_early = early
        return dict(review=review, reason=reason, measured_dbm=measured, peak_dbm=peak_dbm,
                    lookahead_alert=high, residual_alert=early, residual_z=residual_z)
