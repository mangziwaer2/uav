"""Causal GP-BOCPD with the P7 kernel and already legal route menus."""

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import logsumexp

from .passive_map import FixedPriorPosterior
from .passive_pricing_integral import compare_aligned_prices
from .point_gp import validate_points


@dataclass(frozen=True)
class RegimeForecast:
    issued_s: float
    sample_s: float
    xy_m: tuple
    log_weights: tuple
    means_dbm: tuple
    observation_std_db: tuple

    def _log_densities(self, reading_dbm):
        std = np.asarray(self.observation_std_db)
        residual = (reading_dbm - np.asarray(self.means_dbm)) / std
        return -0.5 * (math.log(2 * math.pi) + residual ** 2) - np.log(std)

    def log_predictive(self, sample):
        """Score the frozen forecast only against its matching ordinary point."""
        reading = float(sample["interference_dbm"])
        if (sample["time_seconds"] != self.sample_s or tuple(sample["position_m"]) != self.xy_m
                or not math.isfinite(reading)):
            raise ValueError("sample must match the frozen time and position")
        return float(logsumexp(self.log_weights + self._log_densities(reading)))


@dataclass(frozen=True)
class RegimeBelief:
    now: float
    log_weights: tuple
    posteriors: tuple
    fit_count: int

    def freeze_next(self, xy, sample_s):
        if not math.isfinite(sample_s) or sample_s <= self.now:
            raise ValueError("ordinary sample must be after the checkpoint")
        return _forecast(self.now, xy, sample_s, self.log_weights, self.posteriors)


def _forecast(issued_s, xy, sample_s, log_weights, posteriors):
    means, scales = [], []
    for posterior in posteriors:
        mean, std, _ = posterior.predict([xy], [sample_s])
        means.append(float(mean[0]))
        scales.append(math.hypot(std[0], posterior.settings.measured_error_db))
    return RegimeForecast(issued_s, sample_s, tuple(xy), tuple(log_weights), tuple(means), tuple(scales))


def _segment_models(points, geometry, now, settings):
    return tuple(FixedPriorPosterior(points[len(points) - length:], geometry, now, settings, exact_segment=True)
                 for length in range(len(points) + 1))


def fit_regimes(history, geometry, now, settings, *, change_probability, max_samples):
    """Exact fixed-hazard recursion; run length is the count since the last reset.

    Resets are after observations, so mass at length zero is the hazard, not
    evidence of a detected change. All other lengths retain their own segment.
    """
    if not isinstance(max_samples, int) or not 1 <= max_samples or len(history) > max_samples:
        raise ValueError("point prefix exceeds the fit budget")
    if not 0 < change_probability < 1:
        raise ValueError("change probability must be between zero and one")
    if not math.isfinite(settings.measured_error_db) or settings.measured_error_db <= 0:
        raise ValueError("observation scale must be finite and positive")
    points = validate_points(history, geometry, now)
    times = [point["time_seconds"] for point in points]
    if any(first >= second for first, second in zip(times, times[1:])):
        raise ValueError("ordinary sample times must increase strictly")

    log_reset, log_growth = math.log(change_probability), math.log1p(-change_probability)
    log_weights = np.array([0.0])
    fit_count = 0
    for index, (sample, sample_s) in enumerate(zip(points, times)):
        issued_s = times[index - 1] if index else 0.0
        posteriors = _segment_models(points[:index], geometry, issued_s, settings)
        fit_count += len(posteriors)
        forecast = _forecast(issued_s, sample["position_m"], sample_s, log_weights, posteriors)
        joint = log_weights + forecast._log_densities(sample["interference_dbm"])
        evidence = logsumexp(joint)
        log_weights = np.r_[log_reset, joint - evidence + log_growth]

    posteriors = _segment_models(points, geometry, now, settings)
    return RegimeBelief(now, tuple(log_weights), posteriors, fit_count + len(posteriors))


def price_legal_routes(routes, belief, speed, *, step_s=0.25):
    """Apply the common P7 proxy per regime, conditional on no further reset."""
    comparisons = [compare_aligned_prices(routes, posterior, belief.now, speed, step_s=step_s)
                   for posterior in belief.posteriors]
    world_costs = np.array([[row["costs"]["risk"] for row in result["rows"]] for result in comparisons])
    expected = np.exp(belief.log_weights) @ world_costs
    travel = [row["time_s"] for row in comparisons[0]["rows"]]
    selected = min(range(len(travel)), key=lambda idx: (expected[idx], travel[idx], idx))
    return dict(selected=selected, expected_proxy_tq=expected.tolist(), proxy_by_run_length=world_costs.tolist(),
                queries=sum(result["queries"] for result in comparisons))
