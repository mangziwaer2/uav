"""Build shared, causal evidence for already validated passive-route candidates."""

import math

import numpy as np

from .passive_map import SOURCES, _common_time
from .passive_pricing import UNKNOWN_PRICE
from .passive_pricing_integral import _integrate, _trajectory, compare_aligned_prices


TIME_TOL_S = 1e-8
STEP_S = 0.25


def _xy(value):
    xy = [float(coordinate) for coordinate in value]
    if len(xy) != 2 or not all(map(math.isfinite, xy)):
        raise ValueError("expected finite XY coordinates")
    return xy


def _point_evidence(measurements, forecasts, now):
    points, by_time = [], {}
    previous = -math.inf
    for sample in measurements:
        time, power = float(sample["time_seconds"]), float(sample["interference_dbm"])
        if not (0 <= time <= now and previous < time and math.isfinite(power)):
            raise ValueError("measurements must form a finite chronological prefix")
        by_time[time] = len(points)
        points.append(dict(t_s=time, xy_m=_xy(sample["position_m"]), measured_dbm=power))
        previous = time

    residuals = []
    last_arrival = -math.inf
    for forecast in forecasts:
        issued, arrival = float(forecast["issued_s"]), float(forecast["arrival_s"])
        mean, std = float(forecast["dbm"]), float(forecast["observation_std_db"])
        support = forecast["support"]
        index = by_time.get(arrival)
        if (index is None or not 0 <= issued < arrival or arrival <= last_arrival
                or _xy(forecast["xy"]) != points[index]["xy_m"]
                or not math.isfinite(mean) or not math.isfinite(std) or std <= 0 or support not in SOURCES):
            raise ValueError("expected a prior forecast matching an observed point")
        error = points[index]["measured_dbm"] - mean
        residuals.append(dict(point_idx=index, issued_s=issued, predicted_dbm=mean, observation_std_db=std,
                              error_db=error, standardized_error=error / std, support=support))
        last_arrival = arrival
    return points, residuals


def _positions(track, offsets):
    knots, vertices = track
    return np.column_stack([np.interp(offsets, knots, axis) for axis in vertices.T])


class _SharedSupport:
    def __init__(self, posterior, other, now):
        self.posterior, self.other, self.now = posterior, other, now

    def predict(self, xy, times):
        mean, std, support = self.posterior.predict(xy, times)
        other_support = self.posterior.predict(_positions(self.other, times - self.now), times)[2]
        return mean, std, np.where(other_support < 2, support, 2)


def _tail_evidence(track, other, posterior, now, fork, grid):
    duration, overlap = track[0][-1], min(track[0][-1], other[0][-1])
    tail = np.r_[fork, grid[(grid > fork) & (grid <= duration)]]
    own = _integrate(posterior, now, track, tail)
    shared_grid = tail[tail <= overlap]
    shared = _integrate(_SharedSupport(posterior, other, now), now, track, shared_grid)
    total_supported = max(0.0, duration - own["unknown_s"])
    jointly_supported = max(0.0, duration - shared["unknown_s"])
    return dict(unknown_s=max(0.0, duration - fork - total_supported),
                joint_s=jointly_supported, own_only_s=max(0.0, total_supported - jointly_supported),
                joint_premium_tq=shared["costs"]["risk"] - (1 + UNKNOWN_PRICE) * duration,
                own_only_premium_tq=own["costs"]["risk"] - shared["costs"]["risk"])


def build_decision_evidence(routes, incumbent_idx, posterior, now, speed, *,
                            measurements, frozen_forecasts, next_sample_s):
    """Describe one frozen posterior; never generate routes, sample, or select an action."""
    if not 0 <= incumbent_idx < len(routes) or posterior.now != now:
        raise ValueError("current posterior and incumbent index required")
    if next_sample_s is not None and not (math.isfinite(next_sample_s) and now < next_sample_s):
        raise ValueError("ordinary sample time must be in the future")
    points, residuals = _point_evidence(measurements, frozen_forecasts, now)
    prices = compare_aligned_prices(routes, posterior, now, speed, step_s=STEP_S)["rows"]
    tracks = [_trajectory(route, speed) for route in routes]
    durations = [float(track[0][-1]) for track in tracks]
    grid = np.union1d(np.arange(0, max(durations), STEP_S / 2), durations)
    base, base_duration, base_price = tracks[incumbent_idx], durations[incumbent_idx], prices[incumbent_idx]
    if any(not np.array_equal(track[1][0], base[1][0]) or not np.array_equal(track[1][-1], base[1][-1])
           for track in tracks):
        raise ValueError("candidates must share start and goal")
    if points and points[-1]["t_s"] == now and points[-1]["xy_m"] != base[1][0].tolist():
        raise ValueError("candidate start does not match the current observation")
    rows, next_prediction = [], None
    for index, (track, duration, price) in enumerate(zip(tracks, durations, prices)):
        knots, vertices = track
        row = dict(idx=index, route_xy_m=vertices.tolist(), waypoint_arrival_s=(now + knots).tolist(),
                   travel_s=duration, unknown_s=price["unknown_s"], risk_proxy_tq=price["costs"]["risk"],
                   comparison=None)
        rows.append(row)
        if index == incumbent_idx:
            continue
        fork = min(_common_time(routes[index], routes[incumbent_idx], speed), duration, base_duration)
        if fork >= max(duration, base_duration) - TIME_TOL_S:
            continue
        fork_s = now + fork
        if next_sample_s is not None and next_sample_s < fork_s - TIME_TOL_S:
            if next_prediction is None:
                position = _positions(base, [next_sample_s - now])
                mean, std, support = posterior.predict(position, [next_sample_s])
                next_prediction = dict(t_s=next_sample_s, xy_m=position[0].tolist(), predicted_dbm=float(mean[0]),
                                       latent_std_db=float(std[0]), support=SOURCES[support[0]])
            sample = dict(next_prediction, lead_to_fork_s=fork_s - next_sample_s)
        else:
            sample = None
        trial = _tail_evidence(track, base, posterior, now, fork, grid)
        incumbent = _tail_evidence(base, track, posterior, now, fork, grid)
        time_delta = duration - base_duration
        joint_delta = trial["joint_premium_tq"] - incumbent["joint_premium_tq"]
        row["comparison"] = dict(fork_s=fork_s, next_ordinary_point_before_fork=sample,
                                 candidate_after_fork=trial, incumbent_after_fork=incumbent,
                                 delta_time_s=time_delta,
                                 delta_aligned_proxy_tq=price["costs"]["risk"] - base_price["costs"]["risk"],
                                 delta_joint_premium_tq=joint_delta,
                                 delta_one_side_premium_tq=trial["own_only_premium_tq"]
                                 - incumbent["own_only_premium_tq"],
                                 delta_without_one_side_tq=(1 + UNKNOWN_PRICE) * time_delta + joint_delta)
    return dict(t_s=now, incumbent_idx=incumbent_idx, points=points, forecast_residuals=residuals, routes=rows)
