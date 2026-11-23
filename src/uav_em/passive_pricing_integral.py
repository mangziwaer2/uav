"""Shared arrival-time integration for already validated passive-route candidates."""

import math

import numpy as np

from .passive_map import LIMIT_S
from .passive_pricing import POLICIES, UNKNOWN_PRICE, upper_tail_power


STEP_SIZES = (1.0, 0.5, 0.25)
BOUNDARY_BISECTIONS = 22


def _trajectory(route, speed):
    points = np.asarray(route, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("finite XY route required")
    durations = np.linalg.norm(np.diff(points, axis=0), axis=1) / speed
    moving = durations > 0
    if not moving.any():
        raise ValueError("nonzero route required")
    vertices = np.concatenate((points[:1], points[1:][moving]))
    return np.r_[0.0, np.cumsum(durations[moving])], vertices


def _integrate(model, now, trajectory, times):
    knots, vertices = trajectory
    queries = 0

    def predict(offsets):
        nonlocal queries
        points = np.column_stack([np.interp(offsets, knots, column) for column in vertices.T])
        queries += len(offsets)
        return model.predict(points, now + offsets)

    supported = predict(times)[2] < 2
    starts, ends = times[:-1].copy(), times[1:].copy()
    transitions = np.flatnonzero(supported[:-1] != supported[1:])
    if len(transitions):
        lower, upper = starts[transitions], ends[transitions]
        starts_supported = supported[transitions]
        for _ in range(BOUNDARY_BISECTIONS):
            middle = (lower + upper) / 2
            same_side = (predict(middle)[2] < 2) == starts_supported
            lower = np.where(same_side, middle, lower)
            upper = np.where(same_side, upper, middle)
        boundary = (lower + upper) / 2
        ends[transitions[starts_supported]] = boundary[starts_supported]
        starts[transitions[~starts_supported]] = boundary[~starts_supported]

    active = supported[:-1] | supported[1:]
    starts, ends = starts[active], ends[active]
    widths = ends - starts
    duration = float(knots[-1])
    unknown = max(0.0, duration - float(widths.sum()))
    floor = duration + UNKNOWN_PRICE * unknown
    costs = dict(shortest=duration, mean=floor, risk=floor)
    if len(widths):
        nodes = np.stack((starts, (starts + ends) / 2, ends))
        mean, std, _ = predict(nodes.ravel())
        weights = np.array([1, 4, 1])[:, None] * widths / 6
        costs["mean"] += float(np.sum(weights * (10 ** ((mean + 80) / 10)).reshape(nodes.shape)))
        costs["risk"] += float(np.sum(weights * upper_tail_power(mean, std).reshape(nodes.shape)))
    if not all(map(math.isfinite, costs.values())):
        raise ValueError("nonfinite route price")
    return dict(time_s=duration, unknown_s=unknown, costs=costs, crossings=len(transitions), queries=queries)


def compare_aligned_prices(routes, model, now, speed, *, step_s=0.5):
    """Rank the P3 prices on common clocks, then bisect detected support changes.

    Probe each grid midpoint as well, but do not claim to find support islands
    narrower than the probe spacing. Step refinement is required for diagnosis.
    """
    if not 1 <= len(routes) <= 126 or step_s not in STEP_SIZES:
        raise ValueError("route count or time step outside comparison budget")
    if not math.isfinite(now) or now < 0 or not math.isfinite(speed) or speed <= 0:
        raise ValueError("finite current time and positive speed required")
    trajectories = [_trajectory(route, speed) for route in routes]
    durations = np.array([knots[-1] for knots, _ in trajectories])
    if now + durations.max() > LIMIT_S + 1e-8:
        raise ValueError("candidate duration exceeds comparison budget")
    probes = np.arange(0, durations.max(), step_s / 2)
    common = np.union1d(probes, durations)
    rows = [_integrate(model, now, trajectory, common[common <= duration])
            for trajectory, duration in zip(trajectories, durations)]
    order = {policy: sorted(range(len(rows)), key=lambda idx: (rows[idx]["costs"][policy], rows[idx]["time_s"], idx))
             for policy in POLICIES}
    return dict(rows=rows, order=order, selected={mode: order[mode][0] for mode in POLICIES},
                queries=sum(row["queries"] for row in rows))
