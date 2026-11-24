"""Public-only estimation and bounded route comparison for P1."""

import itertools
import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from .planning import path_length
from .point_gp import space_time_kernel, training_points, validate_points


MODES = ("local", "full")
SOURCES = ("local", "extended", "unknown")
GATES = (240, 400, 620)
LEVELS = (180, 240, 300, 360, 420)
LIMIT_S = 90.0
PRIOR_DBM = -90.0
MIN_REDUCTION = 0.10


class FixedPriorPosterior:
    """One frozen causal posterior. Distant receivers revert to a fixed prior."""

    def __init__(self, history, geometry, now, settings, *, exact_segment=False):
        points = (validate_points(history, geometry, now) if exact_segment else
                  training_points(history, geometry, now, settings))
        self.now, self.settings = now, settings
        self.xy = np.array([point["position_m"] for point in points], float).reshape(-1, 2)
        self.times = np.array([point["time_seconds"] for point in points], float)
        self.prior_variance = settings.unknown_error_db ** 2
        if points:
            gram = space_time_kernel(self.xy, self.times, self.xy, self.times, settings)
            gram.flat[::len(gram) + 1] += settings.measured_error_db ** 2
            self._factor = cho_factor(gram, lower=True)
            values = np.array([point["interference_dbm"] for point in points])
            self._weights = cho_solve(self._factor, values - PRIOR_DBM)

    def predict(self, positions, times):
        points, clocks = np.asarray(positions, float), np.asarray(times, float)
        if (points.ndim != 2 or points.shape[1] != 2 or clocks.shape != (len(points),)
                or not np.isfinite(points).all() or not np.isfinite(clocks).all() or (clocks < self.now).any()):
            raise ValueError("finite current/future XY and times required")
        mean = np.full(len(points), PRIOR_DBM)
        reduction = np.zeros(len(points))
        source = np.full(len(points), 2, dtype=np.uint8)
        if len(self.xy) and len(points):
            covariance = space_time_kernel(self.xy, self.times, points, clocks, self.settings)
            mean += covariance.T @ self._weights
            whitened = solve_triangular(self._factor[0], covariance, lower=True)
            reduction = np.clip(np.sum(whitened ** 2, axis=0) / self.prior_variance, 0, 1)
            distance = np.linalg.norm(self.xy[:, None, :] - points[None, :, :], axis=-1)
            age = clocks[None, :] - self.times[:, None]
            near = ((distance <= self.settings.support_radius_m)
                    & (age <= self.settings.max_age_seconds)).any(axis=0)
            supported = reduction >= MIN_REDUCTION
            source[supported] = 1
            source[supported & near] = 0
        return mean, np.sqrt((1 - reduction) * self.prior_variance), source


def candidates(position, goal, incumbent, geometry, aircraft, now, flown, energy, cap):
    """Enumerate the full shared menu subject to continuous geometry and resources."""
    gates = [gate for gate in GATES if gate > position[0] + 1e-8]
    paths = ([position, *map(list, zip(gates, levels)), goal]
             for levels in itertools.product(LEVELS, repeat=len(gates)))
    legal, seen, edges = [], set(), {}
    for route in itertools.chain(paths, (incumbent,) if incumbent else ()):
        key = tuple(map(tuple, route))
        if key in seen or route[0] != position or route[-1] != goal:
            continue
        seen.add(key)
        distance = path_length(route)
        duration = distance / aircraft.speed_mps
        if (distance + flown > cap + 1e-8 or now + duration > LIMIT_S + 1e-8
                or energy - duration * aircraft.flight_power_w / 3600 < aircraft.reserve_energy_wh):
            continue
        for first, second in zip(key, key[1:]):
            edge = first, second
            if edge not in edges:
                edges[edge] = geometry.edge_free(first, second)
            if not edges[edge]:
                break
        else:
            legal.append(route)
    return legal


def quadrature(route, now, speed):
    points, clocks, weights = [], [], []
    for first, second in zip(route, route[1:]):
        duration = math.dist(first, second) / speed
        if duration == 0:
            continue
        count = math.ceil(duration)
        phase = (np.arange(count) + 0.5) / count
        start, end = np.asarray(first), np.asarray(second)
        points.extend(start + phase[:, None] * (end - start))
        clocks.extend(now + phase * duration)
        weights.extend([duration / count] * count)
        now += duration
    if not 0 < len(clocks) <= 96:
        raise ValueError("candidate quadrature outside frozen budget")
    return np.asarray(points), np.asarray(clocks), np.asarray(weights)


def _coverage(weights, sources):
    return {label: float(weights[sources == code].sum()) for code, label in enumerate(SOURCES)}


def _common_time(first, second, speed):
    def segments(route):
        for vector in np.diff(route, axis=0):
            length = np.linalg.norm(vector)
            if length > 1e-8:
                yield vector / length, length

    left, right = segments(first), segments(second)
    direction_l, length_l = next(left, (None, 0))
    direction_r, length_r = next(right, (None, 0))
    distance = 0.0
    while direction_l is not None and direction_r is not None and np.allclose(direction_l, direction_r, atol=1e-8, rtol=0):
        step = min(length_l, length_r)
        distance += step
        length_l -= step
        length_r -= step
        if length_l <= 1e-8:
            direction_l, length_l = next(left, (None, 0))
        if length_r <= 1e-8:
            direction_r, length_r = next(right, (None, 0))
    return distance / speed


def compare_routes(routes, model, now, speed):
    """Use the same frozen predictors, samples, and exact finite-domain search."""
    samples = [quadrature(route, now, speed) for route in routes]
    if not samples:
        raise ValueError("no legal route to score")
    points, clocks, weights = (np.concatenate(column) for column in zip(*samples))
    mean, _, source = model.predict(points, clocks)
    power = 10 ** ((mean + 80) / 10)
    stops = np.cumsum([len(sample[1]) for sample in samples])
    chunks = [slice(start, stop) for start, stop in zip([0, *stops[:-1]], stops)]
    rows = []
    for route, chunk in zip(routes, chunks):
        dt = weights[chunk]
        costs = {mode: float(dt @ (1 + np.where(source[chunk] <= limit, power[chunk], 1)))
                 for limit, mode in enumerate(MODES)}
        rows.append(dict(route=route, costs=costs, time_s=float(dt.sum()), coverage=_coverage(dt, source[chunk])))
    selected = {mode: min(range(len(rows)), key=lambda idx: (rows[idx]["costs"][mode], rows[idx]["time_s"], idx))
                for mode in MODES}
    first, second = (routes[selected[mode]] for mode in MODES)
    common_until = now + _common_time(first, second, speed)
    split = {}
    for mode, idx in selected.items():
        chunk = chunks[idx]
        start = clocks[chunk] - weights[chunk] / 2
        end = clocks[chunk] + weights[chunk] / 2
        dt = np.maximum(0, end - np.maximum(start, common_until))
        split[mode] = _coverage(dt, source[chunk])
    return dict(candidates=rows, selected=selected, common_until_s=common_until, split_coverage=split)
