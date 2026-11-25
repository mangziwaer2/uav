"""Price an expanded route without letting it change the original comparison window."""

from dataclasses import dataclass
import math

import numpy as np

from .passive_branch_event import _positions
from .passive_event_choice import ALERT_DBM, HORIZON_S
from .passive_local_repair import xy
from .passive_map import _common_time
from .passive_pricing import upper_tail_power
from .passive_pricing_integral import _trajectory


@dataclass(frozen=True)
class LockedPrice:
    time_s: float
    priced_s: float
    unknown_s: float
    outside_s: float
    expected_q: float | None
    tail_q: float | None
    status: str

    @property
    def score(self):
        if self.status == "time_only":
            return self.time_s
        return None if self.tail_q is None else self.time_s + self.tail_q


@dataclass(frozen=True)
class RepairComparison:
    price: LockedPrice
    delta_t_s: float
    split_s: float
    post_common_s: float
    missing_required_s: float
    delta_expected_q: float | None
    delta_tail_q: float | None


@dataclass(frozen=True)
class _Samples:
    durations: np.ndarray
    weights: np.ndarray
    covered: np.ndarray
    expected: np.ndarray
    tail: np.ndarray


class FrozenRepairPricing:
    def __init__(self, routes, model, now, speed, *, min_post_s):
        if (not math.isfinite(now) or now < 0 or getattr(model, "now", now) != now
                or not math.isfinite(speed) or speed <= 0 or not 0 < min_post_s <= HORIZON_S):
            raise ValueError("current posterior, positive speed and post-split requirement needed")
        self.routes = tuple(tuple(map(xy, route)) for route in routes)
        if (not 1 <= len(self.routes) <= 126 or any(len(route) < 2 for route in self.routes)
                or any((route[0], route[-1]) != (self.routes[0][0], self.routes[0][-1]) for route in self.routes)):
            raise ValueError("expected a menu of routes from the current position to one goal")
        self._model, self.now, self.speed, self.min_post_s = model, now, speed, min_post_s
        tracks = [_trajectory(route, speed) for route in self.routes]
        arrivals = [min(track[0][-1], HORIZON_S) for track in tracks]
        self._edges = np.unique(np.r_[np.arange(0, HORIZON_S, 0.5), arrivals, HORIZON_S])
        self._original = original = self._sample(tracks)
        active = original.weights > 0
        self._locked = np.all(~active | original.covered, axis=0)
        status = "baseline" if np.any(active & self._locked) else "time_only"
        self.menu = tuple(self._price(original, index, status) for index in range(len(tracks)))

    @property
    def locked_intervals(self):
        return tuple(zip(self._edges[:-1][self._locked].tolist(), self._edges[1:][self._locked].tolist()))

    def _sample(self, tracks):
        durations = np.asarray([track[0][-1] for track in tracks])
        starts = self._edges[:-1]
        weights = np.maximum(0, np.minimum(durations[:, None], self._edges[1:]) - starts)
        offsets = starts + 0.5 * weights
        active = weights > 0
        points = np.concatenate([_positions(track, offsets[index, active[index]]) for index, track in enumerate(tracks)])
        mean, std, source = self._model.predict(points, self.now + offsets[active])
        support = source < 2
        covered = np.zeros_like(active)
        covered[active] = support
        expected, tail = np.zeros_like(weights), np.zeros_like(weights)
        scale = math.log(10) / 10
        expected[covered] = np.exp(scale * (mean[support] - ALERT_DBM) + 0.5 * (scale * std[support]) ** 2)
        tail[covered] = upper_tail_power(mean[support], std[support])
        if not np.isfinite(expected).all() or not np.isfinite(tail).all():
            raise ValueError("nonfinite supported power")
        return _Samples(durations, weights, covered, expected, tail)

    def _price(self, samples, index, status):
        weights, cover = samples.weights[index], samples.covered[index]
        priced = self._locked & cover
        complete = status in ("baseline", "comparable")
        return LockedPrice(float(samples.durations[index]), float(weights[priced].sum()),
                           float(weights[~cover].sum()), max(0., float(samples.durations[index] - HORIZON_S)),
                           float(weights[priced] @ samples.expected[index, priced]) if complete else None,
                           float(weights[priced] @ samples.tail[index, priced]) if complete else None, status)

    def compare(self, route, reference_id):
        return self.compare_references(route, (reference_id,))[0]

    def compare_references(self, route, reference_ids):
        """Evaluate one geometric path once even when several splices generate it."""
        reference_ids = tuple(reference_ids)
        if not reference_ids or any(type(index) is not int or not 0 <= index < len(self.routes)
                                    for index in reference_ids):
            raise ValueError("repair reference must be a frozen menu route")
        route = tuple(map(xy, route))
        reference = self.routes[reference_ids[0]]
        if len(route) < 2 or (route[0], route[-1]) != (reference[0], reference[-1]):
            raise ValueError("repair must keep the current position and goal")
        samples = self._sample([_trajectory(route, self.speed)])
        return tuple(self._compare(route, samples, index) for index in reference_ids)

    def _compare(self, route, samples, reference_id):
        weights, cover = samples.weights[0], samples.covered[0]
        missing = float(weights[self._locked & ~cover].sum())
        split_s = _common_time(self.routes[reference_id], route, self.speed)
        post = self._locked & (self._edges[:-1] >= split_s - 1e-8) & ((weights == 0) | cover)
        post_common = float(np.maximum(weights, self._original.weights[reference_id])[post].sum())
        original = self.menu[reference_id]
        if missing > 0:
            status = "missing_support"
        elif original.status == "time_only":
            status = "no_baseline_evidence"
        elif post_common + 1e-8 < self.min_post_s:
            status = "insufficient_post_support"
        else:
            status = "comparable"
        price = self._price(samples, 0, status)
        return RepairComparison(price, price.time_s - original.time_s, split_s, post_common, missing,
                                price.expected_q - original.expected_q if status == "comparable" else None,
                                price.tail_q - original.tail_q if status == "comparable" else None)
