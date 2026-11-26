"""Bounded minimum-cost search on a finite public visibility graph."""

from dataclasses import dataclass
from functools import cache
from heapq import heappop, heappush
from itertools import count, product
import math

import numpy as np

from .passive_closed_loop import QueryLimit
from .passive_local_repair import check_remaining_route, xy
from .passive_map import LIMIT_S


QUADRATURE_STEP_S = 0.5


@dataclass(frozen=True)
class RouteCost:
    route: tuple
    time_s: float
    supported_expected_q: float
    unknown_s: float

    @property
    def lower_tq(self):
        return self.time_s + self.supported_expected_q

    @property
    def expected_tq(self):
        return self.lower_tq if self.unknown_s == 0 else None


@dataclass(frozen=True)
class RouteOptimization:
    status: str
    incumbent: tuple
    best_priced: RouteCost | None
    best_unpriced_bound: RouteCost | None
    lower_tq: float
    expansions: int
    geometry_checks: int

    @property
    def upper_tq(self):
        return None if self.best_priced is None else self.best_priced.expected_tq

    @property
    def gap_tq(self):
        return None if self.upper_tq is None else self.upper_tq - self.lower_tq


def _grid(start, goal, reach_m, high, spacing_m, limit):
    ranges = []
    for origin, target, maximum in zip(start, goal, high):
        low = max(0., (origin + target - reach_m) / 2)
        upper = min(float(maximum), (origin + target + reach_m) / 2)
        low_index, high_index = low / spacing_m, upper / spacing_m
        if not math.isfinite(high_index):
            return None
        first, last = math.ceil(low_index), math.floor(high_index)
        ranges.append((first, max(0, last - first + 1)))
    if math.prod(size for _, size in ranges) > limit:
        return None
    if any(size == 0 for _, size in ranges):
        return ()
    return ((east * spacing_m, north * spacing_m)
            for east, north in product(*(range(first, first + size) for first, size in ranges)))


def _edge_exposure(model, start, end, arrival, duration):
    samples = math.ceil(duration / QUADRATURE_STEP_S)
    phase = (np.arange(samples) + 0.5) / samples
    points = np.asarray(start) + phase[:, None] * (np.asarray(end) - start)
    mean, std, support = model.predict(points, arrival + phase * duration)
    covered = support < 2
    scale = math.log(10) / 10
    expected = np.exp(scale * (mean[covered] + 80) + 0.5 * (scale * std[covered]) ** 2)
    if not np.isfinite(expected).all():
        raise ValueError("nonfinite supported power estimate")
    width = duration / samples
    return float(expected.sum() * width), float(np.count_nonzero(~covered) * width)


def optimize_remainder(planner, incumbent, model, *, now, flown_m, energy_wh, spacing_m,
                       max_grid_cells, max_expansions):
    """Use a frozen, budget-counted predictor and return bounds over simple graph paths.

    A missing power estimate contributes no priced exposure to the lower bound.
    Its duration is recorded, and that route cannot supply an upper bound.
    """
    route = tuple(map(xy, incumbent))
    goal = planner.goal
    if (len(route) < 2 or route[-1] != goal or len(set(route)) != len(route) or model.now != now
            or not all(math.isfinite(value) and value >= 0 for value in (now, flown_m, energy_wh))
            or not math.isfinite(spacing_m) or spacing_m <= 0
            or any(type(value) is not int or value < 0 for value in (max_grid_cells, max_expansions))):
        raise ValueError("expected a frozen checkpoint, simple commitment and finite search bounds")
    geometry, aircraft = planner.geometry, planner.aircraft
    state = dict(geometry=geometry, aircraft=aircraft, now=now, flown_m=flown_m, energy_wh=energy_wh)
    if check_remaining_route(route, **state).status != "legal":
        raise ValueError("committed route fails independent verification")

    start, speed = route[0], aircraft.speed_mps
    straight_s = math.dist(start, goal) / speed
    reach_m = min(1.25 * math.dist(geometry.config.start_m, goal) - flown_m + 1e-8,
                  (LIMIT_S - now + 1e-8) * speed)
    grid = _grid(start, goal, reach_m, geometry.high, spacing_m, max_grid_cells)
    if grid is None:
        return RouteOptimization("grid_budget", route, None, None, straight_s, 0, 0)
    nodes = set((*route, *planner.nodes))
    nodes.update(point for point in grid
                 if math.dist(start, point) + math.dist(point, goal) <= reach_m and geometry.free(point))
    to_goal = {point: math.dist(point, goal) for point in nodes}
    order = sorted(nodes, key=lambda point: (to_goal[point], point))

    @cache
    def clear_edge(first, second):
        return geometry.edge_free(first, second)

    @cache
    def price_edge(first, second, distance_m):
        return _edge_exposure(model, first, second, now + distance_m / speed, math.dist(first, second) / speed)

    tickets = count()
    frontier = [(straight_s, next(tickets), (start,), 0., 0., 0.)]
    best_priced = best_unpriced = None
    best_cost = unpriced_bound = interrupted_bound = math.inf
    expansions = 0
    stop = None
    while frontier and frontier[0][0] < best_cost:
        if expansions >= max_expansions:
            stop = "expansion_budget"
            break
        bound, _, prefix, distance_m, supported_q, unknown_s = heappop(frontier)
        head = prefix[-1]
        expansions += 1
        try:
            for point in order:
                if point in prefix:
                    continue
                distance = distance_m + math.dist(head, point)
                total_m = distance + to_goal[point]
                if total_m > reach_m or total_m / speed + supported_q >= best_cost:
                    continue
                if not clear_edge(*sorted((head, point))):
                    continue
                path = (*prefix, point)
                if point == goal and check_remaining_route(path, **state).status != "legal":
                    continue
                edge_q, edge_unknown = price_edge(head, point, distance_m)
                priced_q, unpriced_s = supported_q + edge_q, unknown_s + edge_unknown
                lower = total_m / speed + priced_q
                if point != goal and lower < best_cost:
                    heappush(frontier, (lower, next(tickets), path, distance, priced_q, unpriced_s))
                elif point == goal:
                    estimate = RouteCost(path, distance / speed, priced_q, unpriced_s)
                    if estimate.expected_tq is not None and lower < best_cost:
                        best_priced, best_cost = estimate, lower
                    elif estimate.expected_tq is None and lower < unpriced_bound:
                        best_unpriced, unpriced_bound = estimate, lower
        except QueryLimit:
            interrupted_bound, stop = bound, "query_budget"
            break

    pending_bound = frontier[0][0] if frontier else math.inf
    lower_tq = min(best_cost, unpriced_bound, pending_bound, interrupted_bound)
    if best_priced is not None and best_cost <= lower_tq:
        status = "optimal"
    else:
        status = stop or "unpriced"
    return RouteOptimization(status, route, best_priced, best_unpriced, lower_tq, expansions,
                             clear_edge.cache_info().misses)
