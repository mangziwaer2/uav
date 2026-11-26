"""Generate a bounded menu of public-geometry remainders at the current position."""

from itertools import combinations, product
import math

import numpy as np
from scipy.sparse.csgraph import dijkstra

from .passive_event_choice import HORIZON_S, menu
from .passive_local_repair import check_remaining_route, xy
from .passive_repair_pricing import MAX_ORIGINAL_ROUTES
from .passive_repair_search import normalize_route


ONLINE_DOMAIN = "online_geometry_v1"
FIXED_DOMAIN = "fixed_gate_v1"
FORWARD_FRACTIONS = (0.5, 1.)
LATERAL_FRACTIONS = (-1 / 3, -1 / 6, 1 / 6, 1 / 3)
CORNER_MARGIN_M = 1.0


class OnlineRepairGeometry:
    """Cache obstacle visibility; calculate first moves and their goal suffixes online."""

    def __init__(self, geometry, aircraft):
        self.geometry, self.aircraft = geometry, aircraft
        self.goal = xy(geometry.config.goal_m)
        corners = {point for low, high in (*geometry.boxes, *geometry.building_bounds)
                   for point in product(*((float(lo) - CORNER_MARGIN_M, float(hi) + CORNER_MARGIN_M)
                                          for lo, hi in zip(low, high)))
                   if point != self.goal and geometry.free(point)}
        if len(corners) + len(FORWARD_FRACTIONS) * len(LATERAL_FRACTIONS) + 2 > MAX_ORIGINAL_ROUTES:
            raise ValueError("public obstacle graph exceeds the online menu domain")
        self.nodes = (self.goal, *sorted(corners))
        edges = np.zeros((len(self.nodes), len(self.nodes)))
        for first, second in combinations(range(len(self.nodes)), 2):
            start, end = self.nodes[first], self.nodes[second]
            if geometry.edge_free(start, end):
                edges[first, second] = edges[second, first] = math.dist(start, end)
        self.distances, predecessors = dijkstra(edges, directed=False, indices=0, return_predecessors=True)
        self.suffixes = {self.goal: ()}
        for index, distance in enumerate(self.distances):
            if index and math.isfinite(distance):
                route = []
                node = index
                while node:
                    node = int(predecessors[node])
                    route.append(self.nodes[node])
                self.suffixes[self.nodes[index]] = tuple(route)

    def _suffix(self, start):
        if start in self.suffixes:
            return self.suffixes[start]
        if self.geometry.edge_free(start, self.goal):
            return (self.goal,)
        best = min(((math.dist(start, self.nodes[index]) + distance, index)
                    for index, distance in enumerate(self.distances) if index and math.isfinite(distance)
                    and self.geometry.edge_free(start, self.nodes[index])), default=None)
        if best is None:
            return None
        corner = self.nodes[best[1]]
        return (corner, *self.suffixes[corner])

    def goal_route(self, start):
        start = xy(start)
        suffix = self._suffix(start) if self.geometry.free(start) else None
        return None if suffix is None else (start, *suffix)

    def _local_guides(self, position, incumbent):
        target = incumbent[1] if incumbent else self.goal
        dx, dy = (target[0] - position[0], target[1] - position[1])
        norm = math.hypot(dx, dy)
        forward = (dx / norm, dy / norm)
        horizon = min(self.aircraft.speed_mps * HORIZON_S, math.dist(position, self.goal))
        for fraction, lateral in product(FORWARD_FRACTIONS, LATERAL_FRACTIONS):
            lead, offset = fraction * horizon, lateral * horizon
            yield (position[0] + lead * forward[0] - offset * forward[1],
                   position[1] + lead * forward[1] + offset * forward[0])

    def guides(self, position, incumbent):
        return tuple(dict.fromkeys((*self.nodes[1:], *self._local_guides(position, incumbent))))

    def routes(self, position, incumbent, *, now, flown_m, energy_wh):
        """Keep the commitment, shortest visible route and each legal first-move completion."""
        position, incumbent = xy(position), normalize_route(incumbent)
        if position == self.goal:
            return ()
        if incumbent and (len(incumbent) < 2 or incumbent[0] != position or incumbent[-1] != self.goal):
            raise ValueError("expected a commitment from the current position to the goal")
        state = dict(geometry=self.geometry, aircraft=self.aircraft, now=now, flown_m=flown_m, energy_wh=energy_wh)
        accepted = {}

        def add(route):
            if route not in accepted and len(set(route)) == len(route) and check_remaining_route(route, **state).status == "legal":
                accepted[route] = None

        if incumbent:
            add(incumbent)
            if incumbent not in accepted:
                raise ValueError("committed route failed independent verification")
        suffix = self._suffix(position)
        if suffix is not None:
            add((position, *suffix))
        for point in self.guides(position, incumbent):
            if point in (position, self.goal) or not self.geometry.edge_free(position, point):
                continue
            suffix = self._suffix(point)
            if suffix is not None:
                add((position, point, *suffix))
        return tuple(accepted)


def repair_domain(flight, online, waypoints):
    if online is None:
        return menu(flight)[0], waypoints
    return online.routes(flight.position, flight.incumbent, now=flight.now, flown_m=flight.flown,
                         energy_wh=flight.energy), ()
