"""One paused, public-only decision for a future paired repair flight."""

from dataclasses import dataclass
import math

import numpy as np

from .passive_branch_event import _positions
from .passive_closed_loop import QueryLimit
from .passive_event_choice import ALERT_DBM, HORIZON_S
from .passive_local_repair import LocalRepair, ResolvedRepair, xy
from .passive_map import _common_time
from .passive_pricing_integral import _trajectory
from .passive_repair_actions import FrozenActionTable, choose_numerical, freeze_actions
from .passive_repair_pricing import MAX_ORIGINAL_ROUTES, FrozenRepairPricing, RepairComparison
from .passive_repair_search import build_repair_neighborhood, normalize_route
from .passive_route_exchange import RouteExchange, RouteReceipt, RouteRequest, seal_route_request


INSTRUCTIONS = """Select one frozen legal unflown remainder in a civilian simulation using only this input.
Minimize actual travel time plus normalized linear interference exposure. Routes are [id, XY].
Read action_table.rows with columns; read each action's sources with source_columns. An original
action uses the matching route ID. In a repair source, each edit is [left, right, waypoint_index]:
retain the two vertices of base_id, replace the interior with a direct link or one via the indexed
point in repair_waypoints_xy_m. Null waypoint means direct. Multiple sources can describe one action.
time_s is full remaining time. Q prices use only locked_intervals_s. Window and post-split deltas
are distinct. Null Q is unpriced, not safe or zero exposure, and does not make an action illegal.
Read points as [time, x, y, measured dBm, previous forecast error dB]; larger dBm means stronger interference.
Return one JSON line with request_id, action_id, and reason. Do not use external information,
invent measurements, or abstain."""


class RepairOpportunity:
    def __init__(self, separation_m, *, direction_memory=False):
        if not math.isfinite(separation_m) or separation_m <= 0:
            raise ValueError("positive branch separation required")
        self.separation_m = separation_m
        self.direction_memory = direction_memory
        self._seen = set()

    def check(self, high, incumbent, routes, speed):
        if not high:
            self._seen.clear()
            return "rearmed", ()
        base = _trajectory(incumbent, speed)
        available, compared = set(), set()
        for route in routes:
            if route == incumbent:
                continue
            track = _trajectory(route, speed)
            end = min(HORIZON_S, base[0][-1], track[0][-1])
            offsets = np.union1d(np.r_[end, base[0][base[0] < end]], track[0][track[0] < end])
            distance = np.linalg.norm(_positions(base, offsets) - _positions(track, offsets), axis=1)
            if distance.max() < self.separation_m:
                continue
            split = _common_time(incumbent, route, speed)
            sides = []
            for knots, vertices in (base, track):
                index = np.searchsorted(knots, split + 1e-8, side="right")
                if index >= len(knots):
                    break
                velocity = (vertices[index] - vertices[index - 1]) / (knots[index] - knots[index - 1])
                while index + 1 < len(knots) and np.allclose(
                        velocity,
                        (vertices[index + 1] - vertices[index]) / (knots[index + 1] - knots[index]),
                        atol=1e-8, rtol=0):
                    index += 1
                target = tuple(map(float, vertices[index]))
                key = tuple(velocity / speed) if self.direction_memory else target
                sides.append((target, key))
            if len(sides) == 2:
                compared.update(key for _, key in sides)
                available.add(sides[1])
        if not available:
            return "no_alternative", ()

        same_direction = math.cos(math.radians(10))

        def unseen(key):
            if not self.direction_memory:
                return key not in self._seen
            return all(np.dot(key, seen) < same_direction for seen in self._seen)

        new = tuple(sorted({target for target, key in available if unseen(key)}))
        if not new:
            return "already_reviewed", ()
        state = "new_branch" if self._seen else "reviewed"
        self._seen.update(compared)
        return state, new


@dataclass(frozen=True)
class RepairDecision:
    status: str
    new_targets: tuple
    selected: ResolvedRepair | None = None
    action_id: int | None = None
    reference_id: int | None = None
    edit: LocalRepair | None = None
    routes: tuple = ()
    incumbent_id: int | None = None
    table: FrozenActionTable | None = None
    comparison: RepairComparison | None = None
    evidence_status: str | None = None
    request: RouteRequest | None = None
    receipt: RouteReceipt | None = None


def build_repair_request(table, *, request_id, geometry, aircraft, now, flown_m, energy_wh, peak_dbm,
                         incumbent_id, targets, points):
    packet = dict(request_id=request_id, t_s=now, goal=xy(geometry.config.goal_m),
                  buildings=geometry.config.buildings, flown_m=flown_m, energy_wh=energy_wh,
                  cap_m=1.25 * math.dist(geometry.config.start_m, geometry.config.goal_m),
                  reserve_wh=aircraft.reserve_energy_wh, peak_dbm=peak_dbm, window_s=HORIZON_S,
                  incumbent=incumbent_id, new_first_targets=targets, points=points,
                  flight_geometry=dict(max_xy_m=geometry.high.tolist(), obstacles_m=geometry.config.obstacles_m,
                                       mode=geometry.config.geometry_mode, speed_mps=aircraft.speed_mps,
                                       altitude_m=aircraft.altitude_m, clearance_m=geometry.clearance_m,
                                       vertical_clearance_m=aircraft.vertical_clearance_m), **table.public())
    return seal_route_request(packet, INSTRUCTIONS)


class RepairReview:
    def __init__(self, *, separation_m, min_post_s, direction_memory=False):
        if not 0 < min_post_s <= HORIZON_S:
            raise ValueError("positive post-split support within the lookahead required")
        self._opportunity = RepairOpportunity(separation_m, direction_memory=direction_memory)
        self._exchange = RouteExchange()
        self.min_post_s = min_post_s

    def check(self, incumbent, routes, waypoints, model, *, peak_dbm, arm, geometry, aircraft, position, now,
              flown_m, energy_wh, points=(), request_id=None, transport=None):
        """Use the same legal opportunities in either arm; do not advance or sample the flight."""
        if arm not in ("numerical", "llm"):
            raise ValueError("unknown repair decision arm")
        high = peak_dbm is not None and peak_dbm >= ALERT_DBM
        if not high:
            return RepairDecision(*self._opportunity.check(False, incumbent, (), aircraft.speed_mps))
        incumbent = normalize_route(incumbent)
        routes = tuple(dict.fromkeys(map(normalize_route, routes)))
        if not 1 <= len(routes) <= MAX_ORIGINAL_ROUTES or incumbent not in routes:
            raise ValueError("legal original menu must contain the committed remainder")
        flight = dict(geometry=geometry, aircraft=aircraft, position=position, now=now,
                      flown_m=flown_m, energy_wh=energy_wh)
        neighborhood = build_repair_neighborhood(routes, waypoints, **flight)
        alternatives = (*routes, *(row.route for row, _ in neighborhood.expanded))
        status, targets = self._opportunity.check(True, incumbent, alternatives, aircraft.speed_mps)
        if not targets:
            return RepairDecision(status, targets)
        incumbent_id = routes.index(incumbent)
        common = dict(routes=routes, incumbent_id=incumbent_id)
        try:
            pricing = FrozenRepairPricing(routes, model, now, aircraft.speed_mps, min_post_s=self.min_post_s)
            table = freeze_actions(pricing, neighborhood, incumbent_id)
        except QueryLimit:
            return RepairDecision("compute_budget", targets, **common)
        request, receipt = None, None
        if arm == "numerical":
            action = choose_numerical(table)
        else:
            request = build_repair_request(table, request_id=request_id, geometry=geometry, aircraft=aircraft,
                                           now=now, flown_m=flown_m, energy_wh=energy_wh, peak_dbm=peak_dbm,
                                           incumbent_id=incumbent_id, targets=targets, points=points)
            if transport is None:
                raise ValueError("LLM arm requires a receiving adapter")
            receipt = self._exchange.submit(request, transport)
            if receipt.selection is None:
                return RepairDecision(receipt.status, targets, table=table, request=request, receipt=receipt, **common)
            action = table.rows[receipt.selection.action_id]
        source = action.sources[0]
        edit = source.edits[0] if source.edits else None
        return RepairDecision(status, targets, selected=action.resolved, action_id=action.action_id, table=table,
                              reference_id=action.action_id if edit is None else source.base_id,
                              edit=edit, comparison=source.comparison, evidence_status=action.price.status,
                              request=request, receipt=receipt, **common)
