"""Public operator workspace for one paused route decision."""

from dataclasses import asdict
import json
import math

from .passive_closed_loop import QueryLimit
from .passive_event_choice import HORIZON_S
from .passive_local_repair import check_remaining_route, xy
from .passive_map import LIMIT_S, _common_time
from .passive_repair_pricing import FrozenRepairPricing
from .passive_repair_search import normalize_route
from .passive_route_exchange import RouteRequest, ToolCall, _sha256, _unique_keys
from .planning import path_length


TOOL_DOMAIN = "llm_tools_v1"
INSTRUCTIONS = """Plan the unflown remainder of a civilian simulation. You control route generation,
comparisons and final choice. Only the point prefix is measured. Larger dBm means stronger
interference. Minimize actual flight time plus normalized linear power exposure. Q estimates
only cover locked time slices for the given reference; null is not zero or illegal. No new
measurement occurs during this review. Route 0 is the existing commitment.
Return one JSON line: {"request_id": ..., "tool": ..., "args": {...}, "reason": ...}.
Tools and argument keys:
generate: {"operator":"via","waypoint_xy_m":[x,y]} or
          {"operator":"offset","forward_m":number,"lateral_m":number} (left is positive).
          Optional extra_m limits added distance above shortest_known_m, not the hard cap.
compare: {"route_id":int,"reference_id":int}, distinct registered routes.
commit: {"route_id":int}, explicitly finalize one registered legal remainder.
Use tool failures to revise the next proposal; do not create measurements or request a hard cap change."""


def _finite(number):
    if type(number) not in (int, float) or not math.isfinite(number):
        raise ValueError("expected a finite number")
    return float(number)


def _reject_nonfinite(token):
    raise ValueError(f"nonfinite JSON constant: {token}")


def _parse_float(token):
    return _finite(float(token))


def _parse_call(raw, request):
    line = raw.decode("utf-8").removesuffix("\n").removesuffix("\r")
    if "\n" in line or "\r" in line:
        raise ValueError("expected one tool call line")
    reply = json.loads(line, object_pairs_hook=_unique_keys, parse_constant=_reject_nonfinite, parse_float=_parse_float)
    if (not isinstance(reply, dict) or reply.keys() != {"request_id", "tool", "args", "reason"}
            or reply["request_id"] != request.request_id or reply["tool"] not in ("generate", "compare", "commit")
            or not isinstance(reply["args"], dict) or not isinstance(reply["reason"], str) or not reply["reason"].strip()):
        raise ValueError("expected an explicit tool call bound to this request")
    return ToolCall(reply["tool"], reply["args"], reply["reason"])


class RepairToolWorkspace:
    def __init__(self, planner, incumbent, posterior, *, now, flown_m, energy_wh, points, peak_dbm,
                 min_post_s, ledger, geometry_limit):
        geometry, aircraft = planner.geometry, planner.aircraft
        incumbent = normalize_route(incumbent)
        if (len(incumbent) < 2 or incumbent[-1] != planner.goal or len(incumbent) != len(set(incumbent))
                or not all(_finite(value) >= 0 for value in (now, flown_m, energy_wh))
                or posterior.now != now or not 0 < min_post_s <= HORIZON_S
                or type(geometry_limit) is not int or geometry_limit <= 0):
            raise ValueError("invalid public tool checkpoint")
        rows = []
        for clock, east, north, power, error in points:
            if not 0 <= _finite(clock) <= now or (rows and clock <= rows[-1][0]):
                raise ValueError("only a causal measurement prefix may be sent")
            rows.append((clock, *xy((east, north)), _finite(power), None if error is None else _finite(error)))
        if not rows or rows[-1][:3] != (now, *incumbent[0]):
            raise ValueError("current position measurement required")

        self.planner, self.posterior = planner, posterior
        self.ledger, self.geometry_limit, self.min_post_s = ledger, geometry_limit, min_post_s
        self.state = dict(geometry=geometry, aircraft=aircraft, now=now, flown_m=flown_m, energy_wh=energy_wh)
        baseline = check_remaining_route(incumbent, **self.state)
        if baseline.status != "legal":
            raise ValueError("committed route fails independent verification")
        self.routes = [baseline]
        self.pricing, self.comparisons, self.turns = {}, {}, []
        hard_cap = 1.25 * math.dist(geometry.config.start_m, planner.goal)
        self.shortest_route = planner.goal_route(incumbent[0])
        self.shortest_m = (min(baseline.distance_m, path_length(self.shortest_route))
                           if self.shortest_route else baseline.distance_m)
        self.hard_left_m = hard_cap - flown_m
        self.public = dict(
            t_s=now, xy_m=incumbent[0], goal_xy_m=planner.goal, flown_m=flown_m, cap_m=hard_cap,
            hard_left_m=self.hard_left_m, shortest_known_m=self.shortest_m, energy_wh=energy_wh,
            reserve_wh=aircraft.reserve_energy_wh, flight_power_w=aircraft.flight_power_w, remaining_time_s=LIMIT_S - now,
            peak_dbm=_finite(peak_dbm), window_s=HORIZON_S, points=rows,
            geometry=dict(max_xy_m=geometry.high.tolist(), buildings=geometry.config.buildings,
                          obstacles_m=geometry.config.obstacles_m, speed_mps=aircraft.speed_mps,
                          mode=geometry.config.geometry_mode, altitude_m=aircraft.altitude_m,
                          clearance_m=geometry.clearance_m, vertical_clearance_m=aircraft.vertical_clearance_m))

    def _request(self, checkpoint_id, feedback):
        request_id = f"{checkpoint_id}/{len(self.turns) + 1}"
        packet = dict(request_id=request_id, checkpoint_id=checkpoint_id, state=self.public,
                      routes=[self._route_row(route_id, route) for route_id, route in enumerate(self.routes)],
                      geometry_attempts_left=self.geometry_limit - self.ledger.get("geometry_attempts", 0),
                      comparisons=list(self.comparisons.values()), last_feedback=feedback)
        if "query_limit" in self.ledger:
            packet["gp_query_points_left"] = self.ledger["query_limit"] - self.ledger["queries"]
        wire = json.dumps(dict(instructions=INSTRUCTIONS, input=packet), allow_nan=False,
                          sort_keys=True, separators=(",", ":")).encode("utf-8")
        return RouteRequest(request_id, wire, _sha256(wire), len(self.routes))

    def _route_row(self, route_id, resolved):
        aircraft = self.state["aircraft"]
        return dict(route_id=route_id, xy_m=resolved.route, distance_m=resolved.distance_m, time_s=resolved.time_s,
                    distance_left_m=self.hard_left_m - resolved.distance_m,
                    energy_after_wh=self.state["energy_wh"] - resolved.time_s * aircraft.flight_power_w / 3600)

    def _route_id(self, value):
        if type(value) is not int or not 0 <= value < len(self.routes):
            raise ValueError("route ID is not registered at this checkpoint")
        return value

    def _proposal_waypoint(self, args):
        operator = args.get("operator")
        keys = set(args) - {"extra_m"}
        if operator == "via" and keys == {"operator", "waypoint_xy_m"}:
            return xy(args["waypoint_xy_m"])
        if operator != "offset" or keys != {"operator", "forward_m", "lateral_m"}:
            raise ValueError("expected via or offset arguments")
        forward, lateral = (_finite(args[name]) for name in ("forward_m", "lateral_m"))
        if forward <= 0:
            raise ValueError("forward_m must be positive")
        start, first = self.routes[0].route[:2]
        distance = math.dist(start, first)
        east, north = ((second - initial) / distance for initial, second in zip(start, first))
        return xy((start[0] + east * forward - north * lateral, start[1] + north * forward + east * lateral))

    def _generate(self, args):
        used = self.ledger.get("geometry_attempts", 0)
        if used >= self.geometry_limit:
            return dict(status="geometry_budget")
        self.ledger["geometry_attempts"] = used + 1
        waypoint = self._proposal_waypoint(args)
        extra = _finite(args["extra_m"]) if "extra_m" in args else None
        if extra is not None and not 0 <= extra <= max(0., self.hard_left_m - self.shortest_m) + 1e-8:
            raise ValueError("extra_m exceeds available detour slack or is negative")
        geometry = self.state["geometry"]
        start = self.routes[0].route[0]
        if waypoint in (start, self.planner.goal):
            return dict(status="not_intermediate", waypoint_xy_m=waypoint)
        if not geometry.free(waypoint):
            return dict(status="blocked_waypoint", waypoint_xy_m=waypoint)
        if not geometry.edge_free(start, waypoint):
            return dict(status="blocked_first_edge", waypoint_xy_m=waypoint)
        suffix = self.planner.goal_route(waypoint)
        if suffix is None:
            return dict(status="no_goal_suffix", waypoint_xy_m=waypoint)
        route = (start, *suffix)
        if len(set(route)) != len(route):
            return dict(status="repeated_vertex", waypoint_xy_m=waypoint)
        resolved = check_remaining_route(route, **self.state)
        if resolved.status != "legal" or (extra is not None and resolved.distance_m > self.shortest_m + extra + 1e-8):
            status = resolved.status if resolved.status != "legal" else "soft_detour_limit"
            return dict(status=status, waypoint_xy_m=waypoint, **self._route_row(None, resolved))
        speed = self.state["aircraft"].speed_mps
        for route_id, known in enumerate(self.routes):
            if _common_time(known.route, route, speed) >= max(known.time_s, resolved.time_s) - 1e-8:
                return dict(status="duplicate", waypoint_xy_m=waypoint, **self._route_row(route_id, known))
        self.routes.append(resolved)
        return dict(status="generated", waypoint_xy_m=waypoint, **self._route_row(len(self.routes) - 1, resolved))

    def _compare(self, args):
        if set(args) != {"route_id", "reference_id"}:
            raise ValueError("comparison requires route_id and reference_id")
        route_id, reference_id = (self._route_id(args[name]) for name in ("route_id", "reference_id"))
        if route_id == reference_id:
            raise ValueError("comparison requires distinct routes")
        key = (route_id, reference_id)
        if key not in self.comparisons:
            if reference_id not in self.pricing:
                self.pricing[reference_id] = FrozenRepairPricing(
                    [self.routes[reference_id].route], self.posterior, self.state["now"],
                    self.state["aircraft"].speed_mps, min_post_s=self.min_post_s)
            pricing = self.pricing[reference_id]
            evidence = pricing.compare(self.routes[route_id].route, 0)
            self.ledger["pricings"] = self.ledger.get("pricings", 0) + 1
            self.comparisons[key] = dict(status=evidence.price.status, route_id=route_id, reference_id=reference_id,
                                         locked_intervals_s=pricing.locked_intervals,
                                         reference_price=asdict(pricing.menu[0]), comparison=asdict(evidence))
        return self.comparisons[key]

    def _apply(self, call):
        if call.tool == "generate":
            return self._generate(call.args)
        if call.tool == "compare":
            return self._compare(call.args)
        if set(call.args) != {"route_id"}:
            raise ValueError("commit requires only a registered route_id")
        route_id = self._route_id(call.args["route_id"])
        verified = check_remaining_route(self.routes[route_id].route, **self.state)
        if verified.status != "legal":
            return dict(status="commit_verification_failed", route_id=route_id, detail=verified.status)
        return dict(status="committed", **self._route_row(route_id, verified))

    def execute(self, call):
        if call.tool not in ("generate", "compare", "commit"):
            raise ValueError("unknown repair tool")
        try:
            feedback = self._apply(call)
        except QueryLimit:
            feedback = dict(status="compute_budget")
        except (ValueError, OverflowError) as error:
            feedback = dict(status="invalid_arguments", detail=str(error))
        return dict(tool=call.tool, args=call.args, **feedback)

    def review(self, checkpoint_id, exchange, transport):
        """Exchange tool calls without moving, measuring or substituting a route."""
        if self.turns:
            raise ValueError("tool workspace cannot be reused after a review")
        feedback = None
        while True:
            request = self._request(checkpoint_id, feedback)
            receipt = exchange.submit(request, transport, parser=_parse_call)
            call = receipt.selection
            if call is None:
                self.turns.append((request, receipt, None))
                return receipt.status, None
            feedback = self.execute(call)
            self.turns.append((request, receipt, feedback))
            if feedback["status"] in ("compute_budget", "geometry_budget", "committed", "commit_verification_failed"):
                selection = feedback["route_id"] if feedback["status"] == "committed" else None
                return feedback["status"], selection
