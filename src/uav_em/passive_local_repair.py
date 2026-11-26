"""Splice one unflown subpath and check the complete remainder."""

from dataclasses import dataclass
import math

from .passive_map import LIMIT_S


def xy(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(coordinate) not in (int, float) for coordinate in value)):
        raise ValueError("expected an XY pair")
    try:
        point = tuple(map(float, value))
    except OverflowError as error:
        raise ValueError("expected finite XY coordinates") from error
    if not all(map(math.isfinite, point)):
        raise ValueError("expected finite XY coordinates")
    return point


def waypoint_domain(value):
    if not isinstance(value, (tuple, list)):
        raise ValueError("expected a finite list of repair waypoints")
    points = frozenset(map(xy, value))
    if len(points) != len(value):
        raise ValueError("duplicate repair waypoint")
    return points


@dataclass(frozen=True)
class LocalRepair:
    left: int
    right: int
    waypoint_xy_m: tuple[float, float] | None = None


@dataclass(frozen=True)
class ResolvedRepair:
    route: tuple[tuple[float, float], ...]
    status: str
    distance_m: float
    time_s: float


def check_remaining_route(route, *, geometry, aircraft, now, flown_m, energy_wh):
    """Check a complete canonical route from the current position to the goal."""
    cap = 1.25 * math.dist(geometry.config.start_m, geometry.config.goal_m)
    edges = tuple(zip(route, route[1:]))
    distance = sum(math.dist(first, second) for first, second in edges)
    duration = distance / aircraft.speed_mps
    if (flown_m + distance > cap + 1e-8 or now + duration > LIMIT_S + 1e-8
            or energy_wh - duration * aircraft.flight_power_w / 3600 < aircraft.reserve_energy_wh):
        status = "resources"
    elif not all(geometry.edge_free(first, second) for first, second in edges):
        status = "blocked"
    else:
        status = "legal"
    return ResolvedRepair(route, status, distance, duration)


def resolve_repair(selected_route, edit, *, geometry, aircraft, position, now, flown_m, energy_wh):
    """Keep the controller's selected route if its optional splice is infeasible."""
    base = tuple(map(xy, selected_route))
    start, goal = xy(position), xy(geometry.config.goal_m)
    if len(base) < 2 or base[0] != start or base[-1] != goal:
        raise ValueError("selected route must run from the current position to the goal")
    if not all(math.isfinite(value) and value >= 0 for value in (now, flown_m, energy_wh)):
        raise ValueError("invalid used resources")

    def check(route, success):
        result = check_remaining_route(route, geometry=geometry, aircraft=aircraft, now=now,
                                       flown_m=flown_m, energy_wh=energy_wh)
        return ResolvedRepair(route, success if result.status == "legal" else result.status,
                              result.distance_m, result.time_s)

    selected = check(base, "selected")
    if selected.status != "selected":
        raise ValueError(f"selected menu route failed verification: {selected.status}")
    if edit is None:
        return selected
    if type(edit.left) is not int or type(edit.right) is not int or not 0 <= edit.left < edit.right < len(base):
        return ResolvedRepair(base, "invalid_segment", selected.distance_m, selected.time_s)
    via = () if edit.waypoint_xy_m is None else (xy(edit.waypoint_xy_m),)
    if via and (via[0] == base[edit.left] or via[0] == base[edit.right]):
        via = ()
    repaired = base[:edit.left + 1] + via + base[edit.right:]
    if repaired == base:
        return ResolvedRepair(base, "unchanged", selected.distance_m, selected.time_s)
    result = check(repaired, "repaired")
    if result.status == "repaired":
        return result
    return ResolvedRepair(base, result.status, selected.distance_m, selected.time_s)
