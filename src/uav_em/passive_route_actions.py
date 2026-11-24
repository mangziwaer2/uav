"""Resolve direct or waypoint proposals in the same legal remainder menu."""

import math

from .event_inference import digest
from .passive_closed_loop import EPS, select_route
from .passive_map import candidates


def _proxy_best(rows, indices):
    best_cost = min(rows[index]["risk_proxy_tq"] for index in indices)
    return min((index for index in indices if rows[index]["risk_proxy_tq"] <= best_cost + EPS),
               key=lambda index: (rows[index]["travel_s"], index))


def build_route_action_menu(geometry, aircraft, posterior, *, position, incumbent, now, flown_m, energy_wh):
    """Price one checkpoint with the existing legality and numerical review tools."""
    if (not all(math.isfinite(value) and value >= 0 for value in (now, flown_m, energy_wh))
            or posterior.now != now):
        raise ValueError("current posterior and finite nonnegative flight state required")
    position, goal = list(position), list(geometry.config.goal_m)
    incumbent = [list(point) for point in incumbent]
    cap = 1.25 * math.dist(geometry.config.start_m, goal)
    routes = candidates(position, goal, incumbent, geometry, aircraft, now, flown_m, energy_wh, cap)
    if incumbent not in routes:
        raise ValueError("a legal current remainder is required")

    review = select_route(routes, incumbent, posterior, now, aircraft.speed_mps)
    rows = [dict(route_idx=index, route_xy_m=row["route"], travel_s=row["time_s"],
                 risk_proxy_tq=row["costs"]["risk"], unknown_s=row["unknown_s"],
                 guarded_delta_tq=row.get("delta_common")) for index, row in enumerate(review["candidates"])]
    groups = {}
    for row in rows:
        for point in row["route_xy_m"][1:-1]:
            if point != position and point != goal:
                groups.setdefault(tuple(point), set()).add(row["route_idx"])
    waypoints = []
    for index, (point, indices) in enumerate(sorted(groups.items())):
        indices = sorted(indices)
        waypoints.append(dict(waypoint_idx=index, xy_m=list(point), route_indices=indices,
                              completed_route_idx=_proxy_best(rows, indices)))

    reference = review["reference"]
    reachable = sorted({reference, *(point["completed_route_idx"] for point in waypoints)})
    body = dict(t_s=now, flown_m=flown_m, cap_m=cap, energy_wh=energy_wh, incumbent_idx=reference,
                routes=rows, waypoints=waypoints, waypoint_reachable_route_indices=reachable,
                numeric_guarded_idx=review["selected"], numeric_proxy_idx=_proxy_best(rows, range(len(rows))),
                numeric_waypoint_route_idx=_proxy_best(rows, reachable))
    return dict(body, menu_id=digest(body))


def resolve_route_action(menu, reply, *, mode):
    """Resolve one decoded proposal, without applying a soft veto or moving the UAV."""
    if mode not in ("route", "waypoint"):
        raise ValueError("unknown proposal mode")
    if digest({key: value for key, value in menu.items() if key != "menu_id"}) != menu["menu_id"]:
        raise ValueError("route action menu changed")
    if not isinstance(reply, dict) or set(reply) != {"menu_id", "action", "idx", "reason"}:
        raise ValueError("exact route proposal fields required")
    if reply["menu_id"] != menu["menu_id"]:
        raise ValueError("stale route proposal")
    action, index, reason = reply["action"], reply["idx"], reply["reason"]
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 300:
        raise ValueError("a short proposal reason is required")
    if action == "keep" and index is None:
        return menu["incumbent_idx"]
    options = menu["routes"] if mode == "route" else menu["waypoints"]
    if action != mode or type(index) is not int or not 0 <= index < len(options):
        raise ValueError("proposal is not an allowed option in this mode")
    return index if mode == "route" else options[index]["completed_route_idx"]
