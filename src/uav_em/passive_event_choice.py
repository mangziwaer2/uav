"""Finite-window route choice and causal risk alert for the E1 pilot."""

import numpy as np

from .motion import advance_route
from .passive_map import LIMIT_S, _common_time, candidates
from .passive_pricing import upper_tail_power
from .passive_pricing_integral import _trajectory
from .planning import path_length


HORIZON_S = 6.0
ALERT_DBM = -80.0
MAX_EVENTS = 3


def alert(route, model, now, speed):
    offsets = np.arange(1.0, 5.0)
    offsets = offsets[offsets <= path_length(route) / speed]
    if not len(offsets):
        return None, False
    points = [advance_route(route, delta * speed)[0][0] for delta in offsets]
    means, _, support = model.predict(points, now + offsets)
    supported = means[support < 2]
    peak = float(supported.max()) if len(supported) else None
    return peak, peak is not None and peak >= ALERT_DBM


def menu(flight):
    routes = candidates(flight.position, flight.goal, flight.incumbent, flight.geometry, flight.aircraft,
                        flight.now, flight.flown, flight.energy, flight.cap)
    if flight.incumbent and flight.incumbent not in routes:
        raise ValueError("committed route failed independent verification")
    child = None
    if len(flight.incumbent) >= 3:
        shortcut = [flight.position, *flight.incumbent[2:]]
        length = path_length(shortcut)
        duration = length / flight.speed
        if (shortcut[-1] == flight.goal
                and flight.flown + length <= flight.cap + 1e-8 and flight.now + duration <= LIMIT_S + 1e-8
                and flight.energy - duration * flight.aircraft.flight_power_w / 3600 >= flight.aircraft.reserve_energy_wh
                and all(flight.geometry.edge_free(start, end) for start, end in zip(shortcut, shortcut[1:]))
                and not any(_common_time(shortcut, route, flight.speed)
                            >= max(duration, path_length(route) / flight.speed) - 1e-8 for route in routes)):
            child = len(routes)
            routes.append(shortcut)
    return routes, child


def prices(routes, model, now, speed, incumbent=None, child=None):
    if not 1 <= len(routes) <= 126:
        raise ValueError("route count outside E1 budget")
    tracks = [_trajectory(route, speed) for route in routes]
    durations = np.array([knots[-1] for knots, _ in tracks])
    window = np.minimum(durations, HORIZON_S)
    cut = np.union1d(np.arange(0, HORIZON_S, 0.5), window)
    cut = cut[cut <= max(window)]
    offsets, widths = (cut[:-1] + cut[1:]) / 2, np.diff(cut)
    active = offsets < window[:, None]
    covered, density = np.zeros_like(active), np.zeros_like(active, dtype=float)
    points = np.concatenate([np.column_stack([np.interp(offsets[mask], knots, axis) for axis in vertices.T])
                             for (knots, vertices), mask in zip(tracks, active)])
    clocks = now + np.broadcast_to(offsets, active.shape)[active]
    means, spread, support = model.predict(points, clocks)
    covered[active] = support < 2
    density[covered] = upper_tail_power(means[support < 2], spread[support < 2])
    common = np.all(~active | covered, axis=0)
    compared = (active & common) @ widths
    costs = window + (density * common) @ widths
    if not np.isfinite(costs).all():
        raise ValueError("nonfinite E1 window score")
    reference = routes.index(incumbent) if incumbent else None
    rows = [dict(route=route, time_s=float(window[index]), goal_s=float(durations[index]),
                 costs=dict(risk=float(costs[index])), unknown_s=max(0.0, float(window[index] - covered[index] @ widths)),
                 compared_s=float(compared[index]), individual_q=float(density[index] @ widths),
                 edit="delete_reconnect" if index == child else "enumerated",
                 delta_common=float(costs[index] - costs[reference]) if reference is not None else None)
            for index, route in enumerate(routes)]
    selected = min(range(len(rows)), key=lambda index: (round(float(costs[index]), 8), durations[index], index))
    return rows, selected, reference
