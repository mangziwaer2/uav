"""Public-only P2 trigger checks, with one review per fixed window."""

import math

import numpy as np

from .barbosa_static import gaussian_upper_cvar
from .motion import advance_route
from .passive_map import _common_time
from .planning import path_length


WINDOW_S = 12
FINAL_S = 48


def review_reason(now, used_windows, policy, eligible):
    if now == 0:
        return "initial"
    if not 0 < now <= FINAL_S:
        return None
    window = math.ceil(now / WINDOW_S)
    if window in used_windows:
        return None
    if now == window * WINDOW_S:
        return "scheduled"
    return "early_risk" if policy == "event" and eligible else None


def risk_signal(route, now, speed, current, committed):
    offsets = np.arange(0.5, min(8.0, path_length(route) / speed), 1.0)
    if not len(offsets):
        return dict(comparable=0, witness=None)
    xy = [advance_route(route, float(offset) * speed)[0][0] for offset in offsets]
    clocks = now + offsets
    mean, std, support = current.predict(xy, clocks)
    old_mean, old_std, old_support = committed.predict(xy, clocks)
    comparable = (support < 2) & (old_support < 2)
    new_tail = gaussian_upper_cvar(mean, std, 0.05)
    old_tail = gaussian_upper_cvar(old_mean, old_std, 0.01)
    crossing = np.flatnonzero(comparable & (new_tail > old_tail))
    witness = None
    if len(crossing):
        index = crossing[0]
        witness = dict(xy=xy[index], arrival_s=float(clocks[index]), new_cvar_dbm=float(new_tail[index]),
                       old_cvar_dbm=float(old_tail[index]))
    return dict(comparable=int(comparable.sum()), witness=witness)


def has_alternative(routes, incumbent, speed):
    limit = path_length(incumbent) / speed
    return any(_common_time(route, incumbent, speed) < max(limit, path_length(route) / speed) - 1e-8
               for route in routes)
