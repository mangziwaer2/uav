"""Causal branch opportunities and aligned local evidence for a new pilot."""

import numpy as np

from .passive_event_choice import ALERT_DBM, HORIZON_S
from .passive_map import _common_time
from .passive_pricing import upper_tail_power
from .passive_pricing_integral import _trajectory


def _positions(track, offsets):
    knots, vertices = track
    return np.column_stack([np.interp(offsets, knots, axis) for axis in vertices.T])


class BranchOpportunity:
    def __init__(self, separation_m):
        self.separation_m = separation_m
        self.seen = set()

    def check(self, high, incumbent, routes, speed):
        if not high:
            self.seen.clear()
            return "rearmed", []
        base = _trajectory(incumbent, speed)
        available = set()
        for route in routes:
            if route == incumbent:
                continue
            track = _trajectory(route, speed)
            end = min(HORIZON_S, base[0][-1], track[0][-1])
            offsets = np.unique(np.concatenate(([end], base[0][base[0] < end], track[0][track[0] < end])))
            if np.linalg.norm(_positions(base, offsets) - _positions(track, offsets), axis=1).max() < self.separation_m:
                continue
            split = _common_time(incumbent, route, speed)
            first = np.searchsorted(track[0], split + 1e-8, side="right")
            if first < len(track[1]):
                available.add(tuple(track[1][first]))
        if not available:
            return "no_alternative", []
        new = sorted(available - self.seen)
        if not new:
            return "already_reviewed", []
        status = "new_branch" if self.seen else "reviewed"
        self.seen.update(available)
        return status, [list(point) for point in new]


def price_branches(routes, model, now, speed, incumbent=None):
    if not 1 <= len(routes) <= 126:
        raise ValueError("route count exceeds the shared menu bound")
    reference = routes.index(incumbent) if incumbent else None
    tracks = [_trajectory(route, speed) for route in routes]
    durations = np.array([track[0][-1] for track in tracks])
    windows = np.minimum(durations, HORIZON_S)
    splits = [0.0 if reference is None else _common_time(routes[reference], route, speed) for route in routes]
    cut = np.unique(np.r_[np.arange(0, HORIZON_S, 0.5), windows, np.minimum(splits, windows.max())])
    cut = cut[cut <= windows.max()]
    offsets, widths = (cut[:-1] + cut[1:]) / 2, np.diff(cut)
    active = offsets < windows[:, None]
    points = np.concatenate([_positions(track, offsets[mask]) for track, mask in zip(tracks, active)])
    clocks = now + np.broadcast_to(offsets, active.shape)[active]
    mean, std, source = model.predict(points, clocks)
    support = source < 2
    covered = np.zeros_like(active)
    covered[active] = support
    expected, tail = (np.zeros_like(active, dtype=float) for _ in range(2))
    scale = np.log(10) / 10
    expected[covered] = np.exp(scale * (mean[support] - ALERT_DBM) + 0.5 * (scale * std[support]) ** 2)
    tail[covered] = upper_tail_power(mean[support], std[support])
    common = np.all(~active | covered, axis=0)
    rows = []
    for index, route in enumerate(routes):
        priced = float((active[index] & common) @ widths)
        unknown = float((active[index] & ~covered[index]) @ widths)
        total = float(durations[index] + tail[index, common] @ widths[common])
        row = dict(route=route, time_s=float(durations[index]), compared_s=priced, unknown_s=unknown,
                   costs=dict(risk=total), outside_s=float(durations[index] - windows[index]),
                   supported_not_compared_s=float((covered[index] & ~common) @ widths))
        if reference is not None:
            post = common & (offsets > splits[index]) & (active[index] | active[reference])
            difference = lambda values: float((values[index, post] - values[reference, post]) @ widths[post])
            row.update(delta_t_s=float(durations[index] - durations[reference]), split_s=float(splits[index]),
                       post_common_s=float(widths[post].sum()),
                       delta_expected_q=difference(expected) if post.any() else None,
                       delta_tail_q=difference(tail) if post.any() else None)
        rows.append(row)
    winner = min(range(len(rows)), key=lambda index: (round(rows[index]["costs"]["risk"], 8), durations[index], index))
    return rows, winner, reference
