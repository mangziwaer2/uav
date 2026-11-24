"""Public-only point measurement, frozen CVaR monitoring, and legal route review."""

from dataclasses import asdict
import math

import numpy as np

from .barbosa_static import gaussian_upper_cvar
from .motion import advance_route
from .passive_map import FixedPriorPosterior, LIMIT_S, SOURCES, candidates, _common_time
from .passive_pricing import UNKNOWN_PRICE
from .passive_pricing_integral import _integrate, _trajectory, compare_aligned_prices
from .passive_trigger import FINAL_S, WINDOW_S, review_reason
from .planning import path_length


STEP_S = 0.25
EPS = 1e-6
MAX_POINTS = 61
MAX_REVIEWS = 5
MAX_QUERIES = 70_000_000
MIN_REVIEW_GAP = 3.0


class QueryLimit(RuntimeError):
    pass


class CountedPosterior:
    def __init__(self, posterior, ledger):
        self.posterior, self.ledger = posterior, ledger
        self.now = posterior.now

    def predict(self, xy, times):
        if self.ledger["queries"] + len(times) > MAX_QUERIES:
            raise QueryLimit
        self.ledger["queries"] += len(times)
        return self.posterior.predict(xy, times)


def risk_signal(route, now, speed, current, committed):
    offsets = np.arange(0.5, min(8.0, path_length(route) / speed), 1.0)
    result = dict(probes=len(offsets), current_supported=0, common=0, new_only=0,
                  max_margin_db=None, witness=None)
    if not len(offsets):
        return result
    xy = [advance_route(route, offset * speed)[0][0] for offset in offsets]
    clocks = now + offsets
    mean, std, source = current.predict(xy, clocks)
    old_mean, old_std, old_source = committed.predict(xy, clocks)
    supported = source < 2
    current_tail = gaussian_upper_cvar(mean, std, 0.05)
    old_tail = gaussian_upper_cvar(old_mean, old_std, 0.01)
    margin = current_tail - old_tail
    result.update(current_supported=int(supported.sum()), common=int((supported & (old_source < 2)).sum()),
                  new_only=int((supported & (old_source == 2)).sum()))
    if supported.any():
        result["max_margin_db"] = float(margin[supported].max())
    crossings = np.flatnonzero(supported & (margin > 0))
    if len(crossings):
        index = crossings[0]
        result["witness"] = dict(xy=xy[index], arrival_s=float(clocks[index]), old_support=SOURCES[old_source[index]],
                                 old_cvar_dbm=float(old_tail[index]), new_cvar_dbm=float(current_tail[index]))
    return result


class _JointSupport:
    def __init__(self, model, other, now):
        self.model, self.other, self.now = model, other, now

    def predict(self, xy, times):
        mean, std, source = self.model.predict(xy, times)
        knots, vertices = self.other
        other_xy = np.column_stack([np.interp(times - self.now, knots, axis) for axis in vertices.T])
        other_source = self.model.predict(other_xy, times)[2]
        return mean, std, np.where(other_source < 2, source, 2)


def _joint_delta(model, now, track, reference, common_grid):
    duration, ref_duration = track[0][-1], reference[0][-1]
    times = common_grid[common_grid <= min(duration, ref_duration)]
    trial = _integrate(_JointSupport(model, reference, now), now, track, times)
    base = _integrate(_JointSupport(model, track, now), now, reference, times)
    return trial["costs"]["risk"] - base["costs"]["risk"], max(0.0, duration - trial["unknown_s"])


def select_route(routes, incumbent, model, now, speed):
    prices = compare_aligned_prices(routes, model, now, speed, step_s=STEP_S)
    rows = [dict(route=route, **price) for route, price in zip(routes, prices["rows"])]
    reference = routes.index(incumbent) if incumbent else None
    eligible = []
    if reference is None:
        eligible = prices["order"]["risk"]
    else:
        tracks = [_trajectory(route, speed) for route in routes]
        durations = [track[0][-1] for track in tracks]
        grid = np.union1d(np.arange(0, max(durations), STEP_S / 2), durations)
        for index, row in enumerate(rows):
            if index == reference:
                continue
            delta = row["costs"]["risk"] - rows[reference]["costs"]["risk"]
            row["delta_risk"] = delta
            if delta >= -EPS or _common_time(routes[index], incumbent, speed) >= max(
                    durations[index], durations[reference]) - EPS:
                continue
            joint, support = _joint_delta(model, now, tracks[index], tracks[reference], grid)
            row.update(delta_common=joint, common_support_s=support)
            if joint < -EPS:
                eligible.append(index)
    selected = reference
    if eligible:
        best_price = min(rows[index]["costs"]["risk"] for index in eligible)
        selected = min((index for index in eligible if rows[index]["costs"]["risk"] <= best_price + EPS),
                       key=lambda index: (rows[index]["time_s"], index))
    return dict(candidates=rows, selected=selected, reference=reference, qualified=len(eligible),
                changed=reference is not None and selected != reference)


def _add_feedback(forecast, sample):
    if forecast is not None:
        if (forecast["arrival_s"] != sample["time_seconds"]
                or tuple(forecast["xy"]) != sample["position_m"]):
            raise ValueError("frozen prediction does not match observed arrival")
        forecast["error_db"] = sample["interference_dbm"] - forecast["dbm"]
        forecast["absolute_error_db"] = abs(forecast["error_db"])


def fly(geometry, aircraft, settings, observe, policy):
    if policy not in ("fixed", "event"):
        raise ValueError("unknown review policy")
    position, goal = list(geometry.config.start_m), list(geometry.config.goal_m)
    cap, speed = 1.25 * math.dist(position, goal), aircraft.speed_mps
    now, flown, energy = 0.0, 0.0, aircraft.initial_energy_wh
    history, frames, decisions, traversed = [], [], [], []
    ledger = dict(queries=0, reviews=0, fits=0)
    used, committed, incumbent = set(), None, []
    forecast, moved, last_review = None, [position], 0.0
    display_grid = geometry.grid().reshape(-1, 2)
    while True:
        sample = asdict(observe(position, now))
        _add_feedback(forecast, sample)
        history.append(sample)
        posterior = FixedPriorPosterior(history, geometry, now, settings)
        ledger["fits"] += 1
        current = CountedPosterior(posterior, ledger)
        frame = dict(t=now, xy=position, flown_m=flown, moved=moved, forecast=forecast,
                     measured_dbm=sample["interference_dbm"], prefix_size=len(history), remaining=incumbent)
        frames.append(frame)
        stop = "arrived" if position == goal else None
        if stop is None and now >= LIMIT_S:
            stop = "time_limit"
        if stop is None and len(history) >= MAX_POINTS:
            stop = "point_budget"

        try:
            if stop is None and now <= FINAL_S:
                eligible = False
                if committed is not None:
                    signal = risk_signal(incumbent, now, speed, current, committed)
                    slot_spent = math.ceil(now / WINDOW_S) in used
                    cooldown = now - last_review < MIN_REVIEW_GAP
                    frame["monitor"] = dict(reference_s=committed.now, slot_spent=slot_spent,
                                            cooldown=cooldown, **signal)
                    eligible = bool(signal["witness"]) and not cooldown
                reason = review_reason(now, used, policy, eligible)
                if reason:
                    before = incumbent
                    if ledger["reviews"] >= MAX_REVIEWS:
                        raise QueryLimit
                    ledger["reviews"] += 1
                    routes = candidates(position, goal, before, geometry, aircraft, now, flown, energy, cap)
                    if not routes:
                        stop = "no_legal_remaining_route"
                    else:
                        if before and before not in routes:
                            raise RuntimeError("committed remainder failed independent legality check")
                        action = select_route(routes, before, current, now, speed)
                        incumbent = routes[action["selected"]]
                        if committed is None or action["changed"]:
                            committed = current
                        decisions.append(dict(t=now, reason=reason, prefix_size=len(history),
                                              committed_s=committed.now, **action))
                        frame.update(remaining=incumbent, decision=len(decisions) - 1)
                        last_review = now
                        if now > 0:
                            used.add(math.ceil(now / WINDOW_S))

            if stop is None:
                duration = min(1.0, LIMIT_S - now, path_length(incumbent) / speed)
                if duration <= 0:
                    raise RuntimeError("cannot advance on the committed remainder")
                remaining, moved = advance_route(incumbent, duration * speed)
                next_time, next_xy = now + duration, remaining[0]
                mean, std, support = current.predict([next_xy], [next_time])
                forecast = dict(issued_s=now, arrival_s=next_time, xy=next_xy, dbm=float(mean[0]),
                                support=SOURCES[support[0]],
                                observation_std_db=float(np.hypot(std[0], settings.measured_error_db)))
        except QueryLimit:
            stop = "compute_budget"

        mean, std, support = posterior.predict(display_grid, np.full(len(display_grid), now))
        frame["gp"] = dict(dbm=mean.round(3).tolist(), std_db=std.round(3).tolist(), support=support.tolist())
        if stop is not None:
            break

        elapsed = 0.0
        for first, second in zip(moved, moved[1:]):
            distance = math.dist(first, second)
            if not geometry.edge_free(first, second):
                raise RuntimeError("executed segment violates geometry")
            step = distance / speed
            traversed.append((first, second, now + elapsed, step))
            elapsed += step
            flown += distance
            energy -= step * aircraft.flight_power_w / 3600
        if flown > cap + 1e-7 or energy < aircraft.reserve_energy_wh - 1e-8:
            raise RuntimeError("executed segment violates distance or energy budget")
        position, incumbent, now = next_xy, remaining, next_time
    return dict(mode=policy, frames=frames, decisions=decisions, points=history, stop=stop, energy_left_wh=energy,
                cap_m=cap, resources=ledger), traversed
