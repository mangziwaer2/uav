"""Public checkpoint and shared numerical controller for a one-shot model choice."""

from dataclasses import asdict, replace
import json
import math

import numpy as np

from .motion import advance_route
from .passive_closed_loop import CountedPosterior, MAX_POINTS, MAX_REVIEWS, QueryLimit, _add_feedback, select_route
from .passive_map import FixedPriorPosterior, LIMIT_S, SOURCES, candidates
from .planning import path_length


MODELS = dict(nominal=20.0, short=6.0, long=45.0)
CHECKPOINT = 24.0
REVIEWS = (0, 12, 24, 36, 48)
MAX_PRICINGS = 7
MAX_FITS = 111
MAX_INPUT_BYTES = 24_000
MAX_REPLY_BYTES = 1_000


def numerical_choice(packet):
    history = packet["forecast_feedback"]
    weights = [2 ** (-(packet["t_s"] - row["arrival_s"]) / 6) for row in history]
    scores = {}
    for model in MODELS:
        forecasts = [row["models"][model] for row in history]
        scores[model] = sum(weight * (math.log(item["observation_std_db"])
                                     + 0.5 * (item["error_db"] / item["observation_std_db"]) ** 2)
                            for weight, item in zip(weights, forecasts)) / sum(weights)
    return min(scores, key=scores.get), scores


def parse_reply(raw):
    def unique_keys(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate JSON keys")
        return result

    if len(raw.encode("utf-8")) > MAX_REPLY_BYTES or len(raw.splitlines()) != 1:
        raise ValueError("expected one bounded line")
    reply = json.loads(raw, object_pairs_hook=unique_keys)
    if (not isinstance(reply, dict) or set(reply) != {"model", "reason"}
            or not isinstance(reply["model"], str) or reply["model"] not in MODELS
            or not isinstance(reply["reason"], str) or not reply["reason"].strip()):
        raise ValueError("invalid model choice")
    return reply


class Flight:
    def __init__(self, geometry, aircraft, settings):
        self.geometry, self.aircraft = geometry, aircraft
        self.settings = {model: replace(settings, gp_time_scale_seconds=scale) for model, scale in MODELS.items()}
        self.display_grid = geometry.grid().reshape(-1, 2)
        self.position, self.goal = list(geometry.config.start_m), list(geometry.config.goal_m)
        self.cap = 1.25 * math.dist(self.position, self.goal)
        self.speed, self.energy = aircraft.speed_mps, aircraft.initial_energy_wh
        self.now, self.flown = 0.0, 0.0
        self.active, self.activated = "nominal", 0.0
        self.incumbent, self.moved, self.pending = [], [self.position], {}
        self.points, self.frames, self.decisions, self.traversed = [], [], [], []
        self.ledger = dict(queries=0, reviews=0, fits=0, pricings=0)
        self.stop = None

    def _observe(self, sensor):
        sample = asdict(sensor(self.position, self.now))
        for forecast in self.pending.values():
            _add_feedback(forecast, sample)
        self.points.append(sample)
        frame = dict(t=self.now, xy=self.position, flown_m=self.flown, moved=self.moved, remaining=self.incumbent,
                     prefix_size=len(self.points), measured_dbm=sample["interference_dbm"],
                     forecast=self.pending.get(self.active))
        if self.now <= CHECKPOINT:
            frame["bank_feedback"] = self.pending
        self.frames.append(frame)
        if self.position == self.goal:
            self.stop = "arrived"
        elif self.now >= LIMIT_S:
            self.stop = "time_limit"
        elif len(self.points) >= MAX_POINTS:
            self.stop = "point_budget"

    def _fit(self, models):
        if self.ledger["fits"] + len(models) > MAX_FITS:
            raise QueryLimit
        self.ledger["fits"] += len(models)
        return {model: FixedPriorPosterior(self.points, self.geometry, self.now, self.settings[model])
                for model in models}

    def _menu(self, posteriors):
        routes = candidates(self.position, self.goal, self.incumbent, self.geometry, self.aircraft,
                            self.now, self.flown, self.energy, self.cap)
        if not routes:
            self.stop = "no_legal_remaining_route"
            return {}
        if self.incumbent and self.incumbent not in routes:
            raise RuntimeError("incumbent failed independent legality check")
        if self.ledger["pricings"] + len(posteriors) > MAX_PRICINGS:
            raise QueryLimit
        self.ledger["pricings"] += len(posteriors)
        return {model: select_route(routes, self.incumbent, CountedPosterior(posterior, self.ledger),
                                    self.now, self.speed) for model, posterior in posteriors.items()}

    def _apply(self, review):
        if self.ledger["reviews"] >= MAX_REVIEWS:
            raise QueryLimit
        self.ledger["reviews"] += 1
        self.incumbent = review["candidates"][review["selected"]]["route"]
        reason = "initial" if self.now == 0 else "scheduled"
        self.decisions.append(dict(t=self.now, reason=reason, model=self.active,
                                   prefix_size=len(self.points), **review))
        self.frames[-1].update(remaining=self.incumbent, decision=len(self.decisions) - 1)

    def _display(self, posterior):
        mean, std, support = posterior.predict(self.display_grid, np.full(len(self.display_grid), self.now))
        self.frames[-1].update(active_model=self.active,
                               gp=dict(dbm=mean.round(3).tolist(), std_db=std.round(3).tolist(),
                                       support=support.tolist()))

    def _advance(self, posteriors):
        duration = min(1.0, LIMIT_S - self.now, path_length(self.incumbent) / self.speed)
        if duration <= 0:
            raise RuntimeError("cannot advance on the committed remainder")
        remaining, moved = advance_route(self.incumbent, duration * self.speed)
        arrival, position = self.now + duration, remaining[0]
        forecasts = {}
        for model, posterior in posteriors.items():
            mean, std, support = CountedPosterior(posterior, self.ledger).predict([position], [arrival])
            forecasts[model] = dict(issued_s=self.now, arrival_s=arrival, xy=position, dbm=float(mean[0]),
                                    support=SOURCES[support[0]], model=model,
                                    observation_std_db=float(np.hypot(std[0], self.settings[model].measured_error_db)))
        elapsed, distance, segments = 0.0, 0.0, []
        for first, second in zip(moved, moved[1:]):
            length = math.dist(first, second)
            if not self.geometry.edge_free(first, second):
                raise RuntimeError("executed segment violates geometry")
            step = length / self.speed
            segments.append((first, second, self.now + elapsed, step))
            elapsed += step
            distance += length
        remaining_energy = self.energy - elapsed * self.aircraft.flight_power_w / 3600
        if self.flown + distance > self.cap + 1e-7 or remaining_energy < self.aircraft.reserve_energy_wh - 1e-8:
            raise RuntimeError("executed segment violates distance or energy budget")
        self.traversed.extend(segments)
        self.flown += distance
        self.energy = remaining_energy
        self.position, self.now, self.moved = position, arrival, moved
        self.incumbent, self.pending = remaining, forecasts

    def run(self, sensor, *, pause=False):
        while not self.stop:
            self._observe(sensor)
            review_bank = pause and self.now == CHECKPOINT
            try:
                posteriors = self._fit(MODELS if pause else (self.active,))
            except QueryLimit:
                self.stop = "compute_budget"
                posterior = FixedPriorPosterior([], self.geometry, self.now, self.settings[self.active])
                self._display(posterior)
                break
            try:
                if not self.stop and self.now in REVIEWS:
                    menu = self._menu(posteriors if review_bank else {self.active: posteriors[self.active]})
                    if menu:
                        if review_bank:
                            return posteriors, menu
                        self._apply(menu[self.active])
                self._display(posteriors[self.active])
                if not self.stop:
                    self._advance(posteriors)
            except QueryLimit:
                self.stop = "compute_budget"
                if "gp" not in self.frames[-1]:
                    self._display(posteriors[self.active])
        return None

    def resume(self, model, posterior, review, sensor):
        if self.now != CHECKPOINT or self.stop or model not in MODELS or posterior.now != self.now:
            raise ValueError("invalid frozen checkpoint selection")
        if model != self.active:
            self.active, self.activated = model, self.now
        try:
            self._apply(review)
            self._display(posterior)
            self._advance({model: posterior})
        except QueryLimit:
            self.stop = "compute_budget"
            if "gp" not in self.frames[-1]:
                self._display(posterior)
        self.run(sensor)

    def packet(self, menu):
        if self.now != CHECKPOINT or len(self.frames) != 25 or set(menu) != set(MODELS):
            raise ValueError("public checkpoint prefix is incomplete")
        feedback = [dict(arrival_s=frame["t"], models={model: {key: forecast[key] for key in (
            "issued_s", "dbm", "observation_std_db", "error_db", "support")}
            for model, forecast in frame["bank_feedback"].items()}) for frame in self.frames[-12:]]
        return dict(t_s=self.now, xy=self.position, goal_xy=self.goal, flown_m=self.flown, cap_m=self.cap,
                    remaining_wh=self.energy, reserve_wh=self.aircraft.reserve_energy_wh, speed_mps=self.speed,
                    buildings=self.geometry.config.buildings, point_measurements=self.points, scales_s=MODELS,
                    forecast_feedback=feedback, incumbent_idx=menu[self.active]["reference"],
                    routes=[item["route"] for item in menu[self.active]["candidates"]],
                    review={model: dict(chosen=item["selected"], changed=item["changed"],
                                        rows=[[row["costs"]["risk"], row["unknown_s"], row.get("delta_common"),
                                               row.get("common_support_s")] for row in item["candidates"]])
                            for model, item in menu.items()})

    def log(self, mode):
        return dict(mode=mode, frames=self.frames, decisions=self.decisions, points=self.points,
                    model=self.active, activated_s=self.activated, stop=self.stop, energy_left_wh=self.energy,
                    cap_m=self.cap, resources=self.ledger)
