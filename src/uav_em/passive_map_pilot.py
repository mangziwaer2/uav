"""Single-use P1 paired simulation and self-contained replay publisher."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .config import ProjectConfig
from .environment import ElectromagneticWorld
from .motion import advance_route
from .passive_map import MODES, SOURCES, LIMIT_S, FixedPriorPosterior, candidates, compare_routes, quadrature
from .planning import path_length


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/passive_map_P1_20261123"
CONTRACT = ROOT / "docs/passive_map_P1_contract.md"
TEMPLATE = Path(__file__).with_name("passive_map_replay.html")
REVIEWS = (0, 12, 24, 36, 48)
WORLD_SEED = 43157
NOISE_SEED = 43158


def _write(path, data, mode="x"):
    with path.open(mode, encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _make_world():
    rng = np.random.default_rng(WORLD_SEED)
    fields = tuple(dict(center_m=[float(rng.uniform(low, high)), float(rng.uniform(160, 440))],
                        velocity_mps=rng.uniform(-1.25, 1.25, 2).tolist(),
                        sigma_m=float(rng.uniform(60, 100)), peak_dbm=float(rng.uniform(-75, -65)),
                        start_seconds=start, end_seconds=end)
                   for low, high, start, end in ((200, 320, 0, 90), (360, 520, 12, 42), (530, 680, 24, 90)))
    config = ProjectConfig(seed=NOISE_SEED)
    return ElectromagneticWorld(replace(config, electromagnetic=replace(
        config.electromagnetic, fields=fields, background_dbm=-100, measurement_noise_db=1)))


def _flight(mode, world, on_decision):
    geometry, aircraft, settings = world.geometry, world.config.aircraft, world.config.belief
    position, goal = list(geometry.config.start_m), list(geometry.config.goal_m)
    cap = 1.25 * math.dist(position, goal)
    grid = geometry.grid().reshape(-1, 2)
    now, flown, energy = 0.0, 0.0, aircraft.initial_energy_wh
    observations, frames, decisions, traversed = [], [], [], []
    incumbent, checked, review, moved = [], None, 0, [position]
    stop = "time_limit"
    while True:
        sample = asdict(world.observe(position, now))
        if checked is not None:
            checked["absolute_error_db"] = abs(checked["dbm"] - sample["interference_dbm"])
        observations.append(sample)
        frames.append(dict(t=now, xy=position, arrived=position == goal, flown_m=flown,
                           moved=moved, measured_dbm=sample["interference_dbm"], forecast=checked))
        if position == goal or now >= LIMIT_S:
            stop = "arrived" if position == goal else "time_limit"
            break
        if review < len(REVIEWS) and now == REVIEWS[review]:
            routes = candidates(position, goal, incumbent, geometry, aircraft, now, flown, energy, cap)
            if not routes:
                stop = "no_legal_remaining_route"
                break
            model = FixedPriorPosterior(observations, geometry, now, settings)
            action = compare_routes(routes, model, now, aircraft.speed_mps)
            mean, std, support = model.predict(grid, np.full(len(grid), now))
            incumbent = routes[action["selected"][mode]]
            decisions.append(dict(t=now, xy=position, flown_m=flown, energy_wh=energy,
                                  prefix_size=len(observations), **action,
                                  gp=dict(dbm=mean.round(3).tolist(), std_db=std.round(3).tolist(),
                                          support=support.tolist())))
            review += 1
            on_decision(mode, now, len(routes), len(observations))

        next_review = REVIEWS[review] if review < len(REVIEWS) else LIMIT_S
        duration = min(1.0, next_review - now, path_length(incumbent) / aircraft.speed_mps)
        if duration <= 0:
            raise RuntimeError("execution cannot advance")
        remaining, moved = advance_route(incumbent, duration * aircraft.speed_mps)
        elapsed = 0.0
        for first, second in zip(moved, moved[1:]):
            if not geometry.edge_free(first, second):
                raise RuntimeError("executed segment violated geometry")
            distance = math.dist(first, second)
            step = distance / aircraft.speed_mps
            traversed.append(((np.asarray(first) + second) / 2, now + elapsed + step / 2, step))
            elapsed += step
            flown += distance
            energy -= step * aircraft.flight_power_w / 3600
        if flown > cap + 1e-7 or energy < aircraft.reserve_energy_wh - 1e-8:
            raise RuntimeError("executed segment violated resource constraints")
        now = next_review if duration == next_review - now else now + duration
        position, incumbent = remaining[0], remaining
        prediction, std, support = model.predict([position], [now])
        checked = dict(issued_s=model.now, dbm=float(prediction[0]),
                       observation_std_db=float(np.hypot(std[0], settings.measured_error_db)),
                       support=SOURCES[support[0]])
    return (dict(mode=mode, frames=frames, decisions=decisions, points=observations,
                 stop=stop, energy_wh=aircraft.initial_energy_wh - energy), traversed)


def _truth_q(world, samples):
    return sum(weight * 10 ** ((world.truth(point, clock) + 80) / 10) for point, clock, weight in samples)


def _bucket(errors, labels, definitions):
    return {name: dict(n=int(np.sum(labels == code)),
                       mae_db=float(errors[labels == code].mean()) if (labels == code).any() else None)
            for code, name in definitions}


def _evaluate(world, log, traversed):
    geometry = world.geometry
    grid = geometry.grid().reshape(-1, 2)
    free = np.array([geometry.free(point) for point in grid])
    speed = world.config.aircraft.speed_mps
    observed, frames, decisions = log["points"], log["frames"], log["decisions"]
    forecast_errors = np.array([frame["forecast"]["absolute_error_db"] for frame in frames[1:]])
    forecast_labels = np.array([frame["forecast"]["support"] for frame in frames[1:]], dtype=str)
    trajectory = [dict(dbm=world.truth(grid, frame["t"]).round(2).tolist(),
                       sources=world.source_positions_for_evaluation(frame["t"])) for frame in frames]
    reviews = []
    for decision in decisions:
        now = decision["t"]
        scores = []
        for candidate in decision["candidates"]:
            xy, clocks, weights = quadrature(candidate["route"], now, speed)
            scores.append(_truth_q(world, zip(xy, clocks, weights)) + float(weights.sum()))
        oracle = min(range(len(scores)), key=scores.__getitem__)
        selected = decision["selected"]
        errors = np.abs(world.truth(grid, now) - decision["gp"]["dbm"])
        support = np.asarray(decision["gp"]["support"])
        reviews.append(dict(t=now, selected=selected, n_routes=len(scores), oracle=oracle,
                            truth_rank_cost_s=scores, discrepancy=selected["local"] != selected["full"],
                            selected_truth_cost_s={mode: scores[index] for mode, index in selected.items()},
                            split_coverage=decision["split_coverage"], common_until_s=decision["common_until_s"],
                            map_error=_bucket(errors[free], support[free], enumerate(SOURCES))))
    exposure = _truth_q(world, traversed)
    return dict(metrics=dict(stop=log["stop"], T=frames[-1]["t"], Q=exposure, T_plus_Q=frames[-1]["t"] + exposure,
                             length_m=frames[-1]["flown_m"], energy_wh=log["energy_wh"], observations=len(observed)),
                point_forecast_error=_bucket(forecast_errors, forecast_labels, ((label, label) for label in SOURCES)),
                reviews=reviews, trajectory=trajectory)


def run():
    OUT.mkdir(exist_ok=False)
    contract = CONTRACT.read_bytes()
    (OUT / "contract.md").write_bytes(contract)
    code = [Path(__file__), TEMPLATE, *[Path(__file__).with_name(name + ".py") for name in (
        "passive_map", "config", "environment", "motion", "planning", "point_gp", "propagation", "buildings")]]
    frozen = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in code}
    _write(OUT / "freeze.json", dict(contract_sha256=hashlib.sha256(contract).hexdigest(), implementation=frozen,
                                     world_seed=WORLD_SEED, noise_seed=NOISE_SEED, modes=MODES, reviews_s=REVIEWS))
    progress = OUT / "progress.jsonl"

    def mark(status, **fields):
        with progress.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(status=status, **fields), allow_nan=False) + "\n")

    mark("started")
    try:
        flights = {}
        for mode in MODES:
            world = _make_world()
            log, traversed = _flight(mode, world, lambda arm, clock, count, points: mark(
                "decision", mode=arm, time_s=clock, candidates=count, observations=points))
            flights[mode] = (log, traversed, world)
            _write(OUT / f"public_{mode}.json", log)
            mark("flight_closed", mode=mode, reason=log["stop"])
        arms = {}
        for mode, (log, traversed, world) in flights.items():
            arms[mode] = dict(log=log, evaluation=_evaluate(world, log, traversed))
        map_config = world.geometry.config
        payload = dict(grid=dict(rows=map_config.rows, cols=map_config.columns, spacing=map_config.resolution_m,
                                 obstacles=map_config.obstacles_m, start=map_config.start_m, goal=map_config.goal_m),
                       arms=arms)
        results = {arm: {key: value for key, value in data["evaluation"].items() if key != "trajectory"}
                   for arm, data in arms.items()}
        _write(OUT / "results.json", results)
        serialized = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
        with (OUT / "index.html").open("x", encoding="utf-8") as stream:
            stream.write(TEMPLATE.read_text(encoding="utf-8").replace("{{REPLAY_DATA}}", serialized))
    except Exception as exc:
        mark("failed", error=str(exc))
        raise
    mark("complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.parse_args()
    run()
