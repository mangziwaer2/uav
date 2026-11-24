"""Run the independently authorized P3 fixed-review pricing comparison once."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .motion import advance_route
from .passive_map import FixedPriorPosterior, LIMIT_S, SOURCES, candidates, quadrature
from .passive_map_pilot import REVIEWS, TEMPLATE, _truth_q, _write
from .passive_pricing import POLICIES, compare_prices
from .passive_trigger_pilot import _world
from .planning import path_length


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/passive_pricing_P3_20261123"
CONTRACT = ROOT / "docs/passive_pricing_P3_contract.md"
SEEDS = (43401, 43411)
LABELS = dict(shortest="Shortest", mean="GP Mean", risk="GP Risk")


def _flight(world, policy, report):
    geometry, aircraft, settings = world.geometry, world.config.aircraft, world.config.belief
    position, goal = list(geometry.config.start_m), list(geometry.config.goal_m)
    cap, speed = 1.25 * math.dist(position, goal), aircraft.speed_mps
    grid = geometry.grid().reshape(-1, 2)
    now, flown, energy = 0.0, 0.0, aircraft.initial_energy_wh
    history, frames, decisions, traversed = [], [], [], []
    incumbent, forecast, moved = [], None, [position]
    review, stop = 0, "time_limit"
    while True:
        sample = asdict(world.observe(position, now))
        if forecast is not None:
            forecast["absolute_error_db"] = abs(forecast["dbm"] - sample["interference_dbm"])
        history.append(sample)
        frames.append(dict(t=now, xy=position, flown_m=flown, moved=moved, forecast=forecast,
                           measured_dbm=sample["interference_dbm"]))
        if position == goal or now >= LIMIT_S:
            stop = "arrived" if position == goal else "time_limit"
            break

        if review < len(REVIEWS) and now == REVIEWS[review]:
            routes = candidates(position, goal, incumbent, geometry, aircraft, now, flown, energy, cap)
            if not routes:
                stop = "no_legal_remaining_route"
                break
            if len(routes) > 126:
                raise RuntimeError("P3 candidate budget exceeded")
            model = FixedPriorPosterior(history, geometry, now, settings)
            comparison = compare_prices(routes, model, now, speed)
            incumbent = routes[comparison["selected"][policy]]
            mean, std, support = model.predict(grid, np.full(len(grid), now))
            decisions.append(dict(t=now, xy=position, prefix_size=len(history), flown_m=flown, energy_wh=energy,
                                  **comparison, gp=dict(dbm=mean.round(3).tolist(), std_db=std.round(3).tolist(),
                                                        support=support.tolist())))
            report("decision", t=now, candidates=len(routes), selected=comparison["selected"])
            review += 1

        next_review = REVIEWS[review] if review < len(REVIEWS) else LIMIT_S
        duration = min(1.0, next_review - now, path_length(incumbent) / speed)
        if duration <= 0:
            raise RuntimeError("execution cannot advance")
        remaining, moved = advance_route(incumbent, duration * speed)
        elapsed = 0.0
        for first, second in zip(moved, moved[1:]):
            if not geometry.edge_free(first, second):
                raise RuntimeError("executed segment violated geometry")
            distance = math.dist(first, second)
            step = distance / speed
            traversed.append(((np.asarray(first) + second) / 2, now + elapsed + step / 2, step))
            elapsed += step
            flown += distance
            energy -= step * aircraft.flight_power_w / 3600
        if flown > cap + 1e-7 or energy < aircraft.reserve_energy_wh - 1e-8:
            raise RuntimeError("executed segment violated resource constraints")
        now = next_review if duration == next_review - now else now + duration
        position, incumbent = remaining[0], remaining
        mean, std, source = model.predict([position], [now])
        forecast = dict(issued_s=model.now, dbm=float(mean[0]), support=SOURCES[source[0]],
                        observation_std_db=float(np.hypot(std[0], settings.measured_error_db)))
    if len(history) > 91:
        raise RuntimeError("P3 observation budget exceeded")
    return dict(mode=policy, frames=frames, decisions=decisions, points=history, stop=stop,
                energy_wh=aircraft.initial_energy_wh - energy), traversed


def _evaluate(world, log, traversed):
    grid = world.geometry.grid().reshape(-1, 2)
    speed = world.config.aircraft.speed_mps
    frames = log["frames"]
    reviews = []
    for decision in log["decisions"]:
        scores = []
        for row in decision["candidates"]:
            points, clocks, weights = quadrature(row["route"], decision["t"], speed)
            scores.append(float(weights.sum() + _truth_q(world, zip(points, clocks, weights))))
        oracle = min(range(len(scores)), key=lambda idx: (scores[idx], decision["candidates"][idx]["time_s"], idx))
        selected = decision["selected"]
        reviews.append(dict(t=decision["t"], n_routes=len(scores), selected=selected, oracle=oracle,
                            truth_costs=scores, oracle_gap={mode: scores[idx] - scores[oracle]
                                                           for mode, idx in selected.items()},
                            selected_coverage={mode: decision["candidates"][idx]["coverage"]
                                               for mode, idx in selected.items()},
                            split_vs_mean=decision["split_vs_mean"]))
    exposure = _truth_q(world, traversed)
    metrics = dict(stop=log["stop"], T=frames[-1]["t"], Q=exposure, T_plus_Q=frames[-1]["t"] + exposure,
                   length_m=frames[-1]["flown_m"], energy_wh=log["energy_wh"], observations=len(log["points"]))
    trajectory = [dict(dbm=world.truth(grid, frame["t"]).round(2).tolist(),
                       sources=world.source_positions_for_evaluation(frame["t"])) for frame in frames]
    return dict(metrics=metrics, reviews=reviews, trajectory=trajectory)


def _replay(seed, arms, grid):
    options = "".join(f'<option value="{mode}">{label}</option>' for mode, label in LABELS.items())
    links = "".join(f'<a href="public_{seed}_{mode}.json">{label} 公开日志</a>' for mode, label in LABELS.items())
    edits = {
        "P1 · 被动全图估计回放": f"P3 · 固定复核与三种定价 · {seed}",
        '<option value="local">Local · 近场 GP</option><option value="full">Full · 全图 GP</option>': options,
        '<a href="public_local.json">Local 公开日志</a><a href="public_full.json">Full 公开日志</a>': links,
        'mode==="local"?"full":"local"': 'mode==="mean"?"risk":"mean"',
        'mode==="local"?0:1,std': '1,std',
        'selected.coverage.unknown+(mode==="local"?selected.coverage.extended:0)': 'selected.coverage.unknown',
        '另一评分器首选': '同状态 GP Mean 首选（Mean 组显示 Risk）',
        '未用于该组定价': 'GP 未支持',
    }
    page = TEMPLATE.read_text(encoding="utf-8")
    for before, after in edits.items():
        if before not in page:
            raise ValueError(f"P3 replay marker missing: {before}")
        page = page.replace(before, after)
    label_line = next(line for line in page.splitlines() if line.startswith("const labels"))
    page = page.replace(label_line, f"const labels = {json.dumps(LABELS)};")
    payload = json.dumps(dict(grid=grid, arms=arms), allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    return page.replace("{{REPLAY_DATA}}", payload)


def run(authorized_digest):
    try:
        contract = CONTRACT.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError("execution requires the authorized P3 contract; archived contracts are closed") from exc
    digest = hashlib.sha256(contract).hexdigest()
    if authorized_digest != digest:
        raise ValueError("execution requires the SHA-256 of the authorized P3 contract")
    OUT.mkdir(exist_ok=False)
    (OUT / "contract.md").write_bytes(contract)
    source = Path(__file__).parent
    files = (Path(__file__), TEMPLATE, ROOT / "requirements.txt", *[source / (name + ".py") for name in (
        "passive_pricing", "passive_trigger_pilot", "passive_trigger", "passive_map_pilot", "passive_map",
        "barbosa_static", "point_gp", "environment", "config", "motion", "planning", "propagation", "buildings")])
    _write(OUT / "freeze.json", dict(contract_sha256=digest, seeds=SEEDS, policies=POLICIES, reviews_s=REVIEWS,
                                    implementation={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                                    for path in files}))

    def mark(status, **fields):
        with (OUT / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(status=status, **fields), allow_nan=False) + "\n")

    mark("started")
    results = {}
    try:
        for seed in SEEDS:
            flights = {}
            for policy in POLICIES:
                world = _world(seed)
                log, traversed = _flight(world, policy, lambda status, **data: mark(
                    status, seed=seed, policy=policy, **data))
                _write(OUT / f"public_{seed}_{policy}.json", log)
                flights[policy] = (world, log, traversed)
                mark("flight_closed", seed=seed, policy=policy, reason=log["stop"])
            arms, results[seed] = {}, {}
            for policy, (world, log, traversed) in flights.items():
                evaluation = _evaluate(world, log, traversed)
                arms[policy] = dict(log=log, evaluation=evaluation)
                results[seed][policy] = {key: value for key, value in evaluation.items() if key != "trajectory"}
            config = world.geometry.config
            grid = dict(rows=config.rows, cols=config.columns, spacing=config.resolution_m,
                        obstacles=config.obstacles_m, start=config.start_m, goal=config.goal_m)
            (OUT / f"replay_{seed}.html").write_text(_replay(seed, arms, grid), encoding="utf-8")
        _write(OUT / "results.json", results)
        links = "".join(f'<li><a href="replay_{seed}.html">{seed} 左右双图</a></li>' for seed in SEEDS)
        (OUT / "index.html").write_text(
            '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>P3 · 三种定价</title>'
            '<h1>P3 · 固定复核与三种定价</h1><ul>' + links + '</ul><a href="contract.md">冻结合同</a> · '
            '<a href="results.json">事后结果</a> · <a href="progress.jsonl">运行记录</a></html>', encoding="utf-8")
    except Exception as exc:
        mark("failed", error=str(exc))
        raise
    mark("complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorized-contract-sha256", required=True)
    run(parser.parse_args().authorized_contract_sha256)
