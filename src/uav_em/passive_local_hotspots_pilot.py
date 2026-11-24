"""Freeze and run the single-scenario E2 local-hotspot comparison."""

import argparse
from dataclasses import replace
import json
import math

import numpy as np

from .config import ProjectConfig
from .environment import ElectromagneticWorld
from .passive_closed_loop import MAX_POINTS, MAX_QUERIES
from .passive_closed_loop_pilot import _executed_samples
from .passive_decision_replay import DATA, _payload, render
from .passive_event_choice import ALERT_DBM, HORIZON_S, MAX_EVENTS, prices
from .passive_event_choice_pilot import (
    MAX_REPLY_BYTES, MAX_REQUEST_BYTES, MAX_TOTAL_WAIT_S, MAX_WAIT_S, ROOT, _layout, _public, _request, _sha, fly,
)
from .passive_map import LIMIT_S, candidates
from .passive_map_pilot import _write
from .passive_model_advice_publish import evaluate
from .planning import path_length


OUT = ROOT / "outputs/passive_local_hotspots_E2_20261124"
CONTRACT = ROOT / "docs/passive_local_hotspots_E2_contract.md"
SEED = 43711
LABELS = dict(numerical="N · 完整航时 + 共同窗暴露", llm="LLM · 事件选路")
PROMPT = """Independently select a complete route for a simulated civilian UAV using only the JSON below. Do not use tools, workspace, other sessions or outside information. Minimize actual remaining flight time T plus integrated linear power exposure Q. Every listed route is legal and reaches the goal within remaining resources. Select exactly one listed route ID, including the incumbent when warranted.
Rows are [id, xy_path, full_remaining_T_s, window_T_s, common_Q, compared_s, unknown_s, individual_Q, edit, delta_window_TQ]. The shared numeric proxy for the task is full_remaining_T_s + common_Q (when none is priced, only T is comparable). Use the full time difference, not delta_window_TQ, for detour cost. First-waypoint groups are [xy,route_ids]. Recent real readings are [second,x,y,dBm,prior_error_dB]. Larger dBm is stronger. Compare the first direction changes against positions of measured increases and the frozen common Q, and weigh actual extra travel time. Distinguish predicted exposure on common support from any later or unsupported portion. An unsupported point, a shorter route, or changing direction is not proof of lower risk. Do not assume hidden sources, field shapes, or future measurements. Give a short reason referring to the evidence and uncertainty. Return exactly one JSON line: {"request_id":"...","route_id":0,"reason":"..."}. No markdown or abstention.
"""


def scenario():
    rng = np.random.default_rng(SEED)
    fields = tuple(dict(center_m=(np.array(center) + rng.uniform(-5, 5, 2)).tolist(),
                        velocity_mps=drift, sigma_m=sigma, peak_dbm=peak, start_seconds=start, end_seconds=end)
                   for center, drift, sigma, peak, start, end in (
                       ((215, 375), [0.65, -0.15], 43, -64, 0, 90),
                       ((457, 415), [-0.35, 0.75], 39, -70, 16, 54),
                       ((615, 160), [0.45, 0.5], 35, -68, 27, 90),
                   ))
    base = ProjectConfig(seed=SEED + 1)
    return replace(base, map=_layout(base.map), electromagnetic=replace(
        base.electromagnetic, fields=fields, background_dbm=-100, measurement_noise_db=1))


def score_routes(routes, model, now, speed, incumbent=None, child=None):
    rows, _, reference = prices(routes, model, now, speed, incumbent, child)
    for row in rows:
        row["objective_tq"] = row["goal_s"] + (row["costs"]["risk"] - row["time_s"])
    selected = min(range(len(rows)), key=lambda index: (
        round(rows[index]["objective_tq"], 8), rows[index]["goal_s"], index))
    return rows, selected, reference


def _frozen():
    files = [path for path in ROOT.joinpath("src/uav_em").iterdir() if path.suffix in (".py", ".js", ".html")]
    files.append(ROOT / "requirements.txt")
    return dict(contract_sha256=_sha(CONTRACT.read_bytes()), public_sha256=_sha(_payload(_public()).encode()),
                prompt_sha256=_sha(PROMPT.encode()), seed=SEED, noise_seed=SEED + 1, arms=list(LABELS),
                resources=dict(events_per_arm=MAX_EVENTS, point_fits_per_arm=MAX_POINTS, queries_per_arm=MAX_QUERIES,
                               scores_per_arm=1 + MAX_EVENTS, simulation_limit_s=LIMIT_S, wait_s=MAX_WAIT_S,
                               total_wait_s=MAX_TOTAL_WAIT_S, request_bytes=MAX_REQUEST_BYTES,
                               reply_bytes=MAX_REPLY_BYTES, window_s=HORIZON_S),
                implementation={str(path.relative_to(ROOT)): _sha(path.read_bytes()) for path in sorted(files)})


def freeze():
    frozen = _frozen()
    OUT.mkdir(exist_ok=False)
    (OUT / "contract.md").write_bytes(CONTRACT.read_bytes())
    (OUT / "public_inputs.json").write_text(_payload(_public()), encoding="utf-8")
    _write(OUT / "freeze.json", frozen)
    print(frozen["contract_sha256"], flush=True)


def _route_truth(world, route):
    clock, traversed = 0.0, []
    for first, second in zip(route, route[1:]):
        duration = math.dist(first, second) / world.config.aircraft.speed_mps
        traversed.append((first, second, clock, duration))
        clock += duration
    power, above_s = 0.0, 0.0
    for point, sample_s, weight in _executed_samples(traversed):
        dbm = world.truth(point, sample_s)
        power += weight * 10 ** ((dbm - ALERT_DBM) / 10)
        if dbm >= ALERT_DBM:
            above_s += weight
    return dict(route=route, T=clock, Q=power, T_plus_Q=clock + power, above_alert_fraction=above_s / clock)


def _scene_audit(world):
    aircraft, geometry = world.config.aircraft, world.geometry
    start, goal = list(geometry.config.start_m), list(geometry.config.goal_m)
    grid = geometry.grid().reshape(-1, 2)
    free_grid = grid[[geometry.free(point) for point in grid]]
    footprint = [float(np.mean(world.truth(free_grid, clock) >= ALERT_DBM)) for clock in range(61)]
    routes = candidates(start, goal, [], geometry, aircraft, 0.0, 0.0, aircraft.initial_energy_wh,
                        1.25 * math.dist(start, goal))
    shortest = _route_truth(world, min(routes, key=path_length))
    scored = [_route_truth(world, route) for route in routes]
    eligible = [row for row in scored if (1 <= row["T"] - shortest["T"] <= 12
                and row["Q"] <= 0.5 * shortest["Q"] and row["T_plus_Q"] < shortest["T_plus_Q"]
                and row["above_alert_fraction"] <= 0.10)]
    return dict(footprint_free_grid_fraction=footprint, peak_footprint_fraction=max(footprint),
                legal_routes=len(routes), shortest=shortest,
                best_bounded_reference=min(scored, key=lambda row: row["T_plus_Q"]),
                low_exposure_detour=min(eligible, key=lambda row: row["T_plus_Q"]) if eligible else None,
                locality_pass=0.05 <= max(footprint) <= 0.30, detour_pass=bool(eligible))


def run(digest):
    frozen = json.loads((OUT / "freeze.json").read_bytes())
    if (frozen != _frozen() or digest != frozen["contract_sha256"]
            or _sha((OUT / "contract.md").read_bytes()) != digest
            or _sha((OUT / "public_inputs.json").read_bytes()) != frozen["public_sha256"]):
        raise ValueError("E2 requires frozen implementation, contract and public inputs")
    (OUT / "started").touch(exist_ok=False)
    config, resources, logs = scenario(), {"wait_s": 0}, {}
    for arm in ("llm", "numerical"):
        world = ElectromagneticWorld(config)
        log = fly(world, arm, 201, lambda packet: _request(packet, resources, output=OUT, prompt=PROMPT),
                  price=score_routes)
        _write(OUT / f"public_{arm}.json", log)
        logs[arm] = log
    world = ElectromagneticWorld(config)
    arms = {arm: dict(log=logs[arm], evaluation=evaluate(world, logs[arm])) for arm in LABELS}
    result = {arm: dict(metrics=arms[arm]["evaluation"]["metrics"], resources=logs[arm]["resources"], events=[
        {key: event.get(key) for key in ("t", "selected", "reference", "numerical", "changed", "authority", "request_id")}
        for event in logs[arm]["decisions"]]) for arm in LABELS}
    traces = [[(frame["t"], frame["moved"]) for frame in logs[arm]["frames"]] for arm in LABELS]
    _write(OUT / "results.json", dict(seed=SEED, arms=result, scene_audit=_scene_audit(world),
                                     same_trajectory=traces[0] == traces[1], **resources))
    layout = config.map
    grid = dict(rows=layout.rows, cols=layout.columns, spacing=layout.resolution_m, start=layout.start_m,
                goal=layout.goal_m, obstacles=[[*part["footprint_m"][0], *part["footprint_m"][2]]
                                               for part in layout.buildings])
    page = render(DATA + _payload(dict(grid=grid, arms=arms)) + "</script>", LABELS,
                  f"E2 · 局部动态干扰 · {SEED}", "results.json")
    page = page.replace('>原回放</a>', '>结果数据</a> · <a href="contract.md">冻结合同</a>').replace(
        '<th>T (s)</th><th>未知 (s)</th>', '<th>窗口 T (s)</th><th>窗内未知 (s)</th>').replace(
        '<div class="controls">', '<p>表中 T+Q 为窗口代理。数值组选路用完整剩余 T 加表中共同 Q；'
        '全程真实 T+Q 仅事后核算。</p><div class="controls">')
    (OUT / "index.html").write_text(page, encoding="utf-8")
    (OUT / "closed").touch(exist_ok=False)
    print(_payload(dict(status="closed", output=str(OUT))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze", action="store_true")
    actions.add_argument("--authorized-contract-sha256")
    args = parser.parse_args()
    freeze() if args.freeze else run(args.authorized_contract_sha256)
