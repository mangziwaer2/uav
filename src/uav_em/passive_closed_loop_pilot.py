"""Freeze and execute the single authorized P4 development pair."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .environment import ElectromagneticWorld
from .passive_closed_loop import fly
from .passive_map_pilot import ROOT, TEMPLATE, _truth_q, _write
from .passive_trigger_pilot import _world


OUT = ROOT / "outputs/passive_closed_loop_P4_20261124"
CONTRACT = ROOT / "docs/passive_closed_loop_P4_contract.md"
SEED = 43501
POLICIES = ("fixed", "event")
LABELS = dict(fixed="Fixed · 定时复核", event="Event · 风险提前复核")
MODULES = ("passive_closed_loop", "passive_closed_loop_pilot", "passive_map", "passive_map_pilot",
           "passive_trigger", "passive_trigger_pilot", "passive_pricing_integral", "passive_pricing",
           "barbosa_static", "point_gp", "environment", "config", "motion", "planning", "propagation", "buildings")


def _scenario():
    base = _world(SEED).config
    buildings = tuple(dict(id=f"B{index}", height_m=height,
                           footprint_m=[[left, bottom], [right, bottom], [right, top], [left, top]])
                      for index, ((left, bottom, right, top), height) in enumerate(
                          zip(base.map.obstacles_m, (40, 45)), start=1))
    return replace(base, map=replace(base.map, geometry_mode="fixed_altitude", obstacles_m=(), buildings=buildings))


def _executed_samples(traversed):
    for first, second, start, duration in traversed:
        count = math.ceil(duration / 0.25)
        if count:
            phase = (np.arange(count) + 0.5) / count
            origin = np.asarray(first)
            for fraction in phase:
                yield origin + fraction * (np.asarray(second) - origin), start + fraction * duration, duration / count


def _evaluate(world, log, traversed):
    frames = log["frames"]
    total_time, q = frames[-1]["t"], _truth_q(world, _executed_samples(traversed))
    length = frames[-1]["flown_m"]
    aircraft = world.config.aircraft
    energy = aircraft.initial_energy_wh - log["energy_left_wh"]
    checks = [frame["monitor"] for frame in frames if "monitor" in frame]
    events = [dict(t=item["t"], reason=item["reason"], changed=item["changed"],
                   candidate_count=len(item["candidates"]), qualified=item["qualified"],
                   committed_s=item["committed_s"]) for item in log["decisions"]]
    metrics = dict(T=total_time, Q=q, T_plus_Q=total_time + q, length_m=length, cap_m=log["cap_m"], energy_wh=energy,
                   remaining_wh=log["energy_left_wh"], stop=log["stop"],
                   legal=(length <= log["cap_m"] + 1e-7 and log["energy_left_wh"] >= aircraft.reserve_energy_wh
                          and all(world.geometry.edge_free(first, second) for first, second, _, _ in traversed)))
    monitor = dict(checks=len(checks), no_current_support=sum(check["current_supported"] == 0 for check in checks),
                   with_common=sum(check["common"] > 0 for check in checks),
                   with_new_only=sum(check["new_only"] > 0 for check in checks),
                   crossings=[dict(t=frame["t"], slot_spent=frame["monitor"]["slot_spent"],
                                   cooldown=frame["monitor"]["cooldown"], **frame["monitor"]["witness"])
                              for frame in frames if frame.get("monitor", {}).get("witness")])
    grid = world.geometry.grid().reshape(-1, 2)
    trajectory = [dict(dbm=world.truth(grid, frame["t"]).round(2).tolist(),
                       sources=world.source_positions_for_evaluation(frame["t"])) for frame in frames]
    return dict(metrics=metrics, events=events, monitor=monitor, resources=log["resources"], trajectory=trajectory)


def _replay(arms, grid):
    options = "".join(f'<option value="{mode}">{label}</option>' for mode, label in LABELS.items())
    links = "".join(f'<a href="public_{mode}.json">{label} 日志</a>' for mode, label in LABELS.items())
    page = TEMPLATE.read_text(encoding="utf-8")
    first = page.index('  if (!decision) {')
    last = page.index('\n}', first)
    body = '''  const std=powerLayer.value==="std", monitor=frame.monitor;
  heat(belief,std?frame.gp.std_db:frame.gp.dbm,frame.gp.support,1,std);
  endpoints(belief);
  if (decision?.t===frame.t && decision.changed) path(belief,decision.candidates[decision.reference].route,"#ec9dff",3,true);
  path(belief,frame.remaining,"#a0f5e5",4);
  for (const point of log.points.slice(0,frame.prefix_size)) dot(belief,point.position_m,3.5,color(point.interference_dbm,-105,-65));
  dot(belief,frame.xy,7,"#fff");
  const reason={initial:"起飞规划",scheduled:"定时复核",early_risk:"提前风险复核"};
  const action=decision?.t===frame.t ? ` · ${reason[decision.reason]}${decision.reason==="initial"?"":decision.changed?" / 换路":" / 保持"}` : "";
  const risk=monitor ? ` · 参照 ${fmt(monitor.reference_s,0)} s / ${monitor.witness?"CVaR 越线":monitor.current_supported?"未越线":"无支持"}${monitor.slot_spent?" / 本窗已复核":""}` : "";
  const error=frame.forecast ? ` · 上一步点预测误差 ${fmt(frame.forecast.error_db,1)} dB` : "";
  byId("belief-info").textContent=`${frame.prefix_size} 个原始点${risk}${action}${error}${std?" · 色标 0–15 dB":""}`;'''
    page = page[:first] + body + page[last:]
    edits = {
        "P1 · 被动全图估计回放": f"P4 · 接入 LLM 前的数值闭环 · {SEED}",
        '<option value="local">Local · 近场 GP</option><option value="full">Full · 全图 GP</option>': options,
        '<a href="public_local.json">Local 公开日志</a><a href="public_full.json">Full 公开日志</a>': links,
        'mode=arm.value, other=mode==="local"?"full":"local",': 'mode=arm.value,',
        'const labels = {local:"Local · 近场 GP", full:"Full · 全图 GP"};': f'const labels = {json.dumps(LABELS)};',
        'time_limit:"到时限",': 'time_limit:"到时限", point_budget:"点测预算耗尽", compute_budget:"计算预算耗尽",',
        "另一评分器首选": "换路前余程",
        "冻结决策快照 · 仅当时信息": "同帧 GP 和未飞余程 · 仅当时信息",
        "未用于该组定价": "GP 未支持",
    }
    for before, after in edits.items():
        if before not in page:
            raise ValueError(f"P4 replay marker missing: {before}")
        page = page.replace(before, after)
    payload = json.dumps(dict(grid=grid, arms=arms), allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    return page.replace("{{REPLAY_DATA}}", payload)


def _freeze_record():
    source = Path(__file__).parent
    paths = (TEMPLATE, ROOT / "requirements.txt", *[source / (name + ".py") for name in MODULES])
    return dict(contract_sha256=hashlib.sha256(CONTRACT.read_bytes()).hexdigest(), seed=SEED, noise_seed=SEED + 1,
                policies=list(POLICIES), implementation={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                                         for path in paths})


def freeze():
    OUT.mkdir(exist_ok=False)
    (OUT / "contract.md").write_bytes(CONTRACT.read_bytes())
    _write(OUT / "freeze.json", _freeze_record())


def run(authorized_digest):
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    if (frozen != _freeze_record() or authorized_digest != frozen["contract_sha256"]
            or CONTRACT.read_bytes() != (OUT / "contract.md").read_bytes()):
        raise ValueError("execution requires the frozen P4 contract and unchanged implementation")
    progress = OUT / "progress.jsonl"
    with progress.open("x", encoding="utf-8") as stream:
        stream.write('{"status":"started"}\n')

    def report(status, **data):
        with progress.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(status=status, **data), allow_nan=False) + "\n")

    try:
        config = _scenario()
        _write(OUT / "public_inputs.json", dict(map=asdict(config.map), aircraft=asdict(config.aircraft),
                                                belief=asdict(config.belief), measurement="position_time_noisy_total_dbm"))
        flights = {}
        for policy in POLICIES:
            world = ElectromagneticWorld(config)
            started = time.perf_counter()
            log, traversed = fly(world.geometry, config.aircraft, config.belief, world.observe, policy)
            elapsed = time.perf_counter() - started
            _write(OUT / f"public_{policy}.json", log)
            flights[policy] = world, log, traversed
            report("flight_closed", policy=policy, stop=log["stop"], elapsed_s=elapsed, **log["resources"])
        arms = {policy: dict(log=log, evaluation=_evaluate(world, log, traversed))
                for policy, (world, log, traversed) in flights.items()}
        results = {policy: {key: value for key, value in record["evaluation"].items() if key != "trajectory"}
                   for policy, record in arms.items()}
        _write(OUT / "results.json", results)
        layout = config.map
        grid = dict(rows=layout.rows, cols=layout.columns, spacing=layout.resolution_m,
                    obstacles=[[item["footprint_m"][0][0], item["footprint_m"][0][1],
                                item["footprint_m"][2][0], item["footprint_m"][2][1]] for item in layout.buildings],
                    start=layout.start_m, goal=layout.goal_m)
        (OUT / "index.html").write_text(_replay(arms, grid), encoding="utf-8")
        report("complete")
    except Exception as error:
        report("failed", error=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze", action="store_true")
    actions.add_argument("--authorized-contract-sha256")
    arguments = parser.parse_args()
    if arguments.freeze:
        freeze()
    else:
        run(arguments.authorized_contract_sha256)
