"""Run the frozen P2 four-flight comparison once and publish offline replays."""

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .config import ProjectConfig
from .environment import ElectromagneticWorld
from .motion import advance_route
from .passive_map import FixedPriorPosterior, LIMIT_S, SOURCES, candidates, compare_routes, _common_time
from .passive_map_pilot import _evaluate
from .passive_trigger import FINAL_S, WINDOW_S, has_alternative, review_reason, risk_signal
from .planning import path_length


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/passive_trigger_P2_20261123"
CONTRACT = ROOT / "docs/passive_trigger_P2_contract.md"
TEMPLATE = Path(__file__).with_name("passive_map_replay.html")
SEEDS = (43301, 43311)
POLICIES = ("fixed", "event")


def _world(seed):
    rng = np.random.default_rng(seed)
    fields = tuple(dict(center_m=[float(rng.uniform(low, high)), float(rng.uniform(160, 440))],
                        velocity_mps=rng.uniform(-1.25, 1.25, 2).tolist(),
                        sigma_m=float(rng.uniform(60, 100)), peak_dbm=float(rng.uniform(-75, -65)),
                        start_seconds=start, end_seconds=end)
                   for low, high, start, end in ((200, 320, 0, 90), (360, 520, 12, 42), (530, 680, 24, 90)))
    config = ProjectConfig(seed=seed + 1)
    return ElectromagneticWorld(replace(config, electromagnetic=replace(
        config.electromagnetic, fields=fields, background_dbm=-100, measurement_noise_db=1)))


def fly(world, policy, report):
    geometry, aircraft, settings = world.geometry, world.config.aircraft, world.config.belief
    position, goal = list(geometry.config.start_m), list(geometry.config.goal_m)
    speed, cap = aircraft.speed_mps, 1.25 * math.dist(position, goal)
    grid = geometry.grid().reshape(-1, 2)
    now, flown, energy = 0.0, 0.0, aircraft.initial_energy_wh
    history, frames, decisions, monitors, traversed = [], [], [], [], []
    used, incumbent, forecast, moved = set(), [], None, [position]
    enum_count, stop = 0, "time_limit"
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

        routes = None
        signal = dict(comparable=0, witness=None)
        eligible = False
        if now <= FINAL_S:
            current = FixedPriorPosterior(history, geometry, now, settings)
            if now > 0:
                signal = risk_signal(incumbent, now, speed, current, committed)
            if signal["witness"] or review_reason(now, used, policy, False):
                routes = candidates(position, goal, incumbent, geometry, aircraft, now, flown, energy, cap)
                enum_count += 1
                eligible = bool(incumbent and signal["witness"] and has_alternative(routes, incumbent, speed))
        reason = review_reason(now, used, policy, eligible)
        if 0 < now <= FINAL_S:
            monitors.append(dict(t=now, reference_s=committed.now, eligible=eligible, action=reason, **signal))
        if reason:
            if not routes:
                stop = "no_legal_remaining_route"
                break
            action = compare_routes(routes, current, now, speed)
            selected = routes[action["selected"]["full"]]
            changed = bool(incumbent and _common_time(incumbent, selected, speed)
                           < max(path_length(incumbent), path_length(selected)) / speed - 1e-8)
            incumbent, committed = selected, current
            if now > 0:
                used.add(math.ceil(now / WINDOW_S))
            mean, std, support = current.predict(grid, np.full(len(grid), now))
            decisions.append(dict(t=now, xy=position, reason=reason, changed=changed, signal=signal,
                                  prefix_size=len(history), **action,
                                  gp=dict(dbm=mean.round(3).tolist(), std_db=std.round(3).tolist(),
                                          support=support.tolist())))
            report(policy=policy, t=now, reason=reason, changed=changed, candidates=len(routes))

        duration = min(1.0, LIMIT_S - now, path_length(incumbent) / speed)
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
        now += duration
        position, incumbent = remaining[0], remaining
        active = current if now - duration <= FINAL_S else committed
        mean, std, support = active.predict([position], [now])
        forecast = dict(issued_s=active.now, dbm=float(mean[0]),
                        observation_std_db=float(np.hypot(std[0], settings.measured_error_db)), support=SOURCES[support[0]])

    if len(monitors) > 48 or enum_count > 49 or len(decisions) > 5:
        raise ValueError("P2 resource limit exceeded")
    return (dict(policy=policy, frames=frames, decisions=decisions, points=history, monitors=monitors,
                 enum_count=enum_count, stop=stop, energy_wh=aircraft.initial_energy_wh - energy), traversed)


def _write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _replay(seed, arms, grid):
    replacements = {
        "P1 · 被动全图估计回放": f"P2 · 复核时机回放 · {seed}",
        "Local · 近场 GP": "Fixed · 固定时刻",
        "Full · 全图 GP": "Event · 事件提前",
        "<span><i class=\"dash\"></i>另一评分器首选</span>": "",
        'mode==="local"?0:1,std': '1,std',
        'decision.candidates[decision.selected[mode]]': 'decision.candidates[decision.selected.full]',
        'selected.coverage.unknown+(mode==="local"?selected.coverage.extended:0)': 'selected.coverage.unknown',
        ' · ${decision.prefix_size} 个原始点': ' · ${{initial:"起飞",scheduled:"定时复核",early_risk:"风险提前"}[decision.reason]} · ${decision.prefix_size} 个原始点',
        'href="public_local.json">Local 公开日志': f'href="public_{seed}_fixed.json">Fixed 公开日志',
        'href="public_full.json">Full 公开日志': f'href="public_{seed}_event.json">Event 公开日志',
    }
    page = TEMPLATE.read_text(encoding="utf-8")
    for before, after in replacements.items():
        if before not in page:
            raise ValueError(f"P2 template marker missing: {before}")
        page = page.replace(before, after)
    page = "\n".join(line for line in page.splitlines() if "if (decision.selected[mode]" not in line)
    payload = json.dumps(dict(grid=grid, arms=dict(local=arms["fixed"], full=arms["event"])),
                         allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    (OUT / f"replay_{seed}.html").write_text(page.replace("{{REPLAY_DATA}}", payload), encoding="utf-8")


def run():
    OUT.mkdir(exist_ok=False)
    contract = CONTRACT.read_bytes()
    (OUT / "contract.md").write_bytes(contract)
    source = Path(__file__).parent
    files = (Path(__file__), TEMPLATE, ROOT / "requirements.txt", *[source / (name + ".py") for name in (
        "passive_trigger", "passive_map_pilot", "passive_map", "barbosa_static", "point_gp", "environment",
        "config", "motion", "planning", "propagation", "buildings")])
    _write(OUT / "freeze.json", dict(seeds=SEEDS, policies=POLICIES, contract_sha256=hashlib.sha256(contract).hexdigest(),
                                    implementation={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                                    for path in files}))
    results = {}

    def mark(status, **data):
        with (OUT / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(status=status, **data), allow_nan=False) + "\n")

    mark("started")
    try:
        for seed in SEEDS:
            flights = {}
            for policy in POLICIES:
                world = _world(seed)
                log, traversed = fly(world, policy, lambda **data: mark("decision", seed=seed, **data))
                _write(OUT / f"public_{seed}_{policy}.json", log)
                flights[policy] = (world, log, traversed)
                mark("flight_closed", seed=seed, policy=policy, reason=log["stop"])
            arms, results[seed] = {}, {}
            for policy, (world, log, traversed) in flights.items():
                evaluation = _evaluate(world, log, traversed)
                arms[policy] = dict(log=log, evaluation=evaluation)
                results[seed][policy] = dict(metrics=evaluation["metrics"],
                    events=[{key: event[key] for key in ("t", "reason", "changed")} for event in log["decisions"]],
                    comparable_checks=sum(item["comparable"] > 0 for item in log["monitors"]),
                    risk_checks=sum(item["witness"] is not None for item in log["monitors"]),
                    eligible_checks=sum(item["eligible"] for item in log["monitors"]),
                    enum_count=log["enum_count"])
            config = world.geometry.config
            _replay(seed, arms, dict(rows=config.rows, cols=config.columns, spacing=config.resolution_m,
                                     obstacles=config.obstacles_m, start=config.start_m, goal=config.goal_m))
        _write(OUT / "results.json", results)
        rows = []
        for seed, pair in results.items():
            fixed, event = (pair[policy]["metrics"]["T_plus_Q"] for policy in POLICIES)
            reviews = ", ".join(f"{item['t']:g}" for item in pair["event"]["events"])
            early = sum(item["reason"] == "early_risk" for item in pair["event"]["events"])
            rows.append(f'<tr><td><a href="replay_{seed}.html">{seed} 左右双图</a></td><td>{fixed:.2f}</td>'
                        f'<td>{event:.2f}</td><td>{event-fixed:+.2f}</td><td>{reviews} s</td><td>{early}</td></tr>')
        head = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>P2 · 复核时机实验</title><style>body{font:16px/1.6 system-ui,sans-serif;background:#f7faf9;color:#233537;max-width:930px;margin:32px auto;padding:0 16px}a{color:#196763}div{overflow:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:8px;border-bottom:1px solid #c5d4d1;white-space:nowrap}</style>
<h1>P2 · 固定时刻 vs 风险提前复核</h1><p><a href="../index.html">实验总入口</a> · <a href="contract.md">冻结合同</a> · <a href="results.json">结果数据</a> · <a href="progress.jsonl">运行记录</a></p>
<p>同一时空 GP、同一数值选路和同一复核次数上限。差额为 Event 减 Fixed；负值表示该场景提前复核降低 T+Q。</p>
<div><table><thead><tr><th>场景回放</th><th>Fixed T+Q</th><th>Event T+Q</th><th>差额</th><th>Event 复核时刻</th><th>提前次数</th></tr></thead><tbody>'''
        (OUT / "index.html").write_text(head + "".join(rows) + "</tbody></table></div></html>", encoding="utf-8")
    except Exception as exc:
        mark("failed", error=str(exc))
        raise
    mark("complete")


if __name__ == "__main__":
    run()
