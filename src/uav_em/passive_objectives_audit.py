"""Compare objective choices at frozen P4/P5 public snapshots without flying."""

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .barbosa_static import gaussian_upper_cvar
from .config import AircraftConfig, BeliefConfig, MapConfig
from .environment import Geometry
from .passive_closed_loop import EPS, _JointSupport
from .passive_map import FixedPriorPosterior, _common_time, candidates
from .passive_map_pilot import ROOT, _write
from .passive_pricing import TAIL_MASS, UNKNOWN_PRICE, upper_tail_power
from .passive_pricing_integral import BOUNDARY_BISECTIONS, _trajectory


OUT = ROOT / "outputs/passive_objectives_P6_20261124"
CONTRACT = ROOT / "docs/passive_objectives_P6_audit_contract.md"
P4 = ROOT / "outputs/passive_closed_loop_P4_20261124"
P5 = ROOT / "outputs/passive_model_advice_P5_20261124"
INPUTS = (P4 / "public_inputs.json", P4 / "public_fixed.json", P5 / "public_numerical.json",
          P5 / "public_checkpoint.json")
OBJECTIVES = ("tq", "barbosa")
MODES = ("guarded", "proxy", "full")
ALPHA_DBM = -80.0
GAMMA_PER_DB = 0.1
STEPS = (0.25, 0.5)


def densities(mean, std):
    cvar_dbm = gaussian_upper_cvar(mean, std, TAIL_MASS)
    return np.stack((1 + upper_tail_power(mean, std), np.exp(GAMMA_PER_DB * np.maximum(cvar_dbm - ALPHA_DBM, 0))))


def _integrals(model, now, track, grid, *, full=True):
    knots, vertices = track

    def predict(offsets):
        positions = np.column_stack([np.interp(offsets, knots, column) for column in vertices.T])
        return model.predict(positions, now + offsets)

    def quadrature(starts, ends):
        nodes = np.stack((starts, (starts + ends) / 2, ends))
        mean, std, _ = predict(nodes.ravel())
        values = densities(mean, std).reshape(len(OBJECTIVES), *nodes.shape)
        weights = np.array([1, 4, 1])[:, None] * (ends - starts) / 6
        return np.sum(values * weights, axis=(1, 2))

    starts, ends = grid[:-1].copy(), grid[1:].copy()
    full_cost = quadrature(starts, ends) if full else None
    supported = predict(grid)[2] < 2
    transitions = np.flatnonzero(supported[:-1] != supported[1:])
    if len(transitions):
        lower, upper = starts[transitions], ends[transitions]
        start_supported = supported[transitions]
        for _ in range(BOUNDARY_BISECTIONS):
            middle = (lower + upper) / 2
            same_side = (predict(middle)[2] < 2) == start_supported
            lower = np.where(same_side, middle, lower)
            upper = np.where(same_side, upper, middle)
        boundary = (lower + upper) / 2
        ends[transitions[start_supported]] = boundary[start_supported]
        starts[transitions[~start_supported]] = boundary[~start_supported]
    active = supported[:-1] | supported[1:]
    starts, ends = starts[active], ends[active]
    unknown = max(0.0, float(knots[-1] - np.sum(ends - starts)))
    proxy_cost = np.full(len(OBJECTIVES), unknown * (1 + UNKNOWN_PRICE))
    if len(starts):
        proxy_cost += quadrature(starts, ends)
    return proxy_cost, full_cost, unknown


def compare_objectives(routes, reference, posterior, now, speed, step_s):
    tracks = [_trajectory(route, speed) for route in routes]
    durations = [float(track[0][-1]) for track in tracks]
    grid = np.union1d(np.arange(0, max(durations), step_s / 2), durations)
    values = [_integrals(posterior, now, track, grid[grid <= duration]) for track, duration in zip(tracks, durations)]
    rows = [dict(route=route, time_s=duration, unknown_s=unknown,
                 proxy=dict(zip(OBJECTIVES, proxy.tolist())), full=dict(zip(OBJECTIVES, full.tolist())))
            for route, duration, (proxy, full, unknown) in zip(routes, durations, values)]

    def best(indices, objective, lane):
        minimum = min(rows[index][lane][objective] for index in indices)
        return min((index for index in indices if rows[index][lane][objective] <= minimum + EPS),
                   key=lambda index: (durations[index], index))

    all_indices = list(range(len(routes)))
    eligible = {objective: [] for objective in OBJECTIVES}
    for index, row in enumerate(rows):
        if reference is None or index == reference:
            continue
        base = rows[reference]
        common = _common_time(routes[index], routes[reference], speed)
        if common >= max(durations[index], durations[reference]) - EPS:
            continue
        row["fork_after_s_vs_incumbent"] = common
        improving = [objective for objective in OBJECTIVES if row["proxy"][objective] < base["proxy"][objective] - EPS]
        if not improving:
            continue
        overlap = grid[grid <= min(durations[index], durations[reference])]
        joint = _JointSupport(posterior, tracks[reference], now)
        trial, _, unknown = _integrals(joint, now, tracks[index], overlap, full=False)
        committed, _, _ = _integrals(_JointSupport(posterior, tracks[index], now), now, tracks[reference], overlap,
                                     full=False)
        row["joint_delta"] = dict(zip(OBJECTIVES, (trial - committed).tolist()))
        row["joint_support_s"] = max(0.0, durations[index] - unknown)
        if common <= EPS:
            row["joint_support_after_fork_s"] = row["joint_support_s"]
        elif common < overlap[-1]:
            tail = np.r_[common, overlap[overlap > common]]
            _, _, tail_unknown = _integrals(joint, now, tracks[index], tail, full=False)
            row["joint_support_after_fork_s"] = max(0.0, durations[index] - tail_unknown)
        else:
            row["joint_support_after_fork_s"] = 0.0
        for objective in improving:
            if row["joint_delta"][objective] < -EPS:
                eligible[objective].append(index)

    choices, forks = {}, {}
    for objective in OBJECTIVES:
        raw = best(all_indices, objective, "proxy")
        guarded = raw if reference is None else (best(eligible[objective], objective, "proxy")
                                                 if eligible[objective] else reference)
        choices[objective] = dict(guarded=guarded, proxy=raw, full=best(all_indices, objective, "full"))
    for mode in MODES:
        left, right = (choices[objective][mode] for objective in OBJECTIVES)
        common = _common_time(routes[left], routes[right], speed)
        forks[mode] = common if common < max(durations[left], durations[right]) - EPS else None
    return dict(rows=rows, choices=choices, fork_after_s=forks)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _validation(log):
    rows = []
    for frame in log["frames"]:
        forecast = frame["forecast"]
        if frame["t"] <= 24 or not forecast:
            continue
        if (not 24 <= forecast["issued_s"] < forecast["arrival_s"] == frame["t"]
                or forecast["xy"] != frame["xy"]):
            raise ValueError("P5 forecast was not issued before the matching new point")
        rows.append(forecast)
    stats = {}
    for source in ("local", "extended", "unknown"):
        subset = [row for row in rows if row["support"] == source]
        stats[source] = dict(n=len(subset), mae_db=float(np.mean([abs(row["error_db"]) for row in subset]))
                             if subset else None,
                             within_model_95=sum(abs(row["error_db"]) <= 1.959964 * row["observation_std_db"]
                                                 for row in subset))
    return stats


def analyze():
    public, p4, p5, checkpoint = map(_load, INPUTS)
    aircraft, settings = AircraftConfig(**public["aircraft"]), BeliefConfig(**public["belief"])
    geometry = Geometry(MapConfig(**public["map"]), aircraft.clearance_m, aircraft.altitude_m,
                        aircraft.vertical_clearance_m)
    frames, cases = [], []
    for stage, log in (("P4", p4), ("P5", p5)):
        for decision in log["decisions"]:
            rows = decision["candidates"]
            frames.append((stage, log, decision["t"], decision.get("model", "nominal"),
                           log["points"][:decision["prefix_size"]], [row["route"] for row in rows],
                           decision["reference"], decision["selected"], [row["costs"]["risk"] for row in rows]))
    if checkpoint["point_measurements"] != p5["points"][:25] or checkpoint["t_s"] != 24:
        raise ValueError("frozen P5 checkpoint does not match operating prefix")
    operating = next(item[3] for item in frames if item[0] == "P5" and item[2] == 24)
    for model, entry in checkpoint["review"].items():
        if model != operating:
            frames.append(("P5 shadow", p5, 24, model, checkpoint["point_measurements"], checkpoint["routes"],
                           checkpoint["incumbent_idx"], entry["chosen"], [row[0] for row in entry["rows"]]))
    if len(frames) != 12 or sum(item[0] == "P5 shadow" for item in frames) != 2:
        raise ValueError("frozen snapshot count changed")
    for stage, log, now, model, points, routes, reference, selected, prices in frames:
        frame = next(entry for entry in log["frames"] if entry["t"] == now)
        energy = aircraft.initial_energy_wh - now * aircraft.flight_power_w / 3600
        legal = candidates(frame["xy"], list(geometry.config.goal_m), routes[reference] if reference is not None else [],
                           geometry, aircraft, now, frame["flown_m"], energy, log["cap_m"])
        if legal != routes or points[-1]["time_seconds"] != now:
            raise ValueError("frozen legal routes or causal prefix do not reconstruct")
        scale = checkpoint["scales_s"][model]
        posterior = FixedPriorPosterior(points, geometry, now, replace(settings, gp_time_scale_seconds=scale))
        fine, coarse = (compare_objectives(routes, reference, posterior, now, aircraft.speed_mps, step) for step in STEPS)
        error = max(abs(row["proxy"]["tq"] - price) for row, price in zip(fine["rows"], prices))
        if error > 1e-7 or fine["choices"]["tq"]["guarded"] != selected:
            raise ValueError("original T+Q prices or legal decision do not reconstruct")
        cases.append(dict(stage=stage, t=now, model=model, reference=reference, reconstruction_error=error,
                          coarse_choices=coarse["choices"], **fine))
    summary = {mode: sum(case["fork_after_s"][mode] is not None for case in cases) for mode in MODES}
    return dict(cases=cases, geometric_disagreements=summary,
                step_choice_changes=sum(case["coarse_choices"] != case["choices"] for case in cases),
                p5_new_forecasts=_validation(p5))


def _page(result):
    rows = []
    for case in result["cases"]:
        left, right = (case["choices"][objective] for objective in OBJECTIVES)
        cells = "".join(f'<td>{left[mode]} / {right[mode]}</td>' for mode in MODES)
        rows.append(f'<tr><td>{case["stage"]}</td><td>{case["t"]:g}</td><td>{case["model"]}</td>'
                    f'<td>{len(case["rows"])}</td>{cells}</tr>')
    head = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>P6 · 目标函数只读对照</title><style>body{font:16px/1.6 system-ui,sans-serif;margin:32px auto;padding:0 16px;max-width:980px;color:#203537}a{color:#176563}div{overflow:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:8px;border-bottom:1px solid #ccd7d7;white-space:nowrap}</style>
<h1>P6 · T+Q 与 Barbosa 风险弧长</h1><p><a href="../index.html">实验总入口</a> · <a href="findings.md">结果解读</a> · <a href="contract.md">冻结合同</a> · <a href="results.json">完整排序依据</a></p>
<p>每格是 T+Q / Barbosa 的路线索引（从 0 开始）。候选和 GP 相同；无新飞行。完整后验列含未获支持的点，不代表应执行。</p>
<div><table><thead><tr><th>公开快照</th><th>时刻 (s)</th><th>GP</th><th>合法候选数</th><th>共同换路规则</th><th>代理最优</th><th>完整后验最优</th></tr></thead><tbody>'''
    return head + "".join(rows) + ('</tbody></table></div><p><a href="../passive_closed_loop_P4_20261124/index.html">P4 原回放</a> · '
                                   '<a href="../passive_model_advice_P5_20261124/index.html">P5 原回放</a></p></html>')


def _freeze_record():
    source = Path(__file__).parent
    modules = ("passive_objectives_audit", "passive_reply_delivery", "passive_model_advice", "passive_closed_loop",
               "passive_map", "passive_pricing",
               "passive_pricing_integral", "barbosa_static", "point_gp", "config", "environment", "buildings",
               "planning", "propagation", "motion", "passive_trigger", "passive_map_pilot")
    paths = (CONTRACT, *INPUTS, ROOT / "requirements.txt", *(source / (name + ".py") for name in modules))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def main(freeze):
    if freeze:
        OUT.mkdir(exist_ok=False)
        (OUT / "contract.md").write_bytes(CONTRACT.read_bytes())
        _write(OUT / "freeze.json", _freeze_record())
        return
    if _load(OUT / "freeze.json") != _freeze_record():
        raise ValueError("P6 frozen inputs or implementation changed")
    if (OUT / "results.json").exists():
        raise ValueError("P6 results already published")
    result = analyze()
    _write(OUT / "results.json", result)
    with (OUT / "index.html").open("x", encoding="utf-8") as stream:
        stream.write(_page(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze", action="store_true")
    actions.add_argument("--analyze", action="store_true")
    main(parser.parse_args().freeze)
