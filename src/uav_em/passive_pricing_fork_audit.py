"""Account for the remaining P3 ranking forks using only frozen public inputs."""

import json
from pathlib import Path

import numpy as np

from .config import AircraftConfig, BeliefConfig, MapConfig
from .environment import Geometry
from .passive_map import FixedPriorPosterior, _common_time
from .passive_pricing import UNKNOWN_PRICE
from .passive_pricing_integral import _integrate, _trajectory
from .passive_pricing_integral_audit import ARMS, OUT as GRID_AUDIT, P3, ROOT, SEEDS, _hash


OUT = ROOT / "outputs/passive_pricing_P3_fork_audit_20261123"
STEP_S = 0.25
POSITION_TOL_M = 1e-7
TIME_TOL_S = 1e-8


def _positions(trajectory, offsets):
    clocks, vertices = trajectory
    return np.column_stack([np.interp(offsets, clocks, axis) for axis in vertices.T])


class _JointSupport:
    def __init__(self, posterior, other, now):
        self.posterior, self.other, self.now = posterior, other, now

    def predict(self, xy, times):
        mean, std, source = self.posterior.predict(xy, times)
        alive = (source < 2) & (times <= self.now + self.other[0][-1])
        joint = np.full(len(times), 2, dtype=np.uint8)
        if alive.any():
            clocks = times[alive]
            other_source = self.posterior.predict(_positions(self.other, clocks - self.now), clocks)[2]
            joint[alive] = np.where(other_source < 2, source[alive], 2)
        return mean, std, joint


def account_fork(routes, selected, model, now, speed):
    """Split the pair's risk-cost difference on the unchanged shared GP grid."""
    first, second = selected["mean"], selected["risk"]
    if first == second:
        raise ValueError("distinct route choices required")
    tracks = [_trajectory(route, speed) for route in routes]
    durations = np.array([track[0][-1] for track in tracks])
    fork = _common_time(routes[first], routes[second], speed)
    grid = np.union1d(np.arange(0, durations.max(), STEP_S / 2), durations)
    tail = np.union1d(grid, [fork])
    pair = {}
    for mode, idx, other in (("mean", first, second), ("risk", second, first)):
        duration = durations[idx]
        original = _integrate(model, now, tracks[idx], grid[grid <= duration])
        times = tail[(tail >= fork) & (tail <= duration)]
        own = _integrate(model, now, tracks[idx], times)
        joint = _integrate(_JointSupport(model, tracks[other], now), now, tracks[idx], times)
        joint_premium = joint["costs"]["risk"] - (1 + UNKNOWN_PRICE) * duration
        pair[mode] = dict(idx=idx, route=routes[idx], time_s=float(duration), costs=original["costs"],
                          tail_unknown_s=max(0.0, own["unknown_s"] - fork),
                          tail_joint_s=max(0.0, duration - joint["unknown_s"]),
                          tail_one_side_s=max(0.0, joint["unknown_s"] - own["unknown_s"]),
                          joint_premium=float(joint_premium),
                          one_side_premium=own["costs"]["risk"] - joint["costs"]["risk"])
    left, right = pair["mean"], pair["risk"]
    delta = dict(flight_time=right["time_s"] - left["time_s"],
                 joint_support=right["joint_premium"] - left["joint_premium"],
                 one_side=right["one_side_premium"] - left["one_side_premium"])
    delta["unknown_baseline"] = UNKNOWN_PRICE * delta["flight_time"]
    delta["risk_total"] = right["costs"]["risk"] - left["costs"]["risk"]
    delta["residual"] = delta["risk_total"] - sum(delta[name] for name in (
        "flight_time", "unknown_baseline", "joint_support", "one_side"))
    delta["without_one_side"] = sum(delta[name] for name in ("flight_time", "unknown_baseline", "joint_support"))
    return dict(fork_s=now + fork, pair=pair, delta=delta)


def validate_common_points(history, model, pair, now, fork, speed, noise_db):
    """Score the old posterior on actually recorded samples on both prefixes."""
    points = [point for point in history if now < point["time_seconds"] <= fork + TIME_TOL_S]
    if not points:
        return []
    clocks = np.array([point["time_seconds"] for point in points])
    positions = np.array([point["position_m"] for point in points])
    shared = np.ones(len(points), dtype=bool)
    for mode in ("mean", "risk"):
        expected = _positions(_trajectory(pair[mode]["route"], speed), clocks - now)
        shared &= np.linalg.norm(positions - expected, axis=1) <= POSITION_TOL_M
    if not shared.any():
        return []
    mean, std, source = model.predict(positions[shared], clocks[shared])
    rows = []
    matched = [point for point, keep in zip(points, shared) if keep]
    for point, predicted, uncertainty, support in zip(matched, mean, std, source):
        measured = point["interference_dbm"]
        rows.append(dict(t_s=point["time_seconds"], xy=point["position_m"], measured_dbm=measured,
                         predicted_dbm=float(predicted), error_db=float(measured - predicted),
                         observation_std_db=float(np.hypot(uncertainty, noise_db)), supported=bool(support < 2)))
    return rows


def analyze():
    prior = GRID_AUDIT / "results.json"
    grid = json.loads(prior.read_bytes())
    for name, digest in grid["source_hashes"].items():
        if _hash(ROOT / "src/uav_em" / name) != digest:
            raise ValueError("P3 grid-audit implementation hash mismatch")
    frozen = json.loads((P3 / "freeze.json").read_bytes())
    if (_hash(P3 / "contract.md") != grid["contract_sha256"]
            or any(_hash(ROOT / name) != digest for name, digest in frozen["implementation"].items())):
        raise ValueError("P3 contract or implementation hash mismatch")
    logs = {}
    for seed in SEEDS:
        for arm in ARMS:
            path = P3 / f"public_{seed}_{arm}.json"
            if _hash(path) != grid["public_log_sha256"][path.name]:
                raise ValueError("P3 public log hash mismatch")
            logs[seed, arm] = json.loads(path.read_bytes())

    aircraft, settings = AircraftConfig(), BeliefConfig()
    geometry = Geometry(MapConfig(), aircraft.clearance_m, aircraft.altitude_m, aircraft.vertical_clearance_m)
    cases = []
    for item in grid["snapshots"]:
        aligned = item["grids"][f"{STEP_S:g}"]
        selected = aligned["selected"]
        if selected["mean"] == selected["risk"]:
            continue
        seed, arm, now = item["seed"], item["arm"], item["t_s"]
        log = logs[seed, arm]
        snapshot = next(row for row in log["decisions"] if row["t"] == now)
        count = snapshot["prefix_size"]
        prefix = log["points"][:count]
        if now == 0 or count != item["prefix_size"] or not prefix or prefix[-1]["time_seconds"] != now:
            raise ValueError("P3 fork scope or prefix mismatch")
        model = FixedPriorPosterior(prefix, geometry, now, settings)
        routes = [row["route"] for row in snapshot["candidates"]]
        account = account_fork(routes, selected, model, now, aircraft.speed_mps)
        for row in account["pair"].values():
            expected = aligned["rows"][row["idx"]]["costs"]
            if any(abs(cost - expected[mode]) > 1e-8 for mode, cost in row["costs"].items()):
                raise ValueError("P3 aligned score cannot be reconstructed")
        later = validate_common_points(log["points"][count:], model, account["pair"], now, account["fork_s"],
                                       aircraft.speed_mps, settings.measured_error_db)
        cases.append(dict(seed=seed, arm=arm, issued_s=now, prefix_size=count,
                          common_readings=later, **account))
    if len(cases) != 6:
        raise ValueError("expected exactly six P3 ranking forks")
    return dict(input_sha256=_hash(prior), source_sha256=_hash(Path(__file__)), step_s=STEP_S, cases=cases,
                summary=dict(one_side_needed=sum(row["delta"]["without_one_side"] > 0 for row in cases),
                             cases_with_common_readings=sum(bool(row["common_readings"]) for row in cases),
                             common_readings=sum(len(row["common_readings"]) for row in cases),
                             supported_readings=sum(reading["supported"] for row in cases
                                                    for reading in row["common_readings"]),
                             max_abs_residual=max(abs(row["delta"]["residual"]) for row in cases)))


def _page(data):
    rows, points = [], []
    for item in data["cases"]:
        seed, arm, now = item["seed"], item["arm"], item["issued_s"]
        delta = item["delta"]
        rows.append(f'<tr><td>{seed} {arm}</td><td>{now:g}</td><td>{item["fork_s"]:.3f}</td>'
                    f'<td>{item["pair"]["risk"]["tail_joint_s"]:.3f}</td><td>{delta["risk_total"]:+.3f}</td>'
                    f'<td>{delta["flight_time"] + delta["unknown_baseline"]:+.3f}</td>'
                    f'<td>{delta["joint_support"]:+.3f}</td><td>{delta["one_side"]:+.3f}</td>'
                    f'<td>{len(item["common_readings"])}</td></tr>')
        for reading in item["common_readings"]:
            support = "有" if reading["supported"] else "无"
            points.append(f'<tr><td>{seed} {arm} / {now:g} s</td><td>{reading["t_s"]:g}</td><td>{support}</td>'
                          f'<td>{reading["predicted_dbm"]:.2f}</td><td>{reading["measured_dbm"]:.2f}</td>'
                          f'<td>{reading["error_db"]:+.2f}</td><td>{reading["observation_std_db"]:.2f}</td></tr>')
    head = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>P3 · 分岔证据复核</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 16px;color:#203537}a{color:#176563}div{overflow-x:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{text-align:left;white-space:nowrap;padding:7px;border-bottom:1px solid #ccd7d7}</style>
<h1>P3 · 分岔证据复核</h1><p><a href="../index.html">总入口</a> · <a href="results.json">完整数据</a> · <a href="../passive_pricing_P3_integral_audit_20261123/index.html">积分复核</a> · <a href="../passive_pricing_P3_20261123/replay_43401.html">原 43401 回放</a> · <a href="../passive_pricing_P3_20261123/replay_43411.html">原 43411 回放</a></p>
<p>同一冻结 GP、同一合法候选。差值为 Risk 首选减 Mean 首选，均用风险定价，负值有利于 Risk。基础项是航时差加上按未知代理 1 计的暴露差；其余仅加有支持节点相对代理的价差。不是实际飞行 T+Q。</p>
<div><table><thead><tr><th>场景 / 原组</th><th>复核 (s)</th><th>分岔 (s)</th><th>分岔后双方支持 (s)</th><th>总代价差</th><th>基础项</th><th>双方支持贡献</th><th>仅一方支持贡献</th><th>分岔前新点</th></tr></thead><tbody>'''
    detail = ('</tbody></table></div><h2>旧预测与共同段实测</h2>'
              '<p>只有原日志中位置及时间均落在共同段的正常点测参加打分。未用这些读数重新拟合；它们不直接验证分岔后的暴露。</p>'
              '<div><table><thead><tr><th>场景 / 原组 / 预测签发</th><th>实测 (s)</th><th>支持</th>'
              '<th>旧预测 (dBm)</th><th>观察 (dBm)</th><th>观察减预测 (dB)</th><th>模型观测标准差 (dB)</th>'
              '</tr></thead><tbody>')
    return head + "".join(rows) + detail + "".join(points) + "</tbody></table></div></html>"


def run():
    data = analyze()
    OUT.mkdir(exist_ok=False)
    (OUT / "results.json").write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
                                     encoding="utf-8")
    (OUT / "index.html").write_text(_page(data), encoding="utf-8")
    print(json.dumps(data["summary"], indent=2))


if __name__ == "__main__":
    run()
