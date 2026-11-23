"""One read-only P3 snapshot audit; never reads evaluator data or runs a flight."""

from hashlib import sha256
import json
from pathlib import Path

from .config import AircraftConfig, BeliefConfig, MapConfig
from .environment import Geometry
from .passive_map import FixedPriorPosterior
from .passive_pricing import POLICIES
from .passive_pricing_integral import BOUNDARY_BISECTIONS, STEP_SIZES, compare_aligned_prices


ROOT = Path(__file__).resolve().parents[2]
P3 = ROOT / "outputs/passive_pricing_P3_20261123"
OUT = ROOT / "outputs/passive_pricing_P3_integral_audit_20261123"
SEEDS = (43401, 43411)
ARMS = ("mean", "risk")
REVIEWS = (0, 12, 24, 36, 48)


def _hash(path):
    return sha256(path.read_bytes()).hexdigest()


def analyze():
    frozen = json.loads((P3 / "freeze.json").read_bytes())
    if (_hash(P3 / "contract.md") != frozen["contract_sha256"]
            or any(_hash(ROOT / name) != digest for name, digest in frozen["implementation"].items())):
        raise ValueError("P3 contract or frozen implementation hash mismatch")
    aircraft, settings = AircraftConfig(), BeliefConfig()
    geometry = Geometry(MapConfig(), aircraft.clearance_m, aircraft.altitude_m, aircraft.vertical_clearance_m)
    snapshots, inputs = [], {}
    for seed in SEEDS:
        for arm in ARMS:
            path = P3 / f"public_{seed}_{arm}.json"
            raw = path.read_bytes()
            inputs[path.name] = sha256(raw).hexdigest()
            log = json.loads(raw)
            if tuple(row["t"] for row in log["decisions"]) != REVIEWS:
                raise ValueError("P3 review times differ from the audit scope")
            for row in log["decisions"]:
                now, count = row["t"], row["prefix_size"]
                prefix = log["points"][:count]
                if not prefix or prefix[-1]["time_seconds"] != now or count != sum(
                        point["time_seconds"] <= now for point in log["points"]):
                    raise ValueError("P3 point prefix mismatch")
                routes = [candidate["route"] for candidate in row["candidates"]]
                posterior = FixedPriorPosterior(prefix, geometry, now, settings)
                grids = {f"{step:g}": compare_aligned_prices(routes, posterior, now, aircraft.speed_mps, step_s=step)
                         for step in STEP_SIZES}
                snapshots.append(dict(seed=seed, arm=arm, t_s=now, prefix_size=count, candidates=len(routes),
                                      logged_first=row["selected"], grids=grids))
    coarse, medium, fine = [f"{step:g}" for step in STEP_SIZES]
    summary = {}
    for mode in POLICIES:
        all_match, fine_match, original_match, top_order_match, separable = 0, 0, 0, 0, 0
        max_drift = 0.0
        for item in snapshots:
            first, second, third = (item["grids"][label] for label in (coarse, medium, fine))
            idx = third["selected"][mode]
            fine_match += second["selected"][mode] == idx
            all_match += first["selected"][mode] == second["selected"][mode] == idx
            original_match += item["logged_first"][mode] == idx
            top_order_match += second["order"][mode] == third["order"][mode]
            drift = max(abs(left["costs"][mode] - right["costs"][mode])
                        for left, right in zip(second["rows"], third["rows"]))
            max_drift = max(max_drift, drift)
            ordered = third["order"][mode]
            gap = (None if len(ordered) < 2 else third["rows"][ordered[1]]["costs"][mode]
                   - third["rows"][idx]["costs"][mode])
            item.setdefault("finest", {})[mode] = dict(gap=gap, drift=drift)
            separable += gap is None or gap > 2 * drift
        summary[mode] = dict(all_steps_same_first=all_match, fine_steps_same_first=fine_match,
                             fine_steps_same_whole_order=top_order_match, matches_logged_first=original_match,
                             max_fine_step_score_drift=max_drift, gap_above_twice_drift_or_single=separable)
    totals = {label: sum(item["grids"][label]["queries"] for item in snapshots) for label in (coarse, medium, fine)}
    source = Path(__file__).parent
    return dict(steps_s=STEP_SIZES, boundary_bisections=BOUNDARY_BISECTIONS, source_hashes={
        path.name: _hash(path) for path in (Path(__file__), source / "passive_pricing_integral.py")},
        contract_sha256=frozen["contract_sha256"], public_log_sha256=inputs, summary=summary, snapshots=snapshots,
        prediction_queries=totals)


def _page(data):
    count = len(data["snapshots"])
    summary = []
    for mode, row in data["summary"].items():
        summary.append(f'<tr><th>{mode}</th><td>{row["all_steps_same_first"]}/{count}</td>'
                       f'<td>{row["fine_steps_same_first"]}/{count}</td><td>{row["fine_steps_same_whole_order"]}/{count}</td>'
                       f'<td>{row["matches_logged_first"]}/{count}</td><td>{row["max_fine_step_score_drift"]:.4f}</td></tr>')
    rows = []
    for item in data["snapshots"]:
        risk = [grid["selected"]["risk"] for grid in item["grids"].values()]
        aligned = " / ".join(map(str, risk))
        gap = item["finest"]["risk"]["gap"]
        margin = "单候选" if gap is None else f"{gap:.4f}"
        rows.append(f'<tr><td>{item["seed"]}</td><td>{item["arm"]}</td><td>{item["t_s"]:g}</td>'
                    f'<td>{item["candidates"]}</td><td>{item["logged_first"]["risk"]}</td>'
                    f'<td>{aligned}</td><td>{margin}</td><td>{item["finest"]["risk"]["drift"]:.4f}</td></tr>')
    head = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>P3 · 积分步长复核</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1080px;margin:32px auto;padding:0 16px;color:#203537}a{color:#176563}div{overflow-x:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{text-align:left;white-space:nowrap;padding:7px;border-bottom:1px solid #ccd7d7}</style>
<h1>P3 · 积分步长复核</h1><p><a href="../index.html">总入口</a> · <a href="results.json">完整数据</a> · <a href="../passive_pricing_P3_20261123/replay_43401.html">原 43401 回放</a> · <a href="../passive_pricing_P3_20261123/replay_43411.html">原 43411 回放</a></p>
<p>只重算冻结公开快照。保留候选、GP、10% 支持门槛与定价，改变积分。未重飞，不读取真场。本页没有新航迹。</p>
<div><table><thead><tr><th>策略</th><th>三档首选相同</th><th>两细档首选相同</th><th>两细档全排序相同</th><th>细档与 P3 首选相同</th><th>两细档最大代价差</th></tr></thead><tbody>'''
    details = ('</tbody></table></div><h2>风险首选逐快照</h2><p>候选序号从 0 开始。两细档为 0.5 和 0.25 秒；'
               '两场的起飞前缀各在 Mean 与 Risk 日志中记一次。</p>'
               '<div><table><thead><tr><th>场景</th><th>原飞行组</th><th>时刻 (s)</th><th>候选数</th>'
               '<th>原首选</th><th>1 / 0.5 / 0.25 s 首选</th><th>细档领先差</th><th>两细档最大代价差</th>'
               '</tr></thead><tbody>')
    return head + "".join(summary) + details + "".join(rows) + "</tbody></table></div></html>"


def run():
    data = analyze()
    OUT.mkdir(exist_ok=False)
    (OUT / "results.json").write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
                                     encoding="utf-8")
    (OUT / "index.html").write_text(_page(data), encoding="utf-8")
    print(json.dumps(dict(snapshots=len(data["snapshots"]), summary=data["summary"],
                          prediction_queries=data["prediction_queries"]), indent=2))


if __name__ == "__main__":
    run()
