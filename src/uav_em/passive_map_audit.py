"""Offline P1 fork diagnostics. Reads saved decisions and evaluator frames only."""

import hashlib
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .config import AircraftConfig, BeliefConfig, MapConfig
from .environment import Geometry
from .motion import advance_route
from .passive_map import SOURCES, FixedPriorPosterior, _common_time, quadrature


ROOT = Path(__file__).resolve().parents[2]
P1 = ROOT / "outputs/passive_map_P1_20261123"
TOL = 1e-7


class _ReplayData(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reading = False
        self.chunks = []

    def handle_starttag(self, tag, attrs):
        self.reading = tag == "script" and dict(attrs).get("id") == "data"

    def handle_endtag(self, tag):
        if tag == "script":
            self.reading = False

    def handle_data(self, data):
        if self.reading:
            self.chunks.append(data)


class RecordedTruth:
    """Display-only interpolation, never across an emitter switch or outside saved frames."""

    def __init__(self, grid, frames, truth):
        self.times = np.array([frame["t"] for frame in frames])
        self.active = [tuple(source["id"] for source in frame["sources"]) for frame in truth]
        self.values = np.array([frame["dbm"] for frame in truth]).reshape(-1, grid["rows"], grid["cols"])
        spacing = grid["spacing"]
        self.sample = RegularGridInterpolator(
            (self.times, np.arange(grid["rows"]) * spacing, np.arange(grid["cols"]) * spacing),
            self.values, bounds_error=False, fill_value=np.nan)

    def predict(self, xy, clocks):
        left = np.clip(np.searchsorted(self.times, clocks, side="right") - 1, 0, len(self.times) - 1)
        right = np.minimum(left + 1, len(self.times) - 1)
        same_active = np.array([self.active[first] == self.active[second] for first, second in zip(left, right)])
        exact = np.isclose(self.times[left], clocks, rtol=0, atol=TOL)
        values = self.sample(np.column_stack([clocks, xy[:, 1], xy[:, 0]]))
        return np.where(same_active | exact, values, np.nan)


def common_measurements(points, selected, issued, fork, speed):
    """Logged subsequent measurements on the unchanged common prefix, through the fork."""
    common = []
    for point in points:
        stamp = point["time_seconds"]
        if issued + TOL < stamp <= fork + TOL:
            remaining, _ = advance_route(selected, (stamp - issued) * speed)
            if np.allclose(remaining[0], point["position_m"], rtol=0, atol=TOL):
                common.append(point)
    return common


def _audit_candidate(row, model, time, fork, speed, truth_cost, truth):
    positions, times, dt = quadrature(row["route"], time, speed)
    mean, _, support = model.predict(positions, times)
    post = np.maximum(0, times + dt / 2 - np.maximum(fork, times - dt / 2))
    logged = row["coverage"]
    whole = {name: float(dt[support == code].sum()) for code, name in enumerate(SOURCES)}
    power = 10 ** ((mean + 80) / 10)
    fresh_costs = {name: float(dt @ (1 + np.where(support <= code, power, 1)))
                   for code, name in enumerate(("local", "full"))}
    for original, reconstructed in ((logged, whole), (row["costs"], fresh_costs)):
        if any(abs(value - reconstructed[name]) > TOL for name, value in original.items()):
            raise ValueError("public scores cannot be reconstructed from the frozen P1 prefix")
    tail = {name: float(post[support == code].sum()) for code, name in enumerate(SOURCES)}
    seconds = float(dt.sum())
    saved_profile = truth.predict(positions, times)
    return dict(route=row["route"], t_s=seconds, truth_q=truth_cost - seconds, truth_cost=truth_cost,
                costs=row["costs"], tail=tail,
                profile=[(float(stamp), float(value) if np.isfinite(value) else None)
                         for stamp, value in zip(times, saved_profile)])


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def analyze():
    freeze = _read(P1 / "freeze.json")
    if (_sha(P1 / "contract.md") != freeze["contract_sha256"]
            or any(_sha(Path(__file__).with_name(name)) != digest
                   for name, digest in freeze["implementation"].items())):
        raise ValueError("P1 contract or original implementation no longer matches the saved freeze")
    parser = _ReplayData()
    parser.feed((P1 / "index.html").read_text(encoding="utf-8"))
    replay = json.loads("".join(parser.chunks))
    local, full = (_read(P1 / f"public_{arm}.json") for arm in ("local", "full"))
    saved = _read(P1 / "results.json")
    if any(local[key] != full[key] for key in ("points", "frames", "decisions")):
        raise ValueError("shared-state audit requires identical recorded trajectories and decisions")
    if saved["local"]["reviews"] != saved["full"]["reviews"]:
        raise ValueError("the archived arms disagree on the shared-state oracle")
    if replay["arms"]["full"]["log"] != full:
        raise ValueError("the replay and archived public log disagree")

    grid = replay["grid"]
    truth_frames = replay["arms"]["full"]["evaluation"]["trajectory"]
    truth = RecordedTruth(grid, full["frames"], truth_frames)
    aircraft, settings = AircraftConfig(), BeliefConfig()
    geometry = Geometry(MapConfig(), aircraft.clearance_m)
    points = full["points"]
    rows = []
    for decision, result in zip(full["decisions"], saved["full"]["reviews"], strict=True):
        time = decision["t"]
        menu = decision["candidates"]
        selected, oracle = decision["selected"]["full"], result["oracle"]
        exact_costs = result["truth_rank_cost_s"]
        prefix = points[:decision["prefix_size"]]
        if (time != result["t"] or len(menu) != len(exact_costs)
                or oracle != min(range(len(exact_costs)), key=exact_costs.__getitem__)
                or len(prefix) != sum(point["time_seconds"] <= time for point in points)):
            raise ValueError("inconsistent P1 decision records")
        model = FixedPriorPosterior(prefix, geometry, time, settings)
        common = _common_time(menu[selected]["route"], menu[oracle]["route"], aircraft.speed_mps)
        fork = float(time + common)
        selected_data, oracle_data = (_audit_candidate(menu[index], model, time, fork, aircraft.speed_mps,
                                                       exact_costs[index], truth) for index in (selected, oracle))
        position = advance_route(menu[selected]["route"], common * aircraft.speed_mps)[0][0]
        samples = common_measurements(points, menu[selected]["route"], time, fork, aircraft.speed_mps)
        frame = int(np.searchsorted(truth.times, time))
        rows.append(dict(t=time, count=len(menu), selected=selected_data, oracle=oracle_data,
                         gap=exact_costs[selected] - exact_costs[oracle], fork_s=fork, fork_xy=position,
                         same_route=bool(common >= max(selected_data["t_s"], oracle_data["t_s"]) - TOL),
                         common_points=samples, truth_index=frame, n_points=len(prefix)))
    return rows, grid, full, truth, truth_frames


def _render(path, row, grid, decision, prefix, truth, frame):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    extent = [0, (grid["cols"] - 1) * grid["spacing"], 0, (grid["rows"] - 1) * grid["spacing"]]
    half_cell = grid["spacing"] / 2
    raster_extent = [extent[0] - half_cell, extent[1] + half_cell, extent[2] - half_cell, extent[3] + half_cell]
    figure = Figure(figsize=(12.5, 7), layout="constrained")
    FigureCanvasAgg(figure)
    layout = figure.add_gridspec(2, 2, height_ratios=[3, 1])
    left, right = figure.add_subplot(layout[0, 0]), figure.add_subplot(layout[0, 1])
    profile = figure.add_subplot(layout[1, :])
    image = left.imshow(truth.values[row["truth_index"]], origin="lower", extent=raster_extent, cmap="viridis",
                        vmin=-105, vmax=-65)
    shape = (grid["rows"], grid["cols"])
    right.imshow(np.asarray(decision["gp"]["dbm"]).reshape(shape), origin="lower", extent=raster_extent,
                 cmap="viridis", vmin=-105, vmax=-65)
    unsupported = np.asarray(decision["gp"]["support"]).reshape(shape) == 2
    right.contourf(np.linspace(extent[0], extent[1], shape[1]), np.linspace(extent[2], extent[3], shape[0]),
                   unsupported, levels=[0.5, 1.5], colors=[(0.36, 0.40, 0.43, 0.65)])
    for axis, title in ((left, "Saved truth (evaluation only)"), (right, "Public decision snapshot")):
        axis.set(xlim=extent[:2], ylim=extent[2:], xlabel="x (m)", ylabel="y (m)", title=title)
        for x_min, y_min, x_max, y_max in grid["obstacles"]:
            axis.fill([x_min, x_max, x_max, x_min], [y_min, y_min, y_max, y_max],
                      color="#282b32", edgecolor="#eeeeee", alpha=0.9)
        axis.scatter(*grid["goal"], marker="s", s=35, c="white", edgecolors="black", zorder=5)
    for candidate in decision["candidates"]:
        right.plot(*np.asarray(candidate["route"]).T, color="#ffffff", alpha=0.3, lw=0.7)
    for axis in (left, right):
        axis.plot(*np.asarray(row["selected"]["route"]).T, color="#65f7ff", lw=2, label="Executed choice")
    if not row["same_route"]:
        left.plot(*np.asarray(row["oracle"]["route"]).T, color="#ff8cc5", lw=2, ls="--", label="Ex-post best")
        left.scatter(*row["fork_xy"], c="white", edgecolors="black", marker="D", s=35, label="Fork", zorder=6)
    for source in frame["sources"]:
        left.scatter(*source["position_m"], marker="X", s=55, color="#ffda81", edgecolors="black", zorder=7)
    if row["common_points"] and not row["same_route"]:
        left.scatter(*np.asarray([point["position_m"] for point in row["common_points"]]).T,
                     s=18, facecolors="white", edgecolors="black", zorder=8)
    right.scatter(*np.asarray([point["position_m"] for point in prefix]).T, s=14,
                  c=[point["interference_dbm"] for point in prefix], cmap="viridis", vmin=-105, vmax=-65,
                  edgecolors="white", linewidths=0.5, zorder=8)
    left.legend(loc="upper left", fontsize=8)
    figure.colorbar(image, ax=[left, right], label="Power (dBm)", shrink=0.85)
    profile.set(xlabel="Simulation time (s)", ylabel="Power (dBm)", title="Profile interpolated from saved truth frames")
    profile.grid(alpha=0.2)
    for key, color, style in (("selected", "#087d89", "-"), ("oracle", "#b72e76", "--")):
        if key == "oracle" and row["same_route"]:
            continue
        stamps, values = zip(*row[key]["profile"])
        profile.plot(stamps, np.array(values, dtype=float), c=color, ls=style, label=key)
    if not row["same_route"]:
        profile.axvline(row["fork_s"], ls=":", c="#444444", label="fork")
    profile.legend(loc="best", fontsize=8)
    figure.savefig(path, dpi=130)


def _write_report(output, rows, grid, public, truth, truth_frames):
    for index, row in enumerate(rows):
        decision = public["decisions"][index]
        _render(output / f"fork_{index}.png", row, grid, decision, public["points"][:row["n_points"]],
                truth, truth_frames[row["truth_index"]])
    inputs = ("contract.md", "freeze.json", "index.html", "public_local.json", "public_full.json", "results.json")
    (output / "audit.json").write_text(json.dumps(
        dict(input_sha256={name: _sha(P1 / name) for name in inputs}, reviews=rows),
        ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    table, figures = [], []
    for index, row in enumerate(rows):
        selected, oracle = row["selected"], row["oracle"]
        split = "同路" if row["same_route"] else f"{row['fork_s']:.2f} s"
        samples = "—" if row["same_route"] else str(len(row["common_points"]))
        table.append(f"<tr><td><a href='#s{index}'>{row['t']:g} s</a></td><td>{row['count']}</td>"
                     f"<td>{selected['truth_cost']:.2f}</td><td>{oracle['truth_cost']:.2f}</td>"
                     f"<td>{row['gap']:.2f}</td><td>{split}</td><td>{samples}</td></tr>")
        details = []
        if not row["same_route"]:
            point = ", ".join(f"{coord:.1f}" for coord in row["fork_xy"])
            delta_t, delta_q = selected["t_s"] - oracle["t_s"], selected["truth_q"] - oracle["truth_q"]
            times = ", ".join(f"{point['time_seconds']:g}" for point in row["common_points"]) or "无"
            details.append(f"分岔点 ({point}) m · 所选减事后最优：ΔT {delta_t:+.2f} s，ΔQ {delta_q:+.2f} · "
                           f"共同段后续实际点测时刻：{times}")
            details.append("分岔后 Full 未定价：所选 {:.2f} s，事后最优 {:.2f} s。".format(
                selected["tail"]["unknown"], oracle["tail"]["unknown"]))
        figures.append(f"<details id='s{index}' {'open' if row['t'] == 12 else ''}><summary>{row['t']:g} 秒 · 合法候选 {row['count']} 条</summary>"
                       + "".join(f"<p>{escape(text)}</p>" for text in details)
                       + f"<a href='fork_{index}.png'><img src='fork_{index}.png' loading='lazy' "
                         f"alt='{row['t']:g} 秒分岔：左图真场，右图冻结公开快照，下方为功率剖面'></a></details>")
    page = """<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>P1 分岔复核</title>
<style>body{font:16px/1.6 system-ui,sans-serif;color:#203133;background:#f8faf9;max-width:1200px;margin:auto;padding:20px}
a{color:#126a69}h1{font-size:25px}table{border-collapse:collapse;width:100%;text-align:right;font-variant-numeric:tabular-nums}
td,th{padding:6px 10px;border-bottom:1px solid #cddbd9;white-space:nowrap}th:first-child,td:first-child{text-align:left}
.scroll{overflow:auto}details{margin:14px 0;border-top:1px solid #bccdcb;padding-top:6px}summary{cursor:pointer;font-weight:650}
img{display:block;width:100%;height:auto}p{margin:8px 0}</style>
<h1>P1 · 分岔复核</h1><p><a href='../index.html'>原始飞行回放</a> · <a href='../../index.html'>实验总入口</a> · <a href='audit.json'>复核数据</a></p>
<p>事后分析，只使用已有快照，不重飞或补测。所选与域内最优均按 P1 留档的同批候选真值评分比较。共同段点测按实际日志计数，不代表这些点足以区分两条路。</p>
<p>左图青色为所选，粉色虚线为事后最优，菱形是分岔点；右图仅显示当时点测、GP 和合法路线池。下方剖面用已有真场帧插值，只作定位，跨发射源切换或超出记录留空，不用于表中 T+Q。</p>
<div class='scroll'><table><thead><tr><th>快照</th><th>合法候选</th><th>所选 T+Q</th><th>域内最优 T+Q</th><th>差额</th><th>分岔时刻</th><th>分岔前新点数</th></tr></thead><tbody>"""
    (output / "index.html").write_text(page + "".join(table) + "</tbody></table></div>" + "".join(figures),
                                       encoding="utf-8")


def publish():
    output = P1 / "fork_audit"
    if output.exists():
        raise FileExistsError(output)
    data = analyze()
    with TemporaryDirectory(prefix=".fork_audit-", dir=P1) as directory:
        stage = Path(directory) / "report"
        stage.mkdir()
        _write_report(stage, *data)
        stage.rename(output)


if __name__ == "__main__":
    publish()
