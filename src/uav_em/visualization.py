"""Matplotlib结果图和离线回放；左侧真值只为评价展示，绝不回流给策略。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .config import (AircraftConfig, BeliefConfig, MapConfig, ProjectConfig, ElectromagneticConfig,
                     SensorConfig, SimulationConfig)
from .environment import ElectromagneticWorld


def config_from_result(result: dict) -> ProjectConfig:
    if result.get("schema_version") != 2:
        raise ValueError("expected interference schema v2; old communication-quality logs must be rerun")
    value = result["evaluation_only_config"]
    return ProjectConfig(value["seed"], AircraftConfig(**value["aircraft"]),
                         SimulationConfig(**value["simulation"]), MapConfig(**value["map"]),
                         ElectromagneticConfig(**value["electromagnetic"]), BeliefConfig(**value["belief"]),
                         SensorConfig(**value.get("sensor", {})))


def load_result(path: str | Path) -> dict:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if result.get("project") != "uav_em" or result.get("schema_version") != 2 or not result.get("frames"):
        raise ValueError("expected uav_em interference schema v2; old quality/weather logs are unsupported")
    return result


def interference_region_masks(truth, estimate, supported, reference_dbm: float) -> dict:
    """显示参考等值区域；重合部分仅供事后对比，不代表直接扫描或安全判定。"""
    truth = np.asarray(truth)
    estimate = np.asarray(estimate)
    supported = np.asarray(supported, dtype=bool)
    if truth.ndim != 2 or truth.shape != estimate.shape or truth.shape != supported.shape:
        raise ValueError("region maps must have matching 2d shapes")
    actual = truth >= reference_dbm
    inferred = supported & (estimate >= reference_dbm)
    return {"truth": actual, "inferred": inferred, "overlap": actual & inferred}


def strong_measurement_positions(result: dict, geometry, now: float,
                                 max_age_seconds: float, reference_dbm: float) -> list:
    """只用含噪点测值标记已测强干扰，不读取当前位置的环境真值。"""
    latest = {}
    for event in result["events"]:
        age = now - event["time_seconds"]
        if event["event"] == "observation" and -1e-6 <= age <= max_age_seconds + 1e-6:
            node = geometry.node(event["position_m"])
            if node not in latest or latest[node]["time_seconds"] < event["time_seconds"]:
                latest[node] = event
    return [event["position_m"] for event in latest.values()
            if event["interference_dbm"] >= reference_dbm]


def animation_timeline(result: dict, frame_stride: int, playback_speed: float,
                       decision_pause_seconds: float) -> tuple[list[int], list[int]]:
    """按仿真秒数分配GIF毫秒时长，抽帧不丢决策，额外停顿仅用于阅读。"""
    if type(frame_stride) is not int or frame_stride < 1:
        raise ValueError("frame_stride must be a positive integer")
    if isinstance(playback_speed, bool) or not math.isfinite(playback_speed) or playback_speed <= 0:
        raise ValueError("playback_speed must be finite and positive")
    if (isinstance(decision_pause_seconds, bool) or not math.isfinite(decision_pause_seconds)
            or decision_pause_seconds < 0):
        raise ValueError("decision_pause_seconds must be finite and nonnegative")
    frames = result["frames"]
    times = np.asarray([frame["time_seconds"] for frame in frames])
    indexes = set(range(0, len(frames), frame_stride))
    indexes.add(len(frames) - 1)
    decisions = {min(int(np.searchsorted(times, entry["time_seconds"] - 1e-6)), len(frames) - 1)
                 for entry in result.get("decisions", [])}
    indexes.update(decisions)
    indexes = sorted(indexes)
    durations = []
    for offset, index in enumerate(indexes):
        seconds = ((times[indexes[offset + 1]] - times[index]) / playback_speed
                   if offset + 1 < len(indexes) else 2.0)
        if index in decisions:
            seconds += decision_pause_seconds
        durations.append(max(20, int(round(seconds * 100)) * 10))
    return indexes, durations


def decision_caption(result: dict, now: float) -> tuple[str, str]:
    """展示已经发生的行动与点测，不使用未来决策或隐藏场解释当时行为。"""
    past = [entry for entry in result.get("decisions", []) if entry["time_seconds"] <= now + 1e-6]
    if not past:
        return f"飞行时间 {now:.1f} 秒｜等待首次决策", ""
    entry = past[-1]
    action = {"continue": "继续当前航路", "replan": "向终点重新规划", "probe": "前往补测点",
              "fallback_replan": "数值回退重规划", "retained_previous_route": "保留原航路"}.get(
                  entry["accepted_action"], "未接受可执行航路")
    backend = entry["backend"]
    source = "LLM" if backend == "local" else "模拟响应（非LLM）" if backend == "mock" else "数值规划"
    if backend == "budget_fallback":
        source = "模型调用预算耗尽，转数值规划"
    elif entry.get("error"):
        source = "模型响应失败，转数值回退"
    elif entry.get("planning_failure"):
        source = "候选方案未通过复核"
    elif entry.get("override"):
        source = "模型行动被执行层覆盖"
    trigger = {"initial": "首次决策", "probe_completed": "补测完成", "probe_budget_expired": "补测到期",
               "new_observation": "测量偏离原估计", "sensing_warning": "探测预警复核",
               "source_direction_mismatch": "新测向不符源假设",
               "scheduled": "周期复核"}.get(entry["trigger"], entry["trigger"])
    first = f"飞行 {now:.1f} 秒｜{entry['time_seconds']:.1f} 秒 {trigger} → {source}：{action}"
    stage = entry.get("stage_plan_after")
    explanation = ("阶段计划已暂停，不能记为模型计划成功" if stage and stage["status"] == "suspended"
                   else "阶段目标：" + stage["proposal"]["objective"] if stage else entry["decision"].get("reason", ""))
    measurements = [event for event in result.get("events", [])
                    if event["event"] == "observation" and event["time_seconds"] <= now + 1e-6]
    latest = f"最近点测 {measurements[-1]['interference_dbm']:.1f} dBm；" if measurements else ""
    if measurements and measurements[-1].get("directional_scan"):
        scan = measurements[-1]["directional_scan"]
        peaks = scan["peak_indices"]
        latest += ("测向 " + "/".join(f"{scan['azimuth_deg'][index]:.0f}°" for index in peaks) + "（距离未知）；"
                   if peaks else "测向未检出；")
    return first, latest + explanation[:60]


def save_gif(figure, update, indexes: list[int], durations: list[int], output: Path) -> None:
    """使用GIF逐帧时长，不把抽帧数误当成播放速度。"""
    from PIL import Image

    images = []
    try:
        for index in indexes:
            update(index)
            figure.canvas.draw()
            image = Image.frombytes("RGBA", figure.canvas.get_width_height(), figure.canvas.buffer_rgba().tobytes())
            images.append(image.convert("RGB"))
            image.close()
        images[0].save(str(output), save_all=True, append_images=images[1:], duration=durations,
                       loop=0, disposal=2)
    finally:
        for image in images:
            image.close()


def render(result: dict, output: str | Path, animate: bool = False, frame_stride: int = 1,
           playback_speed: float = 1.0, decision_pause_seconds: float = 2.0) -> None:
    """两图使用同一时间、轨迹、边界与原点；未知位置以灰色明确覆盖。"""
    indexes, durations = animation_timeline(result, frame_stride, playback_speed, decision_pause_seconds)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle, Wedge

    config = config_from_result(result)
    world = ElectromagneticWorld(config)
    geometry = world.geometry
    frames = result["frames"]
    extent = (-config.map.resolution_m / 2, geometry.high[0] + config.map.resolution_m / 2,
              -config.map.resolution_m / 2, geometry.high[1] + config.map.resolution_m / 2)
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    figure.subplots_adjust(bottom=0.11, top=0.77, right=0.88, wspace=0.13)
    first = frames[0]
    electromagnetic = config.electromagnetic
    upper = max([electromagnetic.interference_reference_dbm + 20]
                + [world.truth(item["center_m"], item["start_seconds"]) for item in electromagnetic.fields])
    truth_image = axes[0].imshow(world.truth(geometry.grid(), first["time_seconds"]), origin="lower",
                                 extent=extent, vmin=electromagnetic.background_dbm, vmax=upper, cmap="YlOrRd")
    belief_image = axes[1].imshow(first["estimate_dbm"], origin="lower", extent=extent,
                                  vmin=electromagnetic.background_dbm, vmax=upper, cmap="YlOrRd")
    unknown = axes[1].imshow(np.zeros((*np.shape(first["supported"]), 4)), origin="lower", extent=extent)
    trace_lines, route_lines, markers = [], [], []
    for axis, title in zip(axes, ("Full interference truth (observer only)",
                                 "UAV estimate (gray tint = unknown/stale)")):
        axis.set_title(title)
        axis.set_xlabel("x (m)")
        axis.set_aspect("equal")
        axis.set_xlim(-10, geometry.high[0] + 10)
        axis.set_ylim(-10, geometry.high[1] + 10)
        for box in config.map.obstacles_m:
            axis.add_patch(Rectangle(box[:2], box[2] - box[0], box[3] - box[1], color="#303030", zorder=4))
        axis.plot(*config.map.start_m, "o", color="navy", markersize=5)
        axis.plot(*config.map.goal_m, "*", color="blue", markersize=12)
        trace_lines.append(axis.plot([], [], "-", color="royalblue", linewidth=2, label="executed")[0])
        route_lines.append(axis.plot([], [], "--", color="cyan", linewidth=1.5, label="planned")[0])
        markers.append(axis.plot([], [], "o", color="black", markersize=6)[0])
    axes[0].set_ylabel("y (m)")
    teal = "#007f8b"
    truth_handles, _ = axes[0].get_legend_handles_labels()
    truth_handles.extend([
        Line2D([], [], color="black", linestyle="--", label="reference boundary (not no-fly)"),
        Line2D([], [], marker="x", color="black", linestyle="none", label="actual source S1, S2..."),
        Patch(facecolor="none", edgecolor=teal, hatch="////", label="truth / inferred-region overlap")])
    measurement_markers = axes[1].plot([], [], ".", color="black", markersize=3,
                                      label="point measurements (valid age)", zorder=5)[0]
    strong_markers = [axis.plot([], [], "o", color="#00e5df", markeredgecolor="black",
                               markersize=4, label="measured strong interference", zorder=6)[0]
                      for axis in axes]
    truth_handles.append(strong_markers[0])
    if any(frame.get("directional_observations") for frame in frames):
        truth_handles.append(Patch(facecolor="darkorange", alpha=0.25,
                                   label="same received bearing as right (not range)"))
    axes[0].legend(handles=truth_handles, loc="lower left", fontsize=7)
    estimate_handles, _ = axes[1].get_legend_handles_labels()
    estimate_handles.append(Patch(facecolor="none", edgecolor=teal, hatch="////",
                                  label="inferred interference region (not scanned)"))
    estimate_handles.append(Line2D([], [], marker="D", color="gold", markeredgecolor="black",
                                   linestyle="none", label="active probe target"))
    if any(frame.get("directional_observations") for frame in frames):
        estimate_handles.append(Patch(facecolor="darkorange", alpha=0.25,
                                      label="receive bearing only (ray length is not range)"))
    if any(frame.get("source_hypothesis") is not None for frame in frames):
        estimate_handles.append(Line2D([], [], marker="^", color="purple", linestyle="--",
                                       label="H: source hypothesis / predicted boundary"))
        estimate_handles.append(Patch(facecolor="lightgray", alpha=0.5,
                                      label="tinted unknown: model-only prediction"))
    axes[1].legend(handles=estimate_handles, loc="lower left", fontsize=7)
    color_axis = figure.add_axes([0.90, 0.13, 0.015, 0.61])
    figure.colorbar(truth_image, cax=color_axis, label="Interference power (dBm; higher = stronger)")
    heading = figure.suptitle("")
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    chinese_font = next((name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei")
                         if name in available_fonts), None)
    caption = figure.text(0.05, 0.87, "", fontsize=10, va="top",
                          fontfamily=chinese_font or "DejaVu Sans", linespacing=1.6)
    probe_markers = [axis.plot([], [], "D", color="gold", markeredgecolor="black", markersize=7, zorder=7)[0]
                     for axis in axes]
    region_patches = []
    field_artists = []
    grid = geometry.grid()

    def update(index):
        frame = frames[index]
        now = frame["time_seconds"]
        truth = world.truth(grid, now)
        truth_image.set_data(truth)
        belief_image.set_data(frame["estimate_dbm"])
        support = np.asarray(frame["supported"])
        overlay = np.full((*support.shape, 4), 0.65)
        overlay[:, :, 3] = np.where(support, 0, 0.35 if frame.get("source_prediction_dbm") is not None else 0.96)
        unknown.set_data(overlay)
        for artist in field_artists:
            artist.remove()
        field_artists.clear()
        scans = frame.get("directional_observations", [])
        if scans:
            scan = scans[-1]
            origin = np.asarray(scan["position_m"])
            display_length = 0.22 * min(geometry.high)
            for angle, _ in scan["peaks_deg_dbm"]:
                half_width = scan["beamwidth_deg"] / 2
                endpoint = origin + display_length * np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
                for axis in axes:
                    wedge = Wedge(origin, display_length, angle - half_width, angle + half_width,
                                  facecolor="darkorange", alpha=0.18, zorder=4)
                    axis.add_patch(wedge)
                    field_artists.append(wedge)
                    field_artists.append(axis.plot([origin[0], endpoint[0]], [origin[1], endpoint[1]],
                                                    "--", color="darkorange", zorder=5)[0])
        reference = electromagnetic.interference_reference_dbm
        source_hypothesis = frame.get("source_hypothesis")
        if source_hypothesis is not None and frame.get("source_prediction_dbm") is not None:
            prediction = np.asarray(frame["source_prediction_dbm"])
            if prediction.min() < reference < prediction.max():
                field_artists.append(axes[1].contour(grid[:, :, 0], grid[:, :, 1], prediction,
                                                    levels=[reference], colors="purple",
                                                    linestyles="--", linewidths=1.5, zorder=3))
            for source_index, source in enumerate(source_hypothesis["sources"], start=1):
                center = (np.asarray(source["center_m"]) + np.asarray(source["velocity_mps"])
                          * (now - source_hypothesis["reference_seconds"]))
                field_artists.append(axes[1].plot(*center, "^", color="purple", markersize=7, zorder=8)[0])
                field_artists.append(axes[1].annotate(f"H{source_index}", center, xytext=(5, 5),
                                                     textcoords="offset points", color="purple", zorder=8))
        masks = interference_region_masks(truth, frame["estimate_dbm"], support, reference)
        if truth.min() < reference < truth.max():
            field_artists.append(axes[0].contour(grid[:, :, 0], grid[:, :, 1], truth,
                                                levels=[reference], colors="black",
                                                linestyles="--", linewidths=1.5, zorder=3))
        for axis, name in zip(axes, ("overlap", "inferred")):
            if masks[name].any():
                hatching = axis.contourf(grid[:, :, 0], grid[:, :, 1], masks[name].astype(float),
                                          levels=[0.5, 1.5], colors="none", hatches=["////"], zorder=3)
                hatching.set_edgecolor(teal)
                hatching.set_linewidth(0)
                field_artists.append(hatching)
        for source in world.source_positions_for_evaluation(now):
            source_position = source["position_m"]
            field_artists.append(axes[0].plot(*source_position, "x", color="black",
                                               markersize=7, markeredgewidth=2, zorder=5)[0])
            field_artists.append(axes[0].annotate(source["id"], source_position, xytext=(6, 5),
                                                 textcoords="offset points", fontsize=9, zorder=5))
        strong = np.asarray(strong_measurement_positions(
            result, geometry, now, config.belief.max_age_seconds, reference))
        for marker in strong_markers:
            if strong.size:
                marker.set_data(strong[:, 0], strong[:, 1])
            else:
                marker.set_data([], [])
        measured = np.asarray(frame["observed_positions_m"])
        if measured.size:
            measurement_markers.set_data(measured[:, 0], measured[:, 1])
        else:
            measurement_markers.set_data([], [])
        positions = np.asarray([item["position_m"] for item in frames[:index + 1]])
        route = np.asarray(frame["route_m"])
        for trace, planned, marker in zip(trace_lines, route_lines, markers):
            trace.set_data(positions[:, 0], positions[:, 1])
            planned.set_data(route[:, 0], route[:, 1]) if route.size else planned.set_data([], [])
            marker.set_data([frame["position_m"][0]], [frame["position_m"][1]])
        for patch in region_patches:
            patch.remove()
        region_patches.clear()
        past = [item for item in result["decisions"] if item["time_seconds"] <= now + 1e-9]
        latest = past[-1] if past else None
        if latest:
            for region in latest["decision"]["regions"]:
                low, high = np.asarray(region["low_m"]), np.asarray(region["high_m"])
                patch = Rectangle(low, *(high - low), fill=False, edgecolor="magenta", linewidth=2)
                axes[1].add_patch(patch)
                region_patches.append(patch)
        heading.set_text(f"UAV electromagnetic exploration | t={now:.1f}s | advisor={result['advisor']}")
        for marker in probe_markers:
            target = frame.get("active_probe")
            marker.set_data([target["position_m"][0]] if target else [], [target["position_m"][1]] if target else [])
        if chinese_font:
            caption.set_text("\n".join(decision_caption(result, now)))
        else:
            caption.set_text(f"t={now:.1f}s | latest action={latest['accepted_action'] if latest else 'none'} | "
                             f"backend={latest['backend'] if latest else 'none'}\n"
                             "Gold diamond: active probe target. Detailed decisions: decision_log.txt")
        return []

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if animate:
        save_gif(figure, update, indexes, durations, output)
    else:
        update(len(frames) - 1)
        figure.savefig(output, dpi=130)
    plt.close(figure)
