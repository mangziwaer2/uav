"""可追溯的任务上下文；只接收估计地图，不接收环境真值对象。"""

from __future__ import annotations

import numpy as np

from .belief import MapSnapshot
from .config import ProjectConfig
from .planning import path_length


def route_summary(route, snapshot: MapSnapshot, config: ProjectConfig) -> dict:
    """沿航路取样，仅对有观测支持的位置统计强干扰比例，不输出安全概率。"""
    points = []
    for start, end in zip(route, route[1:]):
        count = max(2, int(np.ceil(np.linalg.norm(np.asarray(end) - start) / config.map.resolution_m)) + 1)
        points.extend(np.linspace(start, end, count).tolist())
    if not points:
        return {"available": bool(route), "length_m": 0.0, "supported_fraction": 0.0,
                "supported_above_reference_fraction": None, "max_supported_estimate_dbm": None}
    values = [snapshot.at(point) for point in points]
    observed = [value[0] for value in values if value[2]]
    predicted = None
    if snapshot.source_prediction_dbm is not None:
        nodes = [snapshot.geometry.node(point) for point in points]
        predicted = [snapshot.source_prediction_dbm[row, column] for column, row in nodes]
    return {"available": True, "length_m": round(path_length(route), 2),
            "supported_fraction": round(float(np.mean([value[2] for value in values])), 3),
            "supported_above_reference_fraction": round(float(np.mean(
                np.asarray(observed) > config.electromagnetic.interference_reference_dbm)), 3) if observed else None,
            "max_supported_estimate_dbm": round(max(observed), 2) if observed else None,
            "model_predicted_max_dbm": round(float(max(predicted)), 2) if predicted is not None else None,
            "model_prediction_is_not_observation": True}


def corridor_id_at(position, config: ProjectConfig) -> str:
    """按摘要使用的南／中／北栅格分区标记位置，不推断干扰源边界。"""
    row = int(np.rint(float(position[1]) / config.map.resolution_m))
    return "south" if row < int(config.map.rows / 3) else "middle" if row < int(2 * config.map.rows / 3) else "north"


def probe_candidates(position, goal, snapshot: MapSnapshot, config: ProjectConfig,
                     visited: set[str]) -> list[dict]:
    """几何可直达的候选测量位置，排序仅为未知程度启发式，不是真实信息增益。"""
    position, goal = np.asarray(position), np.asarray(goal)
    delta = goal - position
    distance = float(np.linalg.norm(delta))
    if distance < config.map.resolution_m:
        return []
    forward = delta / distance
    side = np.array([-forward[1], forward[0]])
    choices = []
    for lateral, advance in ((80.0, 40.0), (-80.0, 40.0), (0.0, 100.0),
                             (100.0, 80.0), (-100.0, 80.0)):
        raw = position + forward * min(advance, distance) + side * lateral
        node = snapshot.geometry.node(raw)
        target = snapshot.geometry.position(node)
        identifier = f"probe_{node[0]}_{node[1]}"
        duration = float(np.linalg.norm(target - position) / config.aircraft.speed_mps)
        if (identifier in visited or duration < 1.0 or duration > config.simulation.probe_budget_seconds
                or not snapshot.geometry.edge_free(position, target)):
            continue
        estimate, error, support = snapshot.at(target)
        choices.append({"id": identifier, "position_m": target.tolist(),
                        "direct_flight_seconds": round(duration, 2),
                        "estimate_dbm": round(estimate, 2) if support else None,
                        "uncertainty_db": round(error, 2),
                        "supported": support,
                        "corridor_id": corridor_id_at(target, config),
                        "goal_distance_reduction_m": round(distance - float(np.linalg.norm(goal - target)), 2),
                        "approach_route": route_summary([position.tolist(), target.tolist()], snapshot, config),
                        "ranking_is_heuristic": True})
    choices.sort(key=lambda item: (item["supported"], -item["uncertainty_db"], item["direct_flight_seconds"]))
    return choices[:3]


def build_context(position, goal, route, snapshot: MapSnapshot, config: ProjectConfig,
                  remaining_energy_wh: float, candidates: list[dict], history: list[dict],
                  active_probe: dict | None) -> dict:
    """固定预算的电磁环境摘要，只提供可见估计，不暴露源与未来脚本。"""
    geometry = snapshot.geometry
    current_interference, current_error, current_support = snapshot.at(position)
    grid_samples = []
    # 稀疏摘要控制输入规模，可能漏掉局部峰值；它不是训练后的注意力／地图编码器。
    for row in np.unique(np.linspace(0, config.map.rows - 1, min(5, config.map.rows)).astype(int)):
        for column in np.unique(np.linspace(0, config.map.columns - 1, min(7, config.map.columns)).astype(int)):
            grid_samples.append([float(column * config.map.resolution_m), float(row * config.map.resolution_m),
                                 round(float(snapshot.estimate_dbm[row, column]), 1)
                                 if snapshot.supported[row, column] else None,
                                 round(float(snapshot.uncertainty_db[row, column]), 1),
                                 bool(snapshot.supported[row, column])])
            if snapshot.source_prediction_dbm is not None:
                grid_samples[-1].append(round(float(snapshot.source_prediction_dbm[row, column]), 1))
    corridors = []
    for label, low_fraction, high_fraction in (("south", 0, 1 / 3), ("middle", 1 / 3, 2 / 3),
                                               ("north", 2 / 3, 1)):
        low = int(low_fraction * config.map.rows)
        high = max(low + 1, int(high_fraction * config.map.rows))
        mask = np.array([[geometry.free(geometry.position((column, row)))
                          for column in range(config.map.columns)] for row in range(low, high)])
        interference = snapshot.estimate_dbm[low:high][mask]
        support = snapshot.supported[low:high][mask]
        observed = interference[support]
        corridors.append({"id": label, "y_range_m": [float(low * config.map.resolution_m),
                                                    float(min(high * config.map.resolution_m, geometry.high[1]))],
                          "supported_fraction": round(float(np.mean(support)), 3) if support.size else 0.0,
                          "supported_high_interference_fraction": round(float(np.mean(
                              observed > config.electromagnetic.interference_reference_dbm)), 3)
                          if observed.size else None,
                          "maximum_supported_estimate_dbm": round(float(np.max(observed)), 2) if observed.size else None,
                          "mean_supported_age_seconds": round(float(np.mean(
                              snapshot.age_seconds[low:high][mask][support])), 2) if observed.size else None,
                          "meaning": "estimated interference on supported cells only; unknown is not safe"})
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1)))) if route else []
    route_points = []
    if route:
        for index in np.unique(np.linspace(0, len(route) - 1, min(8, len(route))).astype(int)):
            # 到达时间由当前恒速折线估计，不代表在这些未来时刻预测过干扰功率。
            route_points.append({"position_m": route[index], "estimated_arrival_seconds": round(
                snapshot.time_seconds + float(cumulative[index]) / config.aircraft.speed_mps, 2)})
    return {
        "time_seconds": round(snapshot.time_seconds, 3),
        "vehicle": {"position_m": list(position), "goal_m": list(goal),
                    "altitude_m": config.aircraft.altitude_m,
                    "speed_mps": config.aircraft.speed_mps, "remaining_energy_wh": round(remaining_energy_wh, 3),
                    "reserve_energy_wh": config.aircraft.reserve_energy_wh},
        "map": {"bounds_m": [[0.0, 0.0], geometry.high.tolist()],
                "obstacles_m": config.map.obstacles_m, "clearance_m": config.aircraft.clearance_m},
        "electromagnetic_estimate": {"observation_count": snapshot.observation_count,
                           "current_grid_estimate_dbm": round(current_interference, 2) if current_support else None,
                           "current_grid_error_scale_db": round(current_error, 2),
                           "current_grid_supported": current_support,
                           "measurement_is_at_vehicle_only": True,
                           "uncertainty_is_calibrated": False,
                           "quantity": "aggregate_interference_power_dbm",
                           "higher_value_means_stronger_interference": True,
                           "interference_reference_dbm": config.electromagnetic.interference_reference_dbm,
                           "supported_fraction": round(float(snapshot.supported.mean()), 3),
                           "sample_columns": ["x_m", "y_m", "estimate_dbm", "uncertainty_db", "supported"]
                           + (["source_prediction_dbm"] if snapshot.source_prediction_dbm is not None else []),
                           "coarse_samples": grid_samples, "corridor_summaries": corridors},
        "route": {**route_summary(route, snapshot, config), "points": route_points},
        "directional_sensing": {
            "available": config.sensor.mode == "power_bearing",
            "observation_kind": ("360_degree_receive_beam_power_at_vehicle"
                                 if config.sensor.mode == "power_bearing" else "not_equipped"),
            "model_scope": "direct_path_dominated_surrogate; reflections_and_array_IQ_not_simulated",
            "calibration_correlation_seconds": config.sensor.calibration_correlation_seconds,
            "bearing_error_std_deg": config.sensor.bearing_error_deg,
            "minimum_peak_separation_deg": config.sensor.beamwidth_deg,
            "azimuth_convention": "east=0_deg; counterclockwise; world_frame",
            "recent_scans": list(snapshot.directional_observations),
            "measured_range_available": False, "remote_field_cells_measured": False,
            "meaning": "angular peaks are not source count or coordinates; no detection is not proof of safety; "
                       "triangulation requires multiple informative positions; powers are received at the vehicle; "
                       "nearby scans have correlated bearing errors; beamwidth is not a confidence interval"},
        "available_planners": {
            "astar": {"representation": "2d_grid", "incremental": False, "sampling_regions": False},
            "rrt_star": {"representation": "continuous_2d", "incremental": False, "sampling_regions": True}},
        "profiles": {"balanced": "length and estimated interference/uncertainty",
                     "avoid_interference": "stronger penalty for estimated interference above reference",
                     "explore": "lower uncertainty penalty, not an information-gain guarantee"},
        "probe_candidates": candidates, "active_probe": active_probe,
        "recent_feedback": history[-3:],
        "constraints": {"geometry_and_energy_are_hard": True, "interference_is_soft_cost": True,
                        "platform_failure_model": "not_implemented", "no_online_code_execution": True},
    }
