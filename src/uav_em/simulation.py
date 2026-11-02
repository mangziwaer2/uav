"""飞行—接收功率与测向—地图更新—直接模型咨询—规划—验证闭环。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import math
import time
from typing import Callable

import numpy as np

from .advisor import Advisor, Decision, parse_decision
from .belief import ElectromagneticBelief
from .config import ProjectConfig
from .context import build_context, probe_candidates
from .decision_state import execution_feedback, update_stage_plan
from .environment import ElectromagneticWorld
from .observation_monitor import ObservationMonitor
from .planning import PlanningProblem, path_length, plan


def advance_route(route: list, distance_m: float) -> tuple[list, list]:
    """返回剩余折线和本次实际经过的折线，保留拐点供连续几何复核。"""
    if not route or distance_m < 0:
        raise ValueError("advance requires a route and nonnegative distance")
    remaining = [list(point) for point in route]
    traversed = [remaining[0]]
    while len(remaining) > 1 and distance_m > 1e-9:
        start, end = np.asarray(remaining[0]), np.asarray(remaining[1])
        length = float(np.linalg.norm(end - start))
        if length <= distance_m + 1e-9:
            distance_m -= length
            traversed.append(end.tolist())
            remaining.pop(0)
        else:
            point = (start + (end - start) * distance_m / length).tolist()
            traversed.append(point)
            remaining[0] = point
            break
    return remaining, traversed


def simulate(config: ProjectConfig, advisor: Advisor | None = None,
             initial_planner: str = "rrt_star", advisor_name: str = "none",
             debug_hook: Callable[[str, dict], None] | None = None) -> dict:
    """模型不可改硬约束；可选调试钩子仅收到可见状态的深拷贝，返回值不参与决策。"""
    world = ElectromagneticWorld(config)
    geometry = world.geometry
    belief = ElectromagneticBelief(config, geometry)
    observation_monitor = ObservationMonitor(config)
    settings, aircraft = config.simulation, config.aircraft
    position = np.array(config.map.start_m, dtype=float)
    goal = np.array(config.map.goal_m, dtype=float)
    clock, next_review, last_review = 0.0, 0.0, -math.inf
    remaining_energy = aircraft.initial_energy_wh
    route, history, frames, events, plans, decisions = [], [], [], [], [], []
    visited: set[str] = set()
    active_probe = None
    route_planner = None
    stage_plan = None
    interval_observations = []
    advisor_calls, advisor_failures, fallbacks, probes = 0, 0, 0, 0
    flight_m, high_interference_seconds, excess_integral = 0.0, 0.0, 0.0
    total_advisor_seconds = 0.0
    termination = "time_limit"

    def emit(stage, **data):
        if debug_hook is not None:
            debug_hook(stage, deepcopy({"time_seconds": clock, **data}))

    emit("initialized", position_m=position.tolist(), goal_m=goal.tolist(),
         energy_wh=remaining_energy, initial_planner=initial_planner, advisor_name=advisor_name)

    def capture(snapshot):
        """记录策略可见估计与当时航路；回放另行读取真值，不把真值放进估计。"""
        frame = {"time_seconds": round(clock, 6), "position_m": position.tolist(),
                 "route_m": route, "energy_wh": round(remaining_energy, 6),
                 "estimate_dbm": np.round(snapshot.estimate_dbm, 2).tolist(),
                 "supported": snapshot.supported.tolist(),
                  "observation_count": snapshot.observation_count,
                 "active_probe": active_probe, "stage_plan": stage_plan}
        frame["source_prediction_dbm"] = (np.round(snapshot.source_prediction_dbm, 2).tolist()
                                         if snapshot.source_prediction_dbm is not None else None)
        frame["source_hypothesis"] = snapshot.source_hypothesis
        frame["source_status"] = belief.source_tracker.status(clock)
        frame["directional_observations"] = list(snapshot.directional_observations)
        frame["observed_positions_m"] = [list(item.position_m) for item in belief.measurements.values()
                                         if 0 <= clock - item.time_seconds <= config.belief.max_age_seconds]
        if frames and abs(frames[-1]["time_seconds"] - clock) < 1e-8:
            frames[-1] = frame
        else:
            frames.append(frame)

    def solve(snapshot, target, decision, purpose):
        """对当前估计重新搜索并记账；没有持久树修复，也不读取隐藏干扰源。"""
        budget = settings.astar_expansion_budget if decision.planner == "astar" else settings.rrt_attempt_budget
        result = plan(snapshot, position, target, decision.planner, decision.profile,
                      config.electromagnetic.interference_reference_dbm, budget, config.seed + len(plans), decision.regions)
        plans.append({"time_seconds": clock, "purpose": purpose, "profile": decision.profile,
                      "regions": [asdict(region) for region in decision.regions], **asdict(result)})
        emit("plan_solved", plan=plans[-1])
        return result

    while clock <= settings.max_seconds + 1e-9:
        # 先保留更新前估计，再比较当前点测量；否则新观测会把自己的创新量抹掉。
        before = belief.snapshot(clock)
        measurement = world.observe(position, clock)
        estimated_before, _, supported_before = before.at(position)
        innovation = abs(measurement.interference_dbm - estimated_before)
        emit("observation_received", measurement=asdict(measurement), estimated_before_dbm=estimated_before,
             supported_before=supported_before, innovation_db=innovation)
        source_check = belief.update(measurement)
        warning = observation_monitor.observe(measurement)
        snapshot = belief.snapshot(clock)
        emit("belief_updated", measurement=asdict(measurement), source_check=source_check,
             supported_fraction=float(snapshot.supported.mean()), observation_count=belief.observation_count,
             source_status=belief.source_tracker.status(clock), sensing_warning=warning)
        observation_event = {"time_seconds": clock, "event": "observation", **asdict(measurement),
                             "innovation_db": innovation, "supported_before": supported_before,
                             "sensing_warning": warning, **source_check}
        events.append(observation_event)
        interval_observations.append(observation_event)
        capture(snapshot)
        if np.linalg.norm(position - goal) < 1e-6:
            termination = "goal_reached"
            break
        if clock >= settings.max_seconds - 1e-9:
            break

        force_reason = None
        probe_event = None
        if active_probe is not None:
            if np.linalg.norm(position - active_probe["position_m"]) < 1e-6:
                force_reason = "probe_completed"
            elif clock >= active_probe["deadline_seconds"] - 1e-9:
                force_reason = "probe_budget_expired"
            if force_reason:
                probe_event = {"event": force_reason, "target_id": active_probe["id"],
                               "measurement": asdict(measurement)}
                events.append({"time_seconds": clock, "event": force_reason, "target_id": active_probe["id"]})
                active_probe = None
        changed = max(innovation, abs(source_check["source_residual_db"] or 0)) >= settings.innovation_trigger_db
        bearing_check = source_check.get("source_bearing_check")
        direction_changed = (bearing_check is not None and np.sqrt(bearing_check["mean_square_deg2"])
                             > config.belief.source_max_bearing_rmse_deg)
        # 每步测量不等于每步咨询；周期、点测偏差与补测事件共同决定咨询时机。
        due = (not route or clock >= next_review - 1e-9 or force_reason is not None
               or (changed or direction_changed or warning["reasons"])
               and clock - last_review >= settings.minimum_review_seconds)
        emit("review_checked", due=bool(due), power_changed=bool(changed), direction_changed=bool(direction_changed),
             sensing_warning=warning, force_reason=force_reason, next_review_seconds=next_review,
             active_probe=active_probe)
        if due:
            trigger = force_reason or ("initial" if not plans else "source_direction_mismatch" if direction_changed
                                       else "new_observation" if changed
                                       else "sensing_warning" if warning["reasons"] else "scheduled")
            candidates = (probe_candidates(position, goal, snapshot, config, visited)
                          if probes < settings.max_probes and active_probe is None else [])
            context = build_context(position, goal, route, snapshot, config, remaining_energy,
                                    candidates, history, active_probe)
            context["preferred_planner"] = initial_planner
            context["decision_trigger"] = trigger
            context["latest_observation"] = {**asdict(measurement), "innovation_db": innovation,
                                             "supported_before": supported_before, **source_check}
            context["sensing_warning"] = warning
            context["source_inference"] = belief.source_tracker.summary(clock)
            context["source_refinement"] = {"round": 1, "max_rounds": settings.source_refinement_rounds,
                                            "previous_evaluation": None}
            context["stage_plan"] = stage_plan
            context["execution_feedback"] = execution_feedback(
                decisions[-1] if decisions else None, interval_observations, clock, flight_m, remaining_energy,
                config.electromagnetic.interference_reference_dbm, probe_event)
            context["remaining_budgets"] = {
                "advisor_calls": max(0, settings.max_advisor_calls - advisor_calls),
                "probes": max(0, settings.max_probes - probes),
                "probe_flight_seconds": settings.probe_budget_seconds,
            }
            context["available_actions"] = (["continue"] if len(route) >= 2 and force_reason is None else []) + ["replan"]
            if candidates:
                context["available_actions"].append("probe")
            emit("context_built", context=context)
            plan_start = len(plans)
            decision = Decision("replan", initial_planner, "balanced", "本地数值基线或预算回退")
            raw, error, backend, elapsed = None, None, "numeric", 0.0
            refinement_trace = []
            if advisor is not None and advisor_calls < settings.max_advisor_calls:
                backend = advisor_name
                previous_source = None
                for round_index in range(settings.source_refinement_rounds):
                    context["remaining_budgets"]["advisor_calls"] = settings.max_advisor_calls - advisor_calls
                    round_context = json.loads(json.dumps(context, allow_nan=False))
                    advisor_calls += 1
                    started = time.perf_counter()
                    source_evaluation = None
                    decision = Decision("replan", initial_planner, "balanced", "本地数值基线或预算回退")
                    raw, error = None, None
                    emit("advisor_started", advisor_call=advisor_calls, context=round_context)
                    try:
                        raw = advisor.decide(json.loads(json.dumps(round_context, allow_nan=False)))
                        decision = parse_decision(raw, geometry.high, {item["id"] for item in candidates})
                        if decision.source_hypothesis is not None:
                            source_evaluation = belief.source_tracker.consider(decision.source_hypothesis, clock)
                    except Exception as exception:
                        error = f"{type(exception).__name__}: {exception}"
                        advisor_failures += 1
                        fallbacks += 1
                    emit("advisor_returned", advisor_call=advisor_calls, raw_response=raw,
                         parsed_decision=asdict(decision) if error is None else None, error=error)
                    if source_evaluation is not None:
                        emit("source_evaluated", advisor_call=advisor_calls, evaluation=source_evaluation,
                             source_model=belief.source_tracker.summary(clock))
                    round_elapsed = time.perf_counter() - started
                    elapsed += round_elapsed
                    total_advisor_seconds += round_elapsed
                    refinement_trace.append({"advisor_call": advisor_calls, "context": round_context,
                                             "raw_response": raw, "error": error,
                                             "decision": asdict(decision), "source_evaluation": source_evaluation,
                                             "advisor_seconds": round_elapsed,
                                             "model_runtime": dict(getattr(advisor, "last_call", {}))})
                    snapshot = belief.snapshot(clock)
                    if (error or decision.source_hypothesis is None
                            or decision.source_hypothesis == previous_source
                            or advisor_calls >= settings.max_advisor_calls
                            or round_index + 1 >= settings.source_refinement_rounds):
                        break
                    previous_source = decision.source_hypothesis
                    candidates = (probe_candidates(position, goal, snapshot, config, visited)
                                  if candidates else [])
                    context.update(build_context(position, goal, route, snapshot, config, remaining_energy,
                                                 candidates, history, active_probe))
                    context["source_inference"] = belief.source_tracker.summary(clock)
                    context["source_refinement"] = {"round": round_index + 2,
                                                    "max_rounds": settings.source_refinement_rounds,
                                                    "previous_evaluation": source_evaluation}
            elif advisor is not None:
                backend = "budget_fallback"
                fallbacks += 1
            proposed_decision = asdict(decision)
            override = None
            if decision.action == "continue" and (not route or force_reason is not None):
                decision = Decision("replan", initial_planner, "balanced", "缺少有效航路或探测结束，覆盖continue")
                override = "route_required"
            accepted, failure = None, None
            if decision.action != "continue":
                target = goal
                if decision.action == "probe":
                    selected = next(item for item in candidates if item["id"] == decision.target_id)
                    target = np.asarray(selected["position_m"])
                result = solve(snapshot, target, decision, decision.action)
                candidate = result.path
                required_length = path_length(candidate) if candidate else math.inf
                if decision.action == "probe" and candidate:
                    # 补测后还需有到目标的数值航路和能量余量，不能只检查去探测点的一段。
                    tail_budget = (settings.astar_expansion_budget if decision.planner == "astar"
                                   else settings.rrt_attempt_budget)
                    tail = plan(snapshot, target, goal, decision.planner, "balanced",
                                config.electromagnetic.interference_reference_dbm, tail_budget,
                                config.seed + len(plans))
                    plans.append({"time_seconds": clock, "purpose": "probe_tail_check", "profile": "balanced",
                                  "regions": [], **asdict(tail)})
                    emit("plan_solved", plan=plans[-1])
                    required_length += path_length(tail.path) if tail.path else math.inf
                    if path_length(candidate) / aircraft.speed_mps > settings.probe_budget_seconds + 1e-9:
                        failure = "probe_flight_budget_exceeded"
                needed = required_length / aircraft.speed_mps * aircraft.flight_power_w / 3600
                needed += settings.decision_delay_seconds * aircraft.hover_power_w / 3600
                if not candidate:
                    failure = result.reason
                elif needed > remaining_energy - aircraft.reserve_energy_wh + 1e-9:
                    failure = "candidate_energy_or_tail_check_failed"
                if failure is None:
                    route = candidate
                    route_planner = result.planner
                    if decision.action == "probe":
                        probes += 1
                        visited.add(decision.target_id)
                        active_probe = {"id": decision.target_id, "position_m": target.tolist(),
                                        "deadline_seconds": clock + settings.decision_delay_seconds
                                        + settings.probe_budget_seconds}
                    else:
                        if active_probe is not None:
                            events.append({"time_seconds": clock, "event": "probe_interrupted",
                                           "target_id": active_probe["id"], "reason": "new_strategy"})
                        active_probe = None
                    accepted = decision.action
                else:
                    fallback = Decision("replan", initial_planner, "balanced", "数值求解回退")
                    replacement = solve(snapshot, goal, fallback, "planning_fallback")
                    remaining_cost = (path_length(replacement.path) / aircraft.speed_mps
                                      * aircraft.flight_power_w / 3600)
                    hover_cost = settings.decision_delay_seconds * aircraft.hover_power_w / 3600
                    if (replacement.path and remaining_cost + hover_cost
                            <= remaining_energy - aircraft.reserve_energy_wh):
                        if active_probe is not None:
                            events.append({"time_seconds": clock, "event": "probe_interrupted",
                                           "target_id": active_probe["id"], "reason": "planning_fallback"})
                        route, active_probe, accepted = replacement.path, None, "fallback_replan"
                        route_planner = replacement.planner
                        fallbacks += 1
                    elif route and math.isfinite(PlanningProblem(
                            snapshot, config.electromagnetic.interference_reference_dbm, "balanced").route_cost(route)):
                        retained_energy = path_length(route) / aircraft.speed_mps * aircraft.flight_power_w / 3600
                        if retained_energy + hover_cost <= remaining_energy - aircraft.reserve_energy_wh:
                            accepted = "retained_previous_route"
                        else:
                            termination = "no_feasible_route_within_budget"
                    else:
                        termination = "no_feasible_route_within_budget"
            else:
                accepted = "continue"
            entry = {"time_seconds": clock, "trigger": trigger, "backend": backend,
                     "advisor_call": advisor_calls if backend not in ("numeric", "budget_fallback") else None,
                     "context": context, "raw_response": raw, "error": error,
                     "proposed_decision": proposed_decision,
                     "decision": asdict(decision), "accepted_action": accepted,
                     "accepted_planner": route_planner,
                     "planning_failure": failure, "override": override, "advisor_seconds": elapsed,
                     "accepted_route_balanced_cost": (PlanningProblem(
                         snapshot, config.electromagnetic.interference_reference_dbm, "balanced").route_cost(route)
                         if route else None),
                     "model_runtime": getattr(advisor, "last_call", {}) if backend == "local" else {}}
            entry["source_refinement_trace"] = refinement_trace
            entry["source_model_after"] = belief.source_tracker.summary(clock)
            entry["execution_checkpoint"] = {"flight_m": flight_m, "remaining_energy_wh": remaining_energy}
            entry["solver_feedback"] = [{key: item.get(key) for key in (
                "purpose", "planner", "profile", "reason", "length_m", "cost", "iterations",
                "iteration_unit")} for item in plans[plan_start:]]
            stage_plan, stage_update = update_stage_plan(stage_plan, entry, len(decisions) + 1)
            entry["stage_plan_after"], entry["stage_plan_update"] = stage_plan, stage_update
            decisions.append(entry)
            emit("decision_executed", decision=entry, route_m=route, active_probe=active_probe,
                 stage_plan=stage_plan)
            history.append({"time_seconds": clock, "requested_action": proposed_decision["action"],
                            "action": accepted, "planner": route_planner,
                            "failure": failure, "model_error": error, "stage_plan_update": stage_update})
            events.append({"time_seconds": clock, "event": "decision", "index": len(decisions) - 1,
                           "action": accepted, "backend": backend})
            capture(snapshot)
            interval_observations = []
            last_review, next_review = clock, clock + settings.review_seconds
            observation_monitor.reviewed(measurement)
            if accepted is None:
                break
            delay = min(settings.decision_delay_seconds, settings.max_seconds - clock)
            hover = delay * aircraft.hover_power_w / 3600
            if remaining_energy - hover < aircraft.reserve_energy_wh:
                termination = "energy_reserve"
                break
            remaining_energy -= hover
            clock += delay

        if not route or len(route) < 2:
            if np.linalg.norm(position - goal) < 1e-6 or active_probe is not None:
                continue
            termination = "no_route"
            break
        step = min(settings.step_seconds, settings.max_seconds - clock,
                   path_length(route) / aircraft.speed_mps)
        if next_review > clock:
            step = min(step, next_review - clock)
        if active_probe is not None:
            step = min(step, max(0.0, active_probe["deadline_seconds"] - clock))
        if step <= 1e-9:
            continue
        energy = step * aircraft.flight_power_w / 3600
        if remaining_energy - energy < aircraft.reserve_energy_wh - 1e-9:
            termination = "energy_reserve"
            break
        remaining, traversed = advance_route(route, step * aircraft.speed_mps)
        if not all(geometry.edge_free(start, end) for start, end in zip(traversed, traversed[1:])):
            raise RuntimeError("execution rejected geometrically invalid route")
        offset = 0.0
        # 以下真值积分只记事后暴露，不据此撤销飞行或触发模型；也不包含决策悬停。
        for start, end in zip(traversed, traversed[1:]):
            duration = float(np.linalg.norm(np.asarray(end) - start) / aircraft.speed_mps)
            count = max(1, int(math.ceil(duration / 0.25)))
            for index in range(count):
                fraction = (index + 0.5) / count
                point = np.asarray(start) + (np.asarray(end) - start) * fraction
                interference = world.truth(point, clock + offset + duration * fraction)
                high_interference_seconds += duration / count * (
                    interference > config.electromagnetic.interference_reference_dbm)
                excess_integral += max(0, interference - config.electromagnetic.interference_reference_dbm) * duration / count
            offset += duration
        clock += step
        remaining_energy -= energy
        flight_m += path_length(traversed)
        route, position = remaining, np.asarray(remaining[0])
        events.append({"time_seconds": clock, "event": "flight", "distance_m": path_length(traversed)})
        emit("flight_executed", position_m=position.tolist(), traversed_m=traversed,
             remaining_route_m=route, energy_wh=remaining_energy, flight_m=flight_m)

    if not frames or abs(frames[-1]["time_seconds"] - clock) > 1e-6:
        capture(belief.snapshot(clock))
    episode = {
        "schema_version": 2, "project": "uav_em", "advisor": advisor_name,
        "evaluation_only_config": config.to_dict(),
        "assumptions": {"fixed_altitude_2d": True, "interference_is_soft_cost": True,
                        "field_quantity": "aggregate_interference_power_dbm",
                        "service_station_model": False, "platform_effects_calibrated": False,
                        "outage_duration_enforced": False, "truth_not_in_advisor_context": True,
                        "model_deployment": "local_research_process_not_link_gated",
                        "actual_compute_advances_simulation": False,
                        "propagation_model": "log_distance_3d_with_prism_loss_or_explicit_gaussian_baseline",
                        "sensor_model": "simultaneous_360_degree_beam_power",
                        "remote_field_scan": False, "range_measurement": False,
                        "multipath_and_diffraction": False,
                        "uncertainty_is_calibrated": False},
        "metrics": {"success": termination == "goal_reached", "termination": termination,
                    "simulation_seconds": clock, "executed_length_m": flight_m,
                    "energy_used_wh": aircraft.initial_energy_wh - remaining_energy,
                    "high_interference_exposure_seconds": high_interference_seconds,
                    "interference_excess_integral_db_seconds": excess_integral,
                    "observations": belief.observation_count,
                    "directional_scans": sum(item["event"] == "observation"
                                             and item.get("directional_scan") is not None for item in events),
                    "directional_detected_scans": sum(item["event"] == "observation"
                        and bool((item.get("directional_scan") or {}).get("peak_indices")) for item in events),
                    "sensing_warning_reviews": sum(item["trigger"] == "sensing_warning" for item in decisions),
                    "final_supported_fraction": float(belief.snapshot(clock).supported.mean()),
                    "decisions": len(decisions), "advisor_calls": advisor_calls,
                    "stage_plan_proposals": sum(item["proposed_decision"]["stage_plan"] is not None for item in decisions),
                    "stage_plan_activations": sum(item["stage_plan_update"] in ("created", "revised") for item in decisions),
                    "advisor_failures": advisor_failures, "fallbacks": fallbacks, "probes": probes,
                    "source_hypotheses_proposed": sum(call["decision"]["source_hypothesis"] is not None
                                                     for item in decisions for call in item["source_refinement_trace"]),
                    "source_hypotheses_accepted": belief.source_tracker.revision,
                    "planning_calls": len(plans),
                    "astar_node_expansions": sum(item["iterations"] for item in plans if item["planner"] == "astar"),
                    "rrt_sample_proposals": sum(item["iterations"] for item in plans if item["planner"] == "rrt_star"),
                    "planner_seconds": sum(item["elapsed_seconds"] for item in plans),
                    "advisor_seconds": total_advisor_seconds,
                    "model_load_seconds": getattr(advisor, "load_seconds", 0.0)},
        "frames": frames, "events": events, "plans": plans, "decisions": decisions,
    }
    emit("finished", metrics=episode["metrics"])
    return episode
