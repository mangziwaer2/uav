"""仅利用观测与实际执行记录维护阶段计划和反馈，不接收隐藏环境。"""

from __future__ import annotations

from copy import deepcopy


def execution_feedback(previous, observations, clock, flight_m, energy_wh, reference_dbm, probe_event):
    """区分实际飞行反馈与提议；功率范围来自点测量，不是完整真值暴露。"""
    values = [item["interference_dbm"] for item in observations]
    feedback = {
        "previous_execution": None,
        "interval": None,
        "point_measurements": {
            "count": len(values), "minimum_dbm": min(values) if values else None,
            "maximum_dbm": max(values) if values else None, "latest_dbm": values[-1] if values else None,
            "above_reference_count": sum(value > reference_dbm for value in values),
            "maximum_innovation_db": max((item["innovation_db"] for item in observations), default=None),
        },
        "probe_event": probe_event,
        "meaning": "point observations and accepted execution only; no truth exposure or future field",
    }
    if previous is not None:
        checkpoint = previous["execution_checkpoint"]
        feedback["interval"] = {
            "elapsed_seconds": clock - previous["time_seconds"],
            "flown_m": flight_m - checkpoint["flight_m"],
            "energy_used_wh": checkpoint["remaining_energy_wh"] - energy_wh,
        }
        feedback["previous_execution"] = {
            "requested_action": previous["proposed_decision"]["action"],
            "accepted_action": previous["accepted_action"], "actual_planner": previous["accepted_planner"],
            "backend": previous["backend"], "override": previous["override"],
            "model_error": previous["error"], "planning_failure": previous["planning_failure"],
            "solver_results": previous["solver_feedback"],
            "stage_plan_update": previous["stage_plan_update"],
        }
    return feedback


def update_stage_plan(current, entry, sequence):
    """只有原提议实际被接受才激活计划；回退或覆盖不会被记成计划成功。"""
    proposed = entry["proposed_decision"].get("stage_plan")
    matched = (entry["backend"] not in ("numeric", "budget_fallback")
               and entry["error"] is None and entry["planning_failure"] is None
               and entry["override"] is None
               and entry["accepted_action"] == entry["proposed_decision"]["action"])
    if matched and proposed is not None:
        if current and current["status"] == "active" and current["proposal"] == proposed:
            state = deepcopy(current)
            state["last_confirmed_seconds"] = entry["time_seconds"]
            return state, "confirmed"
        return {
            "id": f"stage_{sequence}", "status": "active", "source": entry["backend"],
            "created_seconds": entry["time_seconds"], "last_confirmed_seconds": entry["time_seconds"],
            "proposal": deepcopy(proposed),
        }, "created" if current is None else "revised"
    if current is None:
        return None, "untracked"
    state = deepcopy(current)
    if matched and entry["accepted_action"] == "continue" and current["status"] == "active":
        return state, "retained_without_new_proposal"
    state["status"] = "suspended"
    state["suspension_reason"] = "missing_plan_on_new_strategy" if matched else "execution_mismatch"
    return state, "suspended"
