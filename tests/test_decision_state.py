from copy import deepcopy

import pytest

from uav_em.decision_state import execution_feedback, update_stage_plan


@pytest.fixture
def entry():
    return {
        "time_seconds": 5.0, "backend": "test_model", "error": None, "planning_failure": None,
        "override": None, "accepted_action": "replan", "accepted_planner": "rrt_star",
        "proposed_decision": {"action": "replan", "stage_plan": {
            "objective": "比较北侧走廊", "focus": "north", "assumption": "边界未知",
            "revision_condition": "补测或求解失败后复核"}},
        "execution_checkpoint": {"flight_m": 10.0, "remaining_energy_wh": 90.0},
        "solver_feedback": [{"purpose": "replan", "planner": "rrt_star", "reason": "budget_exhausted"}],
        "stage_plan_update": "created",
    }


def test_stage_plan_is_created_confirmed_and_revised_without_mutating_memory(entry):
    state, update = update_stage_plan(None, entry, 1)
    assert update == "created" and state["id"] == "stage_1"
    original = deepcopy(state)
    entry["time_seconds"] = 10
    confirmed, update = update_stage_plan(state, entry, 2)
    assert update == "confirmed" and confirmed["id"] == state["id"]
    assert state == original and confirmed["last_confirmed_seconds"] == 10
    entry["proposed_decision"]["stage_plan"]["focus"] = "south"
    revised, update = update_stage_plan(confirmed, entry, 3)
    assert update == "revised" and revised["id"] == "stage_3"
    assert state["proposal"]["focus"] == "north"


@pytest.mark.parametrize("change", [
    {"backend": "numeric"}, {"backend": "budget_fallback"}, {"error": "invalid JSON"},
    {"planning_failure": "no_path"}, {"override": "route_required"}, {"accepted_action": "fallback_replan"},
])
def test_fallback_override_and_failure_never_activate_a_proposed_plan(entry, change):
    current, _ = update_stage_plan(None, entry, 1)
    entry.update(change)
    assert update_stage_plan(None, entry, 2) == (None, "untracked")
    state, update = update_stage_plan(current, entry, 2)
    assert update == "suspended" and state["status"] == "suspended"
    assert current["status"] == "active"


def test_continue_can_retain_plan_but_unplanned_new_strategy_suspends_it(entry):
    current, _ = update_stage_plan(None, entry, 1)
    entry["proposed_decision"] = {"action": "continue", "stage_plan": None}
    entry["accepted_action"] = "continue"
    retained, update = update_stage_plan(current, entry, 2)
    assert update == "retained_without_new_proposal" and retained == current
    entry["proposed_decision"]["action"] = entry["accepted_action"] = "replan"
    suspended, update = update_stage_plan(current, entry, 3)
    assert update == "suspended" and suspended["suspension_reason"] == "missing_plan_on_new_strategy"


def test_feedback_uses_measurements_and_actual_execution_not_unobserved_exposure(entry):
    observations = [{"interference_dbm": -95, "innovation_db": 1},
                    {"interference_dbm": -70, "innovation_db": 25}]
    event = {"event": "probe_completed", "target_id": "probe_1"}
    feedback = execution_feedback(entry, observations, 10, 85, 89, -80, event)
    assert feedback["interval"] == {"elapsed_seconds": 5, "flown_m": 75, "energy_used_wh": 1}
    assert feedback["point_measurements"]["maximum_dbm"] == -70
    assert feedback["point_measurements"]["above_reference_count"] == 1
    assert feedback["previous_execution"]["actual_planner"] == "rrt_star"
    assert feedback["probe_event"] == event
    assert "high_interference_exposure_seconds" not in feedback


def test_initial_feedback_does_not_invent_previous_execution():
    feedback = execution_feedback(None, [], 0, 0, 100, -80, None)
    assert feedback["previous_execution"] is None and feedback["interval"] is None
    assert feedback["point_measurements"]["maximum_dbm"] is None
