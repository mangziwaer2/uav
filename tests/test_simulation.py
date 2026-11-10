from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from uav_em.advisor import MockAdvisor
from uav_em.config import ProjectConfig, load_config
from uav_em.simulation import advance_route, simulate


@pytest.fixture
def small_config():
    config = ProjectConfig()
    return replace(config,
                   map=replace(config.map, columns=15, rows=11, resolution_m=20,
                               start_m=(20, 100), goal_m=(260, 100), obstacles_m=()),
                   simulation=replace(config.simulation, max_seconds=50, review_seconds=5,
                                      max_advisor_calls=4, rrt_attempt_budget=120),
                   electromagnetic=replace(config.electromagnetic, fields=(), measurement_noise_db=0))


def test_numeric_episode_reaches_goal_and_is_json_serializable(small_config):
    result = simulate(small_config, initial_planner="astar")
    assert result["metrics"]["success"]
    assert result["metrics"]["advisor_calls"] == 0
    assert result["metrics"]["observations"] > 10
    assert result["metrics"]["executed_length_m"] == pytest.approx(240)
    assert result["frames"][0]["observation_count"] == 1
    assert 0 < result["metrics"]["final_supported_fraction"] < 1
    assert result["frames"][-1]["position_m"] == [260, 100]
    json.dumps(result, allow_nan=False)


def test_default_episode_uses_rrt_without_hidden_astar(small_config):
    result = simulate(small_config, MockAdvisor(), advisor_name="mock")
    assert result["metrics"]["success"]
    assert result["metrics"]["astar_node_expansions"] == 0
    assert all(item["planner"] == "rrt_star" for item in result["plans"])
    assert result["schema_version"] == 2
    assert result["assumptions"]["service_station_model"] is False
    assert "station_m" not in json.dumps(result["evaluation_only_config"])


def test_observed_change_triggers_consultation_before_regular_review(small_config):
    field = {"center_m": [130, 100], "sigma_m": 1000, "peak_dbm": -40,
             "start_seconds": 4, "end_seconds": 100, "velocity_mps": [0, 0]}
    config = replace(small_config,
                     electromagnetic=replace(small_config.electromagnetic, fields=(field,)),
                     simulation=replace(small_config.simulation, review_seconds=20))
    result = simulate(config, MockAdvisor(), initial_planner="astar", advisor_name="mock")
    change = next(item for item in result["decisions"] if item["trigger"] == "new_observation")
    assert 4 <= change["time_seconds"] < 20
    assert change["backend"] == "mock"
    observed = change["context"]["latest_observation"]
    assert observed["innovation_db"] >= config.simulation.innovation_trigger_db
    assert observed["interference_dbm"] > config.electromagnetic.interference_reference_dbm
    assert "fields" not in change["context"]["electromagnetic_estimate"]


def test_unobserved_remote_change_does_not_trigger_or_alter_decisions(small_config):
    field = {"center_m": [260, 190], "sigma_m": 3, "peak_dbm": -40,
             "start_seconds": 4, "end_seconds": 100, "velocity_mps": [0, 0]}
    changed = replace(small_config,
                      electromagnetic=replace(small_config.electromagnetic, fields=(field,)))
    baseline = simulate(small_config, initial_planner="astar")
    result = simulate(changed, initial_planner="astar")
    assert result["frames"] == baseline["frames"]
    assert [item["context"] for item in result["decisions"]] == [
        item["context"] for item in baseline["decisions"]]
    assert all(item["trigger"] != "new_observation" for item in result["decisions"])


def test_mock_probe_uses_observations_and_has_bounded_duration(small_config):
    result = simulate(small_config, MockAdvisor(), advisor_name="mock")
    assert result["metrics"]["success"]
    assert result["metrics"]["probes"] == 1
    assert any(item["event"] == "probe_completed" for item in result["events"])
    context = result["decisions"][1]["context"]
    assert context["electromagnetic_estimate"]["observation_count"] > 1
    for forbidden in ("fields", "peak_dbm", "velocity_mps", "truth", "start_seconds"):
        assert f'"{forbidden}":' not in json.dumps(context)
    assert context["source_inference"]["hypothesis"] is None


def test_stage_plan_probe_execution_and_revision_feed_back_into_next_consultation(small_config):
    class PlannedAdvisor:
        def decide(self, context):
            if context["stage_plan"] is None:
                action = "probe"
                stage = {"objective": "补测候选入口", "focus": context["probe_candidates"][0]["corridor_id"],
                         "assumption": "入口情况尚需点测", "revision_condition": "到达补测点后复核"}
            elif context["active_probe"] is not None:
                action, stage = "continue", context["stage_plan"]["proposal"]
            else:
                action = "replan"
                stage = {"objective": "根据新增测量向终点重规划", "focus": "goal",
                         "assumption": "未知航段仍需持续观测", "revision_condition": "新测量偏差增大时复核"}
            value = {"action": action, "planner": "astar", "profile": "balanced", "reason": "测试夹具",
                     "stage_plan": stage}
            if action == "probe":
                value["target_id"] = context["probe_candidates"][0]["id"]
            return json.dumps(value)

    result = simulate(small_config, PlannedAdvisor(), initial_planner="astar", advisor_name="test_planned")
    assert result["metrics"]["success"]
    initial = result["decisions"][0]
    assert initial["stage_plan_update"] == "created"
    assert initial["context"]["execution_feedback"]["previous_execution"] is None
    completed = next(item for item in result["decisions"] if item["trigger"] == "probe_completed")
    assert "continue" not in completed["context"]["available_actions"]
    feedback = completed["context"]["execution_feedback"]
    assert feedback["probe_event"]["event"] == "probe_completed"
    assert feedback["probe_event"]["measurement"] == {
        key: completed["context"]["latest_observation"][key]
        for key in ("position_m", "time_seconds", "interference_dbm", "directional_scan")}
    assert feedback["point_measurements"]["count"] > 0
    assert feedback["interval"]["flown_m"] > 0
    assert completed["stage_plan_update"] == "revised"
    assert initial["stage_plan_after"]["proposal"]["objective"] == "补测候选入口"
    assert completed["stage_plan_after"]["proposal"]["objective"] != "补测候选入口"
    budget = next(item for item in result["decisions"] if item["backend"] == "budget_fallback")
    assert budget["stage_plan_after"]["status"] == "suspended"
    assert result["metrics"]["stage_plan_activations"] >= 2
    json.dumps(result, allow_nan=False)


def test_proposed_plan_survives_in_log_but_not_execution_when_continue_is_overridden(small_config):
    class PlannedContinueAdvisor:
        def decide(self, context):
            return json.dumps({"action": "continue", "planner": "astar", "profile": "balanced", "reason": "test",
                               "stage_plan": {"objective": "保留现有路线", "focus": "goal",
                                              "assumption": "测试无路线覆盖", "revision_condition": "无路线时改计划"}})

    result = simulate(small_config, PlannedContinueAdvisor(), initial_planner="astar", advisor_name="test_override")
    initial = result["decisions"][0]
    assert initial["proposed_decision"]["action"] == "continue"
    assert "continue" not in initial["context"]["available_actions"]
    assert initial["proposed_decision"]["stage_plan"] is not None
    assert initial["decision"]["action"] == initial["accepted_action"] == "replan"
    assert initial["stage_plan_after"] is None
    following = result["decisions"][1]["context"]["execution_feedback"]["previous_execution"]
    assert following["requested_action"] == "continue" and following["accepted_action"] == "replan"
    assert following["override"] == "route_required"


def test_mutating_advisor_cannot_rewrite_stored_stage_plan(small_config):
    class MutatingStageAdvisor:
        def decide(self, context):
            if context["stage_plan"]:
                context["stage_plan"]["proposal"]["objective"] = "corrupted"
                return "invalid"
            return json.dumps({"action": "replan", "planner": "astar", "profile": "balanced", "reason": "test",
                               "stage_plan": {"objective": "original", "focus": "goal",
                                              "assumption": "待验证", "revision_condition": "定期复核"}})

    result = simulate(small_config, MutatingStageAdvisor(), initial_planner="astar", advisor_name="test_mutation")
    assert result["decisions"][0]["stage_plan_after"]["proposal"]["objective"] == "original"
    assert result["decisions"][1]["context"]["stage_plan"]["proposal"]["objective"] == "original"
    assert result["decisions"][1]["stage_plan_after"]["status"] == "suspended"


def test_failed_or_mutating_advisor_cannot_corrupt_geometry(small_config):
    class BadAdvisor:
        def decide(self, context):
            context["map"]["obstacles_m"].append([0, 0, 1000, 1000])
            context["vehicle"]["goal_m"][:] = [0, 0]
            return "not JSON"

    result = simulate(small_config, BadAdvisor(), advisor_name="test_invalid")
    assert result["metrics"]["success"]
    assert result["metrics"]["advisor_failures"] == result["metrics"]["advisor_calls"]
    assert not result["decisions"][0]["context"]["map"]["obstacles_m"]
    assert result["frames"][-1]["position_m"] == [260, 100]


def test_no_route_continue_is_overridden_and_call_budget_is_honest(small_config):
    class ContinueAdvisor:
        def decide(self, context):
            return json.dumps({"action": "continue", "planner": "astar", "profile": "balanced", "reason": "test"})

    config = replace(small_config, simulation=replace(small_config.simulation, max_advisor_calls=1))
    result = simulate(config, ContinueAdvisor(), advisor_name="test_continue")
    assert result["metrics"]["success"]
    assert result["metrics"]["advisor_calls"] == 1
    assert result["decisions"][0]["override"] == "route_required"
    assert any(item["backend"] == "budget_fallback" for item in result["decisions"])


def test_every_decision_within_budget_directly_calls_advisor(small_config):
    class CountingAdvisor:
        def __init__(self):
            self.calls = 0

        def decide(self, context):
            self.calls += 1
            return json.dumps({"action": "replan", "planner": "astar", "profile": "balanced", "reason": "test"})

    advisor = CountingAdvisor()
    result = simulate(small_config, advisor, advisor_name="test_counting")
    assert advisor.calls == min(result["metrics"]["decisions"], small_config.simulation.max_advisor_calls)
    assert all(item["advisor_call"] is not None for item in result["decisions"][:advisor.calls])


def test_truth_exposure_does_not_act_as_a_hidden_safety_shield(small_config):
    field = {"center_m": [130, 100], "sigma_m": 1000, "peak_dbm": -20,
             "start_seconds": 0, "end_seconds": 100, "velocity_mps": [0, 0]}
    config = replace(small_config, electromagnetic=replace(small_config.electromagnetic, fields=(field,)))
    result = simulate(config)
    assert result["metrics"]["success"]
    assert result["metrics"]["high_interference_exposure_seconds"] > 0
    assert result["assumptions"]["outage_duration_enforced"] is False


def test_insufficient_energy_terminates_instead_of_ignoring_reserve(small_config):
    config = replace(small_config, aircraft=replace(small_config.aircraft, initial_energy_wh=15.01))
    result = simulate(config)
    assert not result["metrics"]["success"]
    assert result["metrics"]["energy_used_wh"] <= 0.01 + 1e-9


def test_zero_delay_and_zero_call_budget_do_not_loop_forever(small_config):
    config = replace(small_config, simulation=replace(small_config.simulation, decision_delay_seconds=0,
                                                     max_advisor_calls=0))
    result = simulate(config, MockAdvisor(), advisor_name="mock")
    assert result["metrics"]["success"] and result["metrics"]["advisor_calls"] == 0
    times = [item["time_seconds"] for item in result["frames"]]
    assert all(end > start for start, end in zip(times, times[1:]))


def test_advance_preserves_segment_junctions_and_remaining_route():
    remaining, executed = advance_route([[0, 0], [10, 0], [10, 10]], 15)
    assert executed == [[0, 0], [10, 0], [10, 5]]
    assert remaining == [[10, 5], [10, 10]]


def test_seeded_episode_has_reproducible_decisions_and_trajectory(small_config):
    first = simulate(small_config, MockAdvisor(), advisor_name="mock")
    second = simulate(small_config, MockAdvisor(), advisor_name="mock")
    assert first["frames"] == second["frames"]
    assert [item["decision"] for item in first["decisions"]] == [item["decision"] for item in second["decisions"]]


def test_default_config_file_is_valid():
    config = load_config(Path(__file__).resolve().parent.parent / "configs/local_power.json")
    assert config.map.goal_m == [760, 300]


def test_integer_region_coordinates_remain_serializable_in_full_episode(small_config):
    class RegionAdvisor:
        def decide(self, context):
            return json.dumps({"action": "replan", "planner": "rrt_star", "profile": "balanced",
                               "reason": "integer region test", "regions": [
                                   {"low_m": [0, 0], "high_m": [280, 200], "weight": 1}]})

    config = replace(small_config, simulation=replace(small_config.simulation, max_advisor_calls=1))
    result = simulate(config, RegionAdvisor(), advisor_name="test_region")
    assert result["metrics"]["success"]
    assert result["metrics"]["rrt_sample_proposals"] > 0
    assert result["metrics"]["advisor_failures"] == 0
    json.dumps(result, allow_nan=False)
