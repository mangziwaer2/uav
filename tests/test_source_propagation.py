from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from uav_em.advisor import build_prompt, parse_decision
from uav_em.belief import ElectromagneticBelief
from uav_em.config import ProjectConfig, load_config
from uav_em.decision_log import render_decision_log
from uav_em.environment import ElectromagneticWorld, Geometry, Measurement
from uav_em.propagation import PropagationModel
from uav_em.sensing import DirectionalReceiver
from uav_em.simulation import simulate
from uav_em.source_model import PropagationSource, parse_source_hypothesis, predict_power


def proposed_source(**changes):
    return {"center_m": [300, 100], "eirp_dbm": -2, "height_m": 1.5, "velocity_mps": [0, 0], **changes}


def proposal(sources=None, now=0):
    return parse_source_hypothesis({"reference_seconds": now,
                                    "sources": [proposed_source()] if sources is None else sources}, [800, 600])


@pytest.fixture
def config():
    base = ProjectConfig()
    return replace(base, map=replace(base.map, start_m=(100, 100), goal_m=(700, 100), obstacles_m=()),
                   sensor=replace(base.sensor, noise_db=0, bearing_error_deg=0),
                   electromagnetic=replace(base.electromagnetic, measurement_noise_db=0, fields=(
                       {**proposed_source(), "start_seconds": 0, "end_seconds": 100},)))


def belief_for(config):
    return ElectromagneticBelief(config, Geometry(config.map))


def test_physical_protocol_and_public_forward_model_match_synthetic_environment(config):
    candidate = proposal()
    assert isinstance(candidate.sources[0], PropagationSource)
    decision = parse_decision(json.dumps({"action": "replan", "planner": "rrt_star", "profile": "balanced",
                                         "reason": "test", "source_hypothesis": asdict(candidate)}), [800, 600], set())
    assert decision.source_hypothesis == candidate
    model = PropagationModel.from_config(config)
    points = np.array([[100, 100], [200, 120], [300, 100], [500, 500]])
    np.testing.assert_allclose(predict_power(candidate, points, 0, -100, model),
                               ElectromagneticWorld(config).truth(points, 0))
    with pytest.raises(ValueError, match="explicit public"):
        predict_power(candidate, points, 0, -100)


def test_multiple_moving_sources_and_vectorized_obstacle_loss_use_each_sample_time(config):
    sources = [proposed_source(velocity_mps=[0, 2]),
               proposed_source(center_m=[500, 300], eirp_dbm=-4, height_m=5, velocity_mps=[-1, 0])]
    config = replace(config, map=replace(config.map, obstacles_m=((180, 90, 200, 200),)),
                     electromagnetic=replace(config.electromagnetic, fields=tuple(
                         {**source, "start_seconds": 0, "end_seconds": 100} for source in sources)))
    candidate, model = proposal(sources), PropagationModel.from_config(config)
    points, times = np.array([[100, 100], [120, 180], [400, 100]]), np.array([0, 3, 9])
    actual = predict_power(candidate, points, times, -100, model)
    expected = [ElectromagneticWorld(config).truth(point, now) for point, now in zip(points, times)]
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual, [predict_power(candidate, point, now, -100, model)
                                       for point, now in zip(points, times)])
    assert np.all(actual > predict_power(proposal(sources[:1]), points, times, -100, model))


@pytest.mark.parametrize("changes", [
    {"eirp_dbm": True}, {"eirp_dbm": -151}, {"eirp_dbm": 81}, {"eirp_dbm": float("nan")},
    {"height_m": -1}, {"height_m": 101}, {"height_m": False}, {"height_m": float("inf")},
    {"sigma_m": 40}, {"start_seconds": 0}, {"code": "pass"},
])
def test_physical_source_limits_and_hidden_script_fields_are_strict(changes):
    with pytest.raises(ValueError):
        proposal([proposed_source(**changes)])


def test_gaussian_and_physical_parameters_cannot_be_mixed():
    gaussian = {"center_m": [300, 100], "sigma_m": 40, "peak_dbm": -65, "velocity_mps": [0, 0]}
    with pytest.raises(ValueError, match="cannot be mixed"):
        proposal([proposed_source(), gaussian])
    assert len(proposal([]).sources) == 0


def test_power_equivalent_wrong_bearing_is_rejected_without_erasing_incumbent(config):
    belief = belief_for(config)
    belief.update(ElectromagneticWorld(config).observe([100, 100], 0))
    tracker = belief.source_tracker
    right, wrong = proposal(), proposal([proposed_source(center_m=[100, 300])])
    assert tracker.score(right, 0)["rmse_db"] == pytest.approx(0)
    assert tracker.score(wrong, 0)["rmse_db"] == pytest.approx(0)
    rejected = tracker.consider(wrong, 0)
    assert rejected["status"] == "rejected_bad_bearing"
    assert rejected["candidate_fit"]["bearing_fit"]["rmse_deg"] == pytest.approx(90)
    assert tracker.hypothesis is None
    assert tracker.consider(right, 0)["status"] == "accepted"
    assert tracker.consider(wrong, 0)["status"] == "rejected_bad_bearing"
    assert tracker.hypothesis == right


def test_power_only_mode_does_not_invent_bearing_constraints(config):
    config = replace(config, sensor=replace(config.sensor, mode="power_only"))
    belief = belief_for(config)
    belief.update(ElectromagneticWorld(config).observe([100, 100], 0))
    wrong = proposal([proposed_source(center_m=[100, 300])])
    assert belief.source_tracker.score(wrong, 0)["bearing_fit"]["scan_count"] == 0
    assert belief.source_tracker.consider(wrong, 0)["status"] == "accepted"


def test_matching_hypothesis_tolerates_default_noise_without_knowing_realized_calibration(config):
    config = replace(config, sensor=ProjectConfig().sensor,
                     electromagnetic=replace(config.electromagnetic, measurement_noise_db=0.5))
    world, belief = ElectromagneticWorld(config), belief_for(config)
    for now, point in [(0, [100, 100]), (1, [130, 120]), (3, [160, 140])]:
        belief.update(world.observe(point, now))
    evaluation = belief.source_tracker.consider(proposal(now=3), 3)
    assert evaluation["status"] == "accepted"
    assert evaluation["candidate_fit"]["rmse_db"] < 2
    bearing = evaluation["candidate_fit"]["bearing_fit"]
    assert bearing["scan_count"] > 0
    assert bearing["rmse_deg"] <= config.belief.source_max_bearing_rmse_deg
    assert "calibration_error_deg" not in json.dumps(belief.source_tracker.summary(3))


def test_moving_source_direction_and_power_use_each_observation_time(config):
    moving_source = proposed_source(velocity_mps=[0, 5])
    config = replace(config, electromagnetic=replace(config.electromagnetic, fields=(
        {**moving_source, "start_seconds": 0, "end_seconds": 100},)))
    world, belief = ElectromagneticWorld(config), belief_for(config)
    for now in [0, 5, 10]:
        belief.update(world.observe([100, 100], now))
    moving = proposal([proposed_source(center_m=[300, 150], velocity_mps=[0, 5])], now=10)
    static = proposal(now=10)
    moving_fit = belief.source_tracker.score(moving, 10)
    static_fit = belief.source_tracker.score(static, 10)
    assert moving_fit["rmse_db"] == pytest.approx(0)
    assert moving_fit["bearing_fit"]["rmse_deg"] == pytest.approx(0)
    assert static_fit["bearing_fit"]["rmse_deg"] > 0
    assert static_fit["objective_db2"] > moving_fit["objective_db2"]


def test_angular_wrap_and_blended_sources_compare_predicted_peaks_not_individual_ids(config):
    sources = [proposed_source(center_m=[300, 140]), proposed_source(center_m=[300, 60])]
    config = replace(config, electromagnetic=replace(config.electromagnetic, fields=tuple(
        {**source, "start_seconds": 0, "end_seconds": 100} for source in sources)))
    measurement = ElectromagneticWorld(config).observe([100, 100], 0)
    assert len(measurement.directional_scan.peaks()) == 1
    tracker = belief_for(config).source_tracker
    first = tracker.bearing_check(proposal(sources), measurement)
    assert first == tracker.bearing_check(proposal(list(reversed(sources))), measurement)
    assert first["mean_square_deg2"] == pytest.approx(0)
    assert first["predicted_peaks_deg"] == [0]
    scan = DirectionalReceiver(config).scan([-80], [359])
    assert tracker.bearing_check(proposal(), replace(measurement, directional_scan=scan))["mean_square_deg2"] == 0


def test_absence_of_detected_peaks_is_not_evidence_of_remote_safety(config):
    measurement = ElectromagneticWorld(config).observe([100, 100], 0)
    measurement = replace(measurement, directional_scan=replace(measurement.directional_scan, peak_indices=()))
    tracker = belief_for(config).source_tracker
    tracker.observe(measurement)
    fit = tracker.score(proposal(), 0)
    assert fit["bearing_fit"]["scan_count"] == 0
    assert fit["bearing_fit"]["rmse_deg"] is None
    assert tracker.consider(proposal(), 0)["status"] == "accepted"


def test_correlated_repeated_scans_are_averaged_by_time_blocks(config):
    tracker, world = belief_for(config).source_tracker, ElectromagneticWorld(config)
    wrong_scan = replace(world.observe([100, 100], 0).directional_scan, peak_indices=(6,))
    wrong_scan = replace(wrong_scan, power_dbm=tuple(-80.0 for _ in wrong_scan.power_dbm))
    early = Measurement((100, 100), 0, world.truth([100, 100], 0), wrong_scan)
    late = world.observe([100, 100], 10)
    tracker.observe(early)
    tracker.observe(late)
    first = tracker.score(proposal(), 10)["bearing_fit"]
    for unused in range(20):
        tracker.observe(early)
    duplicated = tracker.score(proposal(), 10)["bearing_fit"]
    assert duplicated["scan_count"] == 22
    assert first["block_count"] == duplicated["block_count"] == 2
    assert duplicated["loss"] == pytest.approx(first["loss"])
    assert duplicated["rmse_deg"] == pytest.approx(first["rmse_deg"])


def test_new_bearing_check_precedes_refit_and_can_suspend_a_power_consistent_model(config):
    tracker = belief_for(config).source_tracker
    measurement = ElectromagneticWorld(config).observe([100, 100], 0)
    tracker.observe(replace(measurement, directional_scan=None))
    wrong = proposal([proposed_source(center_m=[100, 300])])
    assert tracker.consider(wrong, 0)["status"] == "accepted"
    check = tracker.observe(replace(measurement, time_seconds=1))
    assert check["source_revision"] == 1
    assert check["source_residual_db"] == pytest.approx(0)
    assert check["source_bearing_check"]["mean_square_deg2"] == pytest.approx(90 ** 2)
    assert tracker.status(1) == "suspended_bad_bearing"
    assert tracker.active_hypothesis(1) is None
    assert tracker.consider(proposal(now=1), 1)["status"] == "accepted"
    assert tracker.prediction_checks[-1]["source_bearing_check"]["mean_square_deg2"] == pytest.approx(90 ** 2)
    assert tracker.status(37) == "expired"


def test_physical_prediction_never_expands_observation_support_or_uses_hidden_scripts(config):
    belief = belief_for(config)
    measurement = ElectromagneticWorld(config).observe([100, 100], 0)
    belief.update(measurement)
    before = belief.snapshot(0)
    assert belief.source_tracker.consider(proposal(), 0)["status"] == "accepted"
    after = belief.snapshot(0)
    np.testing.assert_array_equal(after.supported, before.supported)
    np.testing.assert_array_equal(after.uncertainty_db, before.uncertainty_db)
    assert not after.at([300, 100])[2]
    assert after.source_prediction_dbm is not None
    other = belief_for(replace(config, electromagnetic=replace(config.electromagnetic, fields=())))
    other.update(measurement)
    assert other.source_tracker.consider(proposal(), 0)["status"] == "accepted"
    np.testing.assert_array_equal(other.snapshot(0).estimate_dbm, after.estimate_dbm)
    assert other.source_tracker.summary(0) == belief.source_tracker.summary(0)
    public = json.dumps(belief.source_tracker.summary(0), allow_nan=False)
    for hidden in ('"fields"', '"start_seconds"', '"end_seconds"', '"calibration_error_deg"'):
        assert hidden not in public


class VisibleBearingFixture:
    def __init__(self):
        self.contexts = []

    def decide(self, context):
        self.contexts.append(context)
        decision = {"action": "replan", "planner": "astar", "profile": "avoid_interference",
                    "reason": "deterministic interface fixture, not an LLM"}
        if len(self.contexts) <= 2:
            observation = context["latest_observation"]
            scan = observation["directional_scan"]
            angle = scan["azimuth_deg"][scan["peak_indices"][0]] + (90 if len(self.contexts) == 1 else 0)
            direction = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
            center = (np.array(observation["position_m"]) + 200 * direction).tolist()
            decision["source_hypothesis"] = {"reference_seconds": context["time_seconds"],
                                              "sources": [proposed_source(center_m=center)]}
        return json.dumps(decision)


def test_mock_refinement_rejects_direction_error_and_accepts_revision_with_readable_log(config):
    config = replace(config, simulation=replace(config.simulation, max_seconds=2, max_advisor_calls=3))
    advisor = VisibleBearingFixture()
    result = simulate(config, advisor, advisor_name="mock", initial_planner="astar")
    trace = result["decisions"][0]["source_refinement_trace"]
    assert [item["source_evaluation"]["status"] if item["source_evaluation"] else None
            for item in trace] == ["rejected_bad_bearing", "accepted", None]
    assert [item["time_seconds"] for item in advisor.contexts] == [0, 0, 0]
    assert result["metrics"]["source_hypotheses_accepted"] == 1
    assert result["metrics"]["advisor_failures"] == 0
    assert any(event.get("source_bearing_check") for event in result["events"])
    assert "方向RMSE" in render_decision_log(result)
    assert "新观测方向残差" in render_decision_log(result)
    assert '"fields"' not in json.dumps(advisor.contexts)
    json.dumps(result, allow_nan=False)


def test_direction_mismatch_requests_review_but_obeys_minimum_interval(config, monkeypatch):
    original = ElectromagneticWorld.observe

    def corrupted_scan(world, position, now):
        observation = original(world, position, now)
        if now >= 1:
            scan = observation.directional_scan
            observation = replace(observation, directional_scan=replace(
                scan, azimuth_deg=tuple((angle + 180) % 360 for angle in scan.azimuth_deg)))
        return observation

    class FixedFixture:
        def decide(self, context):
            return json.dumps({"action": "replan", "planner": "astar", "profile": "balanced", "reason": "fixture",
                               "source_hypothesis": {"reference_seconds": context["time_seconds"],
                                                       "sources": [proposed_source()]}})

    monkeypatch.setattr(ElectromagneticWorld, "observe", corrupted_scan)
    config = replace(config, simulation=replace(config.simulation, max_seconds=5, review_seconds=30,
                                                source_refinement_rounds=1, max_advisor_calls=2))
    result = simulate(config, FixedFixture(), advisor_name="mock", initial_planner="astar")
    reviews = [item for item in result["decisions"] if item["trigger"] == "source_direction_mismatch"]
    assert reviews
    assert reviews[0]["time_seconds"] >= config.simulation.minimum_review_seconds
    assert reviews[0]["time_seconds"] < config.simulation.review_seconds
    assert reviews[0]["context"]["latest_observation"]["source_bearing_check"]


@pytest.mark.parametrize("field,value", [
    ("source_bearing_weight_db2", 0), ("source_bearing_weight_db2", float("nan")),
    ("source_max_bearing_rmse_deg", 0), ("source_max_bearing_rmse_deg", 181),
])
def test_direction_validation_configuration_is_checked(config, field, value):
    with pytest.raises(ValueError):
        replace(config, belief=replace(config.belief, **{field: value})).validate()


def test_default_config_and_prompt_explain_physical_hypothesis_and_evidence(config):
    base = ProjectConfig()
    primary = replace(base, electromagnetic=replace(base.electromagnetic,
                      propagation_mode="single_reflection", path_loss_exponent=2))
    assert json.loads(json.dumps(load_config("configs/demo.json").to_dict())) == json.loads(
        json.dumps(primary.to_dict()))
    public = belief_for(config).source_tracker.summary(0)
    assert public["recommended_source_model"] == "log_distance"
    assert public["source_fields"] == ["center_m", "eirp_dbm", "height_m", "velocity_mps"]
    prompt = build_prompt({"source_inference": public, "probe_candidates": []})
    assert "eirp_dbm不是当前位置接收功率" in prompt
    assert "rejected_bad_bearing" in prompt
    assert "证据不足可不提交" in prompt
    assert "相关扫描按时间块汇总" in prompt
