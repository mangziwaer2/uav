from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from uav_em.advisor import parse_decision
from uav_em.belief import ElectromagneticBelief
from uav_em.config import ProjectConfig
from uav_em.context import build_context
from uav_em.decision_log import render_decision_log
from uav_em.environment import Geometry, Measurement
from uav_em.planning import PlanningProblem
from uav_em.simulation import simulate
from uav_em.source_model import SourceTracker, parse_source_hypothesis, predict_power


def source(**changes):
    return {"center_m": [180, 100], "sigma_m": 40, "peak_dbm": -65,
            "velocity_mps": [0, 0], **changes}


def hypothesis(sources=None, reference=0):
    return parse_source_hypothesis({"reference_seconds": reference,
                                    "sources": [source()] if sources is None else sources}, [800, 600])


def tracker(window=35):
    return SourceTracker([800, 600], -100, window, 8, 1)


def observe_exact(model, candidate, points, times):
    for point, timestamp in zip(points, times):
        model.observe(Measurement(tuple(point), timestamp, predict_power(candidate, point, timestamp, -100)))


def test_linear_power_sum_and_timestamped_motion():
    candidate = hypothesis([source(), source()])
    expected = 10 * np.log10(10 ** -10 + 2 * 10 ** -6.5)
    assert predict_power(candidate, [180, 100], 0, -100) == pytest.approx(expected)
    moving = hypothesis([source(velocity_mps=[2, 0])])
    predictions = predict_power(moving, [[180, 100], [200, 100]], [0, 10], -100)
    np.testing.assert_allclose(predictions, predict_power(hypothesis(), [180, 100], 0, -100))


@pytest.mark.parametrize("changes", [
    {"center_m": [True, 100]}, {"center_m": [-1, 100]}, {"center_m": [801, 100]},
    {"sigma_m": 0}, {"sigma_m": 1001}, {"peak_dbm": float("nan")}, {"peak_dbm": 1},
    {"velocity_mps": [21, 0]}, {"velocity_mps": [False, 0]}, {"code": "pass"},
    {"start_seconds": 24},
])
def test_rejects_invalid_or_executable_source_fields(changes):
    with pytest.raises(ValueError):
        hypothesis([source(**changes)])


def test_source_counts_and_invalid_reference_checked_before_prediction(monkeypatch):
    assert len(hypothesis([]).sources) == 0
    assert len(hypothesis([source()] * 3).sources) == 3
    with pytest.raises(ValueError):
        hypothesis([source()] * 4)
    model = tracker()
    model.observe(Measurement((180, 100), 0, -65))
    monkeypatch.setattr("uav_em.source_model.predict_power", lambda *args: pytest.fail("invalid time evaluated"))
    assert model.consider(hypothesis(reference=1e308), 0)["status"] == "rejected_reference_time"


def test_pre_update_prediction_residual_and_bad_candidate_preserves_incumbent():
    model, candidate = tracker(), hypothesis()
    observe_exact(model, candidate, [[180, 100]], [0])
    assert model.consider(candidate, 0)["status"] == "accepted"
    check = model.observe(Measurement((180, 100), 1, predict_power(candidate, [180, 100], 1, -100) + 10))
    assert check["source_residual_db"] == pytest.approx(10)
    assert check["source_revision"] == 1
    assert model.prediction_checks[-1]["source_residual_db"] == pytest.approx(10)
    assert model.consider(hypothesis([source(peak_dbm=-30)], reference=1), 1)["status"] == "rejected_bad_fit"
    assert model.hypothesis == candidate


def test_source_count_penalty_and_nonidentifiability():
    model, candidate = tracker(), hypothesis()
    observe_exact(model, candidate, [[140, 100], [180, 100], [220, 100]], [0, 1, 2])
    extra = hypothesis([source(), source(center_m=[700, 500], peak_dbm=-150)])
    assert model.score(extra, 2)["objective_db2"] > model.score(candidate, 2)["objective_db2"]
    above = hypothesis([source(center_m=[180, 160])])
    below = hypothesis([source(center_m=[180, 40])])
    np.testing.assert_allclose(predict_power(above, [[140, 100], [220, 100]], [0, 2], -100),
                               predict_power(below, [[140, 100], [220, 100]], [0, 2], -100))


def test_repeated_point_retains_time_information_for_moving_hypothesis():
    model = tracker()
    moving = hypothesis([source(sigma_m=20, velocity_mps=[4, 0])])
    observe_exact(model, moving, [[180, 100]] * 4, [0, 10, 20, 30])
    assert len(model.recent(30)) == 4
    assert model.score(moving, 30)["rmse_db"] == pytest.approx(0)
    assert model.score(hypothesis([source(sigma_m=20)]), 30)["rmse_db"] > 8


def test_expiration_bad_fit_and_renewal():
    model, candidate = tracker(), hypothesis()
    observe_exact(model, candidate, [[180, 100]], [0])
    model.consider(candidate, 0)
    assert model.active_hypothesis(0) == candidate
    model.observe(Measurement((180, 100), 1, -20))
    assert model.status(1) == "suspended_bad_fit"
    assert model.active_hypothesis(1) is None
    assert model.status(36) == "expired"
    renewed = hypothesis(reference=37)
    observe_exact(model, renewed, [[180, 100]], [37])
    assert model.consider(renewed, 37)["status"] == "accepted"


def test_empty_sources_can_remove_obsolete_model():
    model, candidate = tracker(window=2), hypothesis()
    observe_exact(model, candidate, [[180, 100]], [0])
    model.consider(candidate, 0)
    model.observe(Measurement((180, 100), 2.1, -100))
    assert model.consider(hypothesis([], reference=2.1), 2.1)["status"] == "accepted"
    assert not model.hypothesis.sources


def test_prediction_changes_cost_but_not_observation_support_or_unknown_error():
    config = ProjectConfig()
    config = replace(config, map=replace(config.map, obstacles_m=()))
    belief = ElectromagneticBelief(config, Geometry(config.map))
    candidate = hypothesis([source(center_m=[400, 300], sigma_m=200)])
    value = predict_power(candidate, [40, 300], 0, -100)
    belief.update(Measurement((40, 300), 0, value))
    before = belief.snapshot(0)
    assert belief.source_tracker.consider(candidate, 0)["status"] == "accepted"
    after = belief.snapshot(0)
    np.testing.assert_array_equal(before.supported, after.supported)
    np.testing.assert_allclose(after.uncertainty_db[~after.supported], config.belief.unknown_error_db)
    assert not after.at([400, 300])[2]
    assert after.at([400, 300])[0] > -70
    context = build_context([40, 300], [760, 300], [], after, config, 100, [], [], None)
    samples = context["electromagnetic_estimate"]["coarse_samples"]
    assert all(sample[2] is None and sample[5] is not None for sample in samples if not sample[4])
    assert (PlanningProblem(after, -80, "balanced").edge_cost([300, 300], [500, 300])
            > PlanningProblem(before, -80, "balanced").edge_cost([300, 300], [500, 300]))
    other = replace(config, electromagnetic=replace(config.electromagnetic, fields=()))
    second = ElectromagneticBelief(other, Geometry(other.map))
    second.update(Measurement((40, 300), 0, value))
    second.source_tracker.consider(candidate, 0)
    np.testing.assert_array_equal(after.estimate_dbm, second.snapshot(0).estimate_dbm)


def test_moving_prediction_does_not_freeze_old_peak_in_supported_cells():
    config = ProjectConfig()
    config = replace(config, map=replace(config.map, obstacles_m=()))
    belief = ElectromagneticBelief(config, Geometry(config.map))
    moving = hypothesis([source(sigma_m=20, velocity_mps=[4, 0])])
    value = predict_power(moving, [180, 100], 0, -100)
    belief.update(Measurement((180, 100), 0, value))
    belief.source_tracker.consider(moving, 0)
    current = belief.snapshot(10)
    assert current.at([180, 100])[2]
    assert current.at([180, 100])[0] < value - 8
    assert current.at([220, 100])[0] == pytest.approx(value)


def test_optional_source_decision_and_unknown_code_field_rejected():
    decision = {"action": "replan", "planner": "astar", "profile": "balanced", "reason": "test"}
    assert parse_decision(json.dumps(decision), [800, 600], set()).source_hypothesis is None
    decision["source_hypothesis"] = json.loads(json.dumps(asdict(hypothesis())))
    assert parse_decision(json.dumps(decision), [800, 600], set()).source_hypothesis == hypothesis()
    decision["source_hypothesis"]["sources"][0]["code"] = "print(1)"
    with pytest.raises(ValueError):
        parse_decision(json.dumps(decision), [800, 600], set())


class RefiningAdvisor:
    def __init__(self, malformed_final=False):
        self.contexts = []
        self.malformed_final = malformed_final

    def decide(self, context):
        self.contexts.append(context)
        round_number = context["source_refinement"]["round"]
        if round_number == 3 and self.malformed_final:
            return "{"
        decision = {"action": "replan", "planner": "astar", "profile": "avoid_interference",
                    "reason": "test response based only on visible measurement"}
        if round_number <= 2:
            measurement = context["latest_observation"]
            proposed = [] if round_number == 1 else [source(
                center_m=measurement["position_m"], peak_dbm=measurement["interference_dbm"])]
            decision["source_hypothesis"] = {"reference_seconds": context["time_seconds"], "sources": proposed}
        return json.dumps(decision)


def refinement_episode(budget=3, malformed=False):
    config = ProjectConfig()
    config = replace(config, sensor=replace(config.sensor, mode="power_only"),
                     map=replace(config.map, columns=15, rows=11, resolution_m=20,
                                 start_m=(20, 100), goal_m=(260, 100), obstacles_m=()),
                     simulation=replace(config.simulation, max_seconds=2, max_advisor_calls=budget),
                     electromagnetic=replace(config.electromagnetic, measurement_noise_db=0, fields=(
                         {"center_m": [140, 100], "sigma_m": 80, "peak_dbm": -65,
                          "velocity_mps": [0, 0], "start_seconds": 0, "end_seconds": 200},)))
    advisor = RefiningAdvisor(malformed)
    return simulate(config, advisor, advisor_name="mock", initial_planner="astar"), advisor


def test_refinement_freezes_time_data_and_records_feedback_and_prediction():
    result, advisor = refinement_episode()
    assert result["metrics"]["advisor_calls"] == 3
    assert result["metrics"]["source_hypotheses_proposed"] == 2
    assert result["metrics"]["source_hypotheses_accepted"] == 1
    assert [item["time_seconds"] for item in advisor.contexts] == [0, 0, 0]
    assert [item["source_inference"]["measurement_count"] for item in advisor.contexts] == [1, 1, 1]
    assert [item["remaining_budgets"]["advisor_calls"] for item in advisor.contexts] == [3, 2, 1]
    trace = result["decisions"][0]["source_refinement_trace"]
    assert trace[0]["source_evaluation"]["status"] == "rejected_bad_fit"
    assert trace[1]["source_evaluation"]["status"] == "accepted"
    assert advisor.contexts[2]["source_refinement"]["previous_evaluation"]["status"] == "accepted"
    assert any(event.get("source_residual_db") is not None for event in result["events"])
    json.dumps(result, allow_nan=False)
    assert "源假设数值验证" in render_decision_log(result)
    assert "evaluation_only_config" not in json.dumps(advisor.contexts)
    assert '"fields"' not in json.dumps(advisor.contexts)


def test_refinement_obeys_total_budget():
    result, advisor = refinement_episode(budget=2)
    assert len(advisor.contexts) == result["metrics"]["advisor_calls"] == 2
    assert result["decisions"][0]["source_model_after"]["revision"] == 1


def test_bad_final_refinement_falls_back_without_erasing_accepted_model():
    result, advisor = refinement_episode(malformed=True)
    assert len(advisor.contexts) == 3
    assert result["metrics"]["advisor_failures"] == 1
    assert result["metrics"]["fallbacks"] == 1
    assert result["decisions"][0]["source_model_after"]["revision"] == 1


@pytest.mark.parametrize("section,field,value", [
    ("simulation", "source_refinement_rounds", 0), ("simulation", "source_refinement_rounds", True),
    ("belief", "source_max_fit_rmse_db", 0), ("belief", "source_max_fit_rmse_db", float("inf")),
    ("belief", "source_count_penalty_db2", -1), ("belief", "source_count_penalty_db2", True),
])
def test_source_configuration_validation(section, field, value):
    config = ProjectConfig()
    config = replace(config, **{section: replace(getattr(config, section), **{field: value})})
    with pytest.raises(ValueError):
        config.validate()


def test_model_is_not_used_before_its_acceptance_time():
    model, candidate = tracker(), hypothesis(reference=10)
    observe_exact(model, candidate, [[180, 100]], [10])
    assert model.consider(candidate, 10)["status"] == "accepted"
    assert model.active_hypothesis(9) is None
    assert model.observe(Measurement((140, 100), 9, -80))["source_prediction_dbm"] is None


def test_identical_source_proposal_stops_refinement_early():
    class RepeatingAdvisor(RefiningAdvisor):
        def decide(self, context):
            self.contexts.append(context)
            return json.dumps({"action": "replan", "planner": "astar", "profile": "balanced",
                               "reason": "repeated source", "source_hypothesis": {
                                   "reference_seconds": context["time_seconds"], "sources": []}})

    config = ProjectConfig()
    config = replace(config,
                     map=replace(config.map, columns=15, rows=11, resolution_m=20,
                                 start_m=(20, 100), goal_m=(260, 100), obstacles_m=()),
                     simulation=replace(config.simulation, max_seconds=1, max_advisor_calls=6),
                     electromagnetic=replace(config.electromagnetic, fields=(), measurement_noise_db=0))
    advisor = RepeatingAdvisor()
    result = simulate(config, advisor, advisor_name="mock", initial_planner="astar")
    assert len(advisor.contexts) == result["metrics"]["advisor_calls"] == 2


def test_hypothesis_markers_use_logged_model_not_hidden_sources(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from pathlib import Path
    from uav_em.visualization import render

    result, _ = refinement_episode()
    captured = []

    def capture(figure, path, **kwargs):
        left, right = figure.axes[:2]
        assert any(text.get_text() == "S1" for text in left.texts)
        assert not any(text.get_text().startswith("S") for text in right.texts)
        marker = next(text for text in right.texts if text.get_text() == "H1")
        assert tuple(marker.xy) == (20, 100)
        assert marker.get_zorder() > right.get_legend().get_zorder()
        assert any(line.get_marker() == "^" for line in right.lines)
        captured.append(path)

    monkeypatch.setattr(Figure, "savefig", capture)
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    render(result, "unused.png")
    assert len(captured) == 1
