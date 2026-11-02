from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import runpy
import tempfile

import pytest

from uav_em.advisor import MockAdvisor
from uav_em.config import ProjectConfig
from uav_em.simulation import simulate


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory(prefix="interference_debug_tests_", dir=ROOT / "temp") as directory:
        path = Path(directory).resolve()
        assert path.parent == (ROOT / "temp").resolve()
        yield path


@pytest.fixture
def small_config():
    base = ProjectConfig()
    return replace(base, map=replace(base.map, columns=15, rows=11, start_m=(20, 100),
                                    goal_m=(260, 100), obstacles_m=()),
                   simulation=replace(base.simulation, max_seconds=30, review_seconds=5, max_advisor_calls=3),
                   electromagnetic=replace(base.electromagnetic, fields=(), measurement_noise_db=0))


def without_timing(value):
    if isinstance(value, dict):
        return {key: without_timing(item) for key, item in value.items() if key not in (
            "elapsed_seconds", "advisor_seconds", "planner_seconds", "model_load_seconds")}
    if isinstance(value, (tuple, list)):
        return [without_timing(item) for item in value]
    return value


def test_debug_hook_is_observation_only_and_cannot_mutate_or_steer_simulation(small_config):
    stages = []

    def mutate(stage, data):
        stages.append((stage, deepcopy(data)))
        data.clear()
        return {"action": "probe", "position_m": [-100, -100]}

    plain = simulate(small_config, MockAdvisor(), initial_planner="astar", advisor_name="mock")
    observed = simulate(small_config, MockAdvisor(), initial_planner="astar", advisor_name="mock", debug_hook=mutate)
    assert without_timing(plain) == without_timing(observed)
    assert stages[0][0] == "initialized" and stages[-1][0] == "finished"
    names = [stage for stage, data in stages]
    assert names.index("observation_received") < names.index("belief_updated") < names.index("review_checked")
    assert names.index("advisor_started") < names.index("advisor_returned") < names.index("decision_executed")
    assert "plan_solved" in names and "flight_executed" in names
    encoded = json.dumps(stages, allow_nan=False)
    for forbidden in ('"evaluation_only_config"', '"fields"', '"start_seconds"', '"end_seconds"',
                      '"calibration_error_deg"', '"truth_dbm"'):
        assert forbidden not in encoded


def test_debug_hook_exceptions_propagate_instead_of_being_counted_as_model_failures(small_config):
    def stop(stage, data):
        if stage == "advisor_returned":
            raise RuntimeError("debugger stopped")

    with pytest.raises(RuntimeError, match="debugger stopped"):
        simulate(small_config, MockAdvisor(), initial_planner="astar", debug_hook=stop)


def test_source_validation_stage_carries_actual_fit_without_adding_calls(small_config):
    class SourceFixture:
        def decide(self, context):
            return json.dumps({"action": "replan", "planner": "astar", "profile": "balanced", "reason": "fixture",
                               "source_hypothesis": {"reference_seconds": context["time_seconds"], "sources": []}})

    stages = []
    result = simulate(small_config, SourceFixture(), initial_planner="astar", advisor_name="mock",
                      debug_hook=lambda stage, data: stages.append((stage, data)))
    evaluations = [data["evaluation"] for stage, data in stages if stage == "source_evaluated"]
    logged = [entry["source_evaluation"] for decision in result["decisions"]
              for entry in decision["source_refinement_trace"] if entry["source_evaluation"] is not None]
    assert evaluations == logged
    assert len([stage for stage, data in stages if stage == "advisor_started"]) == result["metrics"]["advisor_calls"]


@pytest.fixture
def debug_module():
    return runpy.run_path(str(ROOT / "temp/debug_full_pipeline.py"))


def test_debug_entry_records_full_numeric_loop_without_loading_model(debug_module, tmp_path):
    directory = tmp_path / "new_debug"
    result = debug_module["main"](["--advisor", "none", "--planner", "astar", "--max-seconds", "2",
                                   "--quiet", "--output-dir", str(directory)])
    assert result == directory
    episode = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    assert episode["advisor"] == "none" and episode["metrics"]["advisor_calls"] == 0
    stages = [json.loads(line) for line in (directory / "debug_stages.jsonl").read_text(encoding="utf-8").splitlines()]
    assert stages[0]["stage"] == "initialized" and stages[-1]["stage"] == "finished"
    assert any(item["stage"] == "flight_executed" for item in stages)
    metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "complete" and metadata["debug_stage_count"] == len(stages)
    assert (directory / "decision_log.txt").is_file()
    assert not (directory / "calls.jsonl").exists()


def test_debug_entry_mock_logs_exact_outputs_and_refuses_existing_directory(debug_module, tmp_path):
    directory = tmp_path / "mock_debug"
    arguments = ["--advisor", "mock", "--planner", "astar", "--max-seconds", "2", "--max-calls", "1",
                 "--quiet", "--output-dir", str(directory)]
    debug_module["main"](arguments)
    call = json.loads((directory / "calls.jsonl").read_text(encoding="utf-8"))
    assert (directory / "call_0001_response.txt").read_text(encoding="utf-8") == call["raw_response"]
    assert json.loads((directory / "call_0001_context.json").read_text(encoding="utf-8")) == call["context"]
    episode = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    assert episode["decisions"][0]["source_refinement_trace"][0]["raw_response"] == call["raw_response"]
    assert not (directory / "call_0001_model_input.txt").exists()
    original = (directory / "metrics.json").read_bytes()
    with pytest.raises(FileExistsError):
        debug_module["main"](arguments)
    assert (directory / "metrics.json").read_bytes() == original


def test_recorded_advisor_preserves_generation_error_and_actual_template_input(debug_module, tmp_path):
    class FailedLocal:
        last_call = {"status": "error", "model_input_text": "actual-template-input"}

        def decide(self, context):
            raise ValueError("input too large")

    advisor = debug_module["RecordedAdvisor"](FailedLocal(), tmp_path)
    context = {"time_seconds": 0, "source_refinement": {"round": 1}, "decision_trigger": "initial"}
    with pytest.raises(ValueError, match="input too large"):
        advisor.decide(context)
    call = json.loads((tmp_path / "calls.jsonl").read_text(encoding="utf-8"))
    assert call["raw_response"] is None
    assert call["generation_error"] == "ValueError: input too large"
    assert (tmp_path / "call_0001_model_input.txt").read_text(encoding="utf-8") == "actual-template-input"


def test_stage_pause_uses_builtin_debugger_only_when_requested(debug_module, tmp_path, monkeypatch):
    paused = []
    monkeypatch.setattr("pdb.set_trace", lambda: paused.append(True))
    debugger = debug_module["PipelineDebugger"](tmp_path, pause_at=["belief_updated"], quiet=True)
    debugger("initialized", {"time_seconds": 0})
    assert not paused
    debugger("belief_updated", {"time_seconds": 0})
    assert paused == [True]
