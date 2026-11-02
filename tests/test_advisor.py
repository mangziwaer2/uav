import json

import pytest

from uav_em.advisor import LocalModelAdvisor, LocalModelConfig, build_prompt, parse_decision


def response(**changes):
    value = {"action": "replan", "planner": "astar", "profile": "avoid_interference", "reason": "test"}
    value.update(changes)
    return json.dumps(value)


def test_valid_probe_and_region_decisions():
    decision = parse_decision(response(action="probe", target_id="p1"), [800, 600], {"p1"})
    assert decision.target_id == "p1"
    decision = parse_decision(response(planner="rrt_star", regions=[
        {"low_m": [0, 0], "high_m": [100, 100], "weight": 1}]), [800, 600], set())
    assert decision.regions[0].weight == 1


@pytest.mark.parametrize("text", [
    response(action="bounded_outage_cross"), response(planner="rrtx"), response(extra=1),
    response(action="probe", target_id="missing"), response(target_id="p1"),
    response(regions=[{"low_m": [0, 0], "high_m": [100, 100], "weight": 1}]),
    response(planner="rrt_star", regions=[{"low_m": [True, 0], "high_m": [100, 100], "weight": 1}]),
    response(planner="rrt_star", regions=[{"low_m": [0, 0], "high_m": [900, 100], "weight": 1}]),
    response(planner="rrt_star", regions=[{"low_m": [0, 0], "high_m": [100, 100], "weight": float("nan")}]),
    '{"action":"continue","action":"replan","planner":"astar","profile":"avoid_interference","reason":"x"}',
    '```json\n' + response() + '\n```', 'null',
])
def test_invalid_decisions_are_rejected_without_repair(text):
    with pytest.raises((ValueError, TypeError)):
        parse_decision(text, [800, 600], {"p1"})


@pytest.mark.parametrize("change", [
    {"max_new_tokens": True}, {"generation_seconds": 0}, {"device": "unknown"},
    {"generation_seconds": True}, {"generation_seconds": -1},
    {"generation_seconds": float("nan")}, {"generation_seconds": float("inf")},
])
def test_local_model_resource_validation(change):
    with pytest.raises(ValueError):
        LocalModelConfig("models/Qwen3-1.7B", **change)


@pytest.mark.parametrize("time_limit", [None, 12.5])
def test_generation_defaults_to_no_time_limit_and_preserves_explicit_limit(time_limit, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import Mock
    import numpy as np

    class InputTensor:
        shape = (1, 2)

        def to(self, device):
            return self

    tokenizer = Mock(return_value={"input_ids": InputTensor()})
    tokenizer.apply_chat_template.return_value = "actual input"
    tokenizer.decode.return_value = response()
    tokenizer.pad_token_id = 0
    generate = Mock(return_value=np.array([[10, 11, 12, 13]]))
    config = (LocalModelConfig("models/Qwen3-1.7B") if time_limit is None
              else LocalModelConfig("models/Qwen3-1.7B", generation_seconds=time_limit))
    advisor = LocalModelAdvisor(config)
    advisor.tokenizer = tokenizer
    advisor.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=40960), generate=generate)
    advisor.torch = SimpleNamespace(inference_mode=nullcontext)
    monkeypatch.setattr(advisor, "load", lambda: None)

    assert advisor.decide({"probe_candidates": []}) == response()
    assert config.generation_seconds == time_limit
    assert generate.call_args.kwargs["max_time"] == time_limit
    assert generate.call_args.kwargs["max_new_tokens"] == 1024
    assert advisor.last_call["output_tokens"] == 2


@pytest.mark.parametrize("arguments,time_limit", [([], None), (["--model-timeout", "12.5"], 12.5)])
def test_cli_generation_time_limit_defaults_to_unlimited(arguments, time_limit, monkeypatch):
    from pathlib import Path
    import sys
    from unittest.mock import Mock
    import run_experiment

    factory = Mock()
    monkeypatch.setattr(run_experiment, "LocalModelAdvisor", factory)
    monkeypatch.setattr(run_experiment, "simulate", Mock(return_value={"metrics": {}}))
    monkeypatch.setattr(run_experiment, "render_decision_log", Mock(return_value=""))
    monkeypatch.setattr(Path, "mkdir", Mock())
    monkeypatch.setattr(Path, "write_text", Mock())
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--advisor", "local", "--no-plot", *arguments])

    run_experiment.main()

    assert factory.call_args.args[0].generation_seconds == time_limit
    factory.return_value.load.assert_called_once()


def test_prompt_uses_current_probe_ids_and_disables_unavailable_probe():
    context = {"probe_candidates": [{"id": "probe_3_4"}]}
    assert '"target_id": "probe_3_4"' in build_prompt(context)
    assert "action=probe" in build_prompt({"probe_candidates": []})


def test_prompt_explains_interference_direction_and_primary_planner():
    prompt = build_prompt({"probe_candidates": [], "preferred_planner": "rrt_star"})
    assert "dBm" in prompt and "-60强于-90" in prompt
    assert "avoid_interference" in prompt
    assert "preferred_planner" in prompt


def test_local_model_keeps_actual_input_when_tokenization_fails(monkeypatch):
    class FailingTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "<user>" + messages[0]["content"] + "</user>"

        def __call__(self, text, **kwargs):
            raise RuntimeError("tokenization failed")

    advisor = LocalModelAdvisor(LocalModelConfig("models/Qwen3-1.7B"))
    advisor.tokenizer = FailingTokenizer()
    monkeypatch.setattr(advisor, "load", lambda: None)
    context = {"probe_candidates": []}
    with pytest.raises(RuntimeError, match="tokenization failed"):
        advisor.decide(context)
    assert advisor.last_call["model_input_text"] == "<user>" + build_prompt(context) + "</user>"
    assert advisor.last_call["status"] == "error"


def stage_plan(**changes):
    value = {"objective": "验证北侧入口", "focus": "north", "assumption": "入口尚未测量",
             "revision_condition": "补测后重新比较走廊"}
    value.update(changes)
    return value


def test_stage_plan_is_structured_and_legacy_decisions_remain_explicitly_untracked():
    decision = parse_decision(response(stage_plan=stage_plan()), [800, 600], set())
    assert decision.stage_plan.focus == "north"
    assert parse_decision(response(), [800, 600], set()).stage_plan is None


@pytest.mark.parametrize("stage", [
    None, [], {}, stage_plan(focus="hidden_source"), stage_plan(objective=""),
    stage_plan(assumption=" " * 3), stage_plan(revision_condition="x" * 161),
    stage_plan(code="return True"), stage_plan(focus=1),
])
def test_invalid_stage_plans_do_not_expand_executable_protocol(stage):
    with pytest.raises(ValueError):
        parse_decision(response(stage_plan=stage), [800, 600], set())


def test_prompt_requests_stage_plan_and_actual_execution_feedback():
    prompt = build_prompt({"probe_candidates": []})
    assert "stage_plan" in prompt and "execution_feedback" in prompt
    assert "revision_condition" in prompt
    assert "不要输出代码" in prompt
    assert "available_actions" in prompt and "补测完成或到期后不得continue" in prompt
