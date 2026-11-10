from copy import deepcopy
import json
from pathlib import Path
import sys
from uuid import uuid4

import pytest

from uav_em.decision_log import render_decision_log


@pytest.fixture
def episode():
    return {
        "evaluation_only_config": {"secret_source": "HIDDEN_TRUTH_SENTINEL"},
        "decisions": [{
            "time_seconds": 0.0, "trigger": "initial", "backend": "local", "advisor_call": 1,
            "context": {"unknown_dbm": None, "latest_observation": {"interference_dbm": -75.0}},
            "raw_response": '{"action":"continue","reason":"保留路线"}',
            "decision": {"action": "replan", "reason": "缺少有效航路"},
            "accepted_action": "replan", "accepted_planner": "rrt_star",
            "override": "route_required", "error": None, "planning_failure": None,
            "advisor_seconds": 0.2, "accepted_route_balanced_cost": 100.0,
            "model_runtime": {"model_input_text": "<user>实际模型输入</user>", "input_tokens": 12},
        }],
        "plans": [{"time_seconds": 0.0, "purpose": "replan", "planner": "rrt_star", "length_m": 90}],
        "metrics": {"success": True, "high_interference_exposure_seconds": 2},
    }


def test_log_preserves_input_output_override_and_solver_result_without_hidden_truth(episode):
    original = deepcopy(episode)
    text = render_decision_log(episode)
    assert "实际模型输入" in text
    assert episode["decisions"][0]["raw_response"] in text
    assert '"unknown_dbm": null' in text
    assert "route_required" in text and "接受行动：replan" in text
    assert '"length_m": 90' in text
    assert "HIDDEN_TRUTH_SENTINEL" not in text
    assert episode == original


@pytest.mark.parametrize("backend", ["numeric", "mock", "budget_fallback", "local"])
def test_missing_raw_response_does_not_claim_model_success(episode, backend):
    entry = episode["decisions"][0]
    entry.update(backend=backend, raw_response=None, error="RuntimeError: 推理失败", model_runtime={})
    text = render_decision_log(episode)
    assert "未取得原始响应" in text
    assert "RuntimeError: 推理失败" in text
    if backend == "local":
        assert "实际提示词未记录" in text


def test_invalid_response_and_planning_fallback_remain_visible(episode):
    entry = episode["decisions"][0]
    entry.update(raw_response="```json\n无效输出\n```", error="ValueError: invalid JSON",
                 planning_failure="no_path", accepted_action="fallback_replan")
    text = render_decision_log(episode)
    assert "```json\n无效输出\n```" in text
    assert "ValueError: invalid JSON" in text
    assert "规划失败：no_path" in text
    assert "接受行动：fallback_replan" in text


def test_log_distinguishes_proposed_stage_plan_and_suspended_execution(episode):
    entry = episode["decisions"][0]
    entry["proposed_decision"] = {"action": "probe", "stage_plan": {"objective": "探测北侧"}}
    entry["stage_plan_after"] = {"status": "suspended"}
    entry["stage_plan_update"] = "suspended"
    text = render_decision_log(episode)
    assert "探测北侧" in text and '"status": "suspended"' in text
    assert "active不表示目标已实现" in text


def test_budgeted_thinking_log_distinguishes_generated_text_control_and_action(episode):
    episode['decisions'][0]['model_runtime'] = {
        'response_is_action_codebook_mapping': True, 'selection_mode': 'budgeted_reasoning_logits',
        'raw_model_response': 'unfinished observation analysis', 'final_action_letter': 'C',
        'action_readout_input_text': 'unfinished observation analysis</think>\n\n',
        'thinking_closed_by_controller': True, 'inserted_control_token_ids': [99, 3],
    }
    text = render_decision_log(episode)
    assert '模型生成原文；思考可能未完成' in text
    assert '由控制器关闭思考】True' in text
    assert '插入的控制token，非模型生成' in text
    assert '【最终动作字母】\nC\n' in text
    assert '不是模型自由生成的JSON' in text
    assert '原始动作字母；logits模式' not in text


def test_cli_writes_readable_log_beside_episode(monkeypatch):
    import run_experiment

    temp_root = Path(__file__).resolve().parent.parent / "temp"
    root = temp_root / f"decision_log_test_{uuid4().hex}"
    monkeypatch.setattr(sys, "argv", [
        "run_experiment.py", "--advisor", "none", "--planner", "astar", "--max-calls", "0",
        "--output-dir", str(root), "--no-plot",
    ])
    try:
        run_experiment.main()
        result = json.loads((root / "episode.json").read_text(encoding="utf-8"))
        text = (root / "decision_log.txt").read_text(encoding="utf-8")
        assert text == render_decision_log(result)
        assert "后端：numeric" in text
        assert result["metrics"]["advisor_calls"] == 0
    finally:
        assert root.resolve().parent == temp_root.resolve()
        if root.exists():
            for artifact in root.iterdir():
                artifact.unlink()
            root.rmdir()
