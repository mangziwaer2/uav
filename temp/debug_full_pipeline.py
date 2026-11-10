"""完整仿真调试入口，直接复用正式simulate，不复制算法或生成虚假的源答案。

从main进入：读取配置 → 构造后端 → simulate → 保存日志 → 可选绘图／GIF。
在PipelineDebugger.__call__设置IDE断点，可检查stage与data；按F7进入simulate可看全部局部变量。
--pause-at observation_received --pause-at source_evaluated可在指定阶段进入pdb。
默认none后端不用模型；mock是工程夹具；local才使用已有本地LLM。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_em.advisor import LocalModelAdvisor, LocalModelConfig, MockAdvisor, build_prompt
from uav_em.config import load_config
from uav_em.decision_log import render_decision_log
from uav_em.simulation import simulate


STAGE_TITLES = {
    "initialized": "01 初始化无人机、地图与观测缓存",
    "observation_received": "02 接收本地点功率和可选含噪测向；尚未更新地图",
    "belief_updated": "03 检验旧源的新观测预测并更新地图；未测区域仍未知",
    "review_checked": "04 检查周期、观测变化、方向失配及补测结束事件",
    "context_built": "05 构建只含可见信息的决策上下文",
    "advisor_started": "06 咨询后端；local才是实际LLM调用",
    "advisor_returned": "07 原始输出、严格解析或明确回退；不修补响应",
    "source_evaluated": "08 检验候选源功率／方向一致性，不自动反演源位置",
    "plan_solved": "09 传统算子搜索；补测还会检查尾程",
    "decision_executed": "10 几何／能量复核后的接受行动与阶段计划",
    "flight_executed": "11 执行短段并扣能量，随后返回观测循环",
    "finished": "12 仿真终止；到达、失败和回退分别记录",
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


class PipelineDebugger:
    """这里是推荐断点：stage为阶段名，data为深拷贝，可展开但修改不影响真实仿真。"""

    def __init__(self, directory, pause_at=(), quiet=False):
        self.directory = Path(directory)
        self.pause_at = set(pause_at)
        self.quiet = quiet
        self.sequence = 0

    def __call__(self, stage, data):
        self.sequence += 1
        record = {"sequence": self.sequence, "stage": stage, "title": STAGE_TITLES[stage], "data": data}
        with (self.directory / "debug_stages.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        if not self.quiet:
            print(f"[{self.sequence:04d}] t={data['time_seconds']:.3f}s {STAGE_TITLES[stage]}", flush=True)
            if stage == "source_evaluated":
                print(json.dumps(data["evaluation"], ensure_ascii=False), flush=True)
            elif stage == "decision_executed":
                decision = data["decision"]
                print(f"  backend={decision['backend']} action={decision['accepted_action']} "
                      f"trigger={decision['trigger']} error={decision['error']}", flush=True)
        if "all" in self.pause_at or stage in self.pause_at:
            import pdb
            pdb.set_trace()


class RecordedAdvisor:
    """只保存输入输出并原样转交；不改提示词、不猜源、不修复JSON、不自动换模型。"""

    def __init__(self, delegate, directory):
        self.delegate = delegate
        self.directory = Path(directory)
        self.calls = 0

    @property
    def last_call(self):
        return getattr(self.delegate, "last_call", {})

    @property
    def load_seconds(self):
        return getattr(self.delegate, "load_seconds", 0.0)

    def decide(self, context):
        self.calls += 1
        prefix = self.directory / f"call_{self.calls:04d}"
        write_json(prefix.with_name(prefix.name + "_context.json"), context)
        prefix.with_name(prefix.name + "_prompt.txt").write_text(build_prompt(context), encoding="utf-8")
        print(f"CALL_START {self.calls} t={context['time_seconds']:.3f}s "
              f"round={context['source_refinement']['round']} trigger={context['decision_trigger']}", flush=True)
        raw, error = None, None
        started = time.perf_counter()
        try:
            raw = self.delegate.decide(context)
            return raw
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
            raise
        finally:
            runtime = dict(self.last_call)
            record = {"advisor_call": self.calls, "context": context, "raw_response": raw,
                      "generation_error": error, "model_runtime": runtime}
            with (self.directory / "calls.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            if raw is not None:
                prefix.with_name(prefix.name + "_response.txt").write_text(raw, encoding="utf-8")
            if runtime.get("model_input_text") is not None:
                prefix.with_name(prefix.name + "_model_input.txt").write_text(
                    runtime["model_input_text"], encoding="utf-8")
            print(f"CALL_END {self.calls} tokens={runtime.get('input_tokens')}/{runtime.get('output_tokens')} "
                  f"elapsed={time.perf_counter() - started:.2f}s error={error}", flush=True)
            print("RAW_RESPONSE " + (raw if raw is not None else "<no response>"), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/local_power.json")
    parser.add_argument("--advisor", choices=("none", "mock", "local"), default="none")
    parser.add_argument("--planner", choices=("rrt_star", "astar"), default="rrt_star")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/Qwen3-1.7B")
    parser.add_argument("--model-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--max-seconds", type=float, help="可选仿真时长上限，不是模型生成时限")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, help="必须是不存在的新目录；不会覆盖已有结果")
    parser.add_argument("--pause-at", action="append", choices=("all", *STAGE_TITLES), default=[])
    parser.add_argument("--quiet", action="store_true", help="不打印逐阶段进度；调用输入输出仍保存")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--gif", action="store_true", help="另生成1倍速、决策停顿2秒GIF")
    return parser.parse_args(argv)


def main(argv=None):
    """在此逐步执行；进入simulate后可跟踪接收→估计→咨询→规划→执行的真实循环。"""
    args = parse_args(argv)
    config = load_config(args.config.resolve())
    changes = {}
    if args.max_calls is not None:
        changes["max_advisor_calls"] = args.max_calls
    if args.max_seconds is not None:
        changes["max_seconds"] = args.max_seconds
    if changes:
        config = replace(config, simulation=replace(config.simulation, **changes))
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    config.validate()
    directory = (args.output_dir.resolve() if args.output_dir is not None else ROOT / "outputs" /
                 f"interference_debug_{args.advisor}_{datetime.now():%Y%m%d_%H%M%S_%f}")
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    public_files = [Path(__file__), args.config.resolve(), *sorted((ROOT / "src/uav_em").glob("*.py"))]
    metadata = {"status": "initialized", "started_local": datetime.now().astimezone().isoformat(),
                "command": subprocess.list2cmdline(sys.orig_argv), "python_executable": sys.executable,
                "python_version": platform.python_version(), "advisor": args.advisor,
                "initial_planner": args.planner, "seed": config.seed,
                "source_hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in public_files},
                "purpose": "完整闭环逐步调试／有预算的真实接口诊断；非同预算收益评价",
                "model_timeout_seconds": None, "response_transformed": False,
                "timing_note": "真实计算耗时独立记账，断点停顿不推进仿真时间；仿真固定计决策悬停",
                "debug_only": True}
    write_json(directory / "run_metadata.json", metadata)
    write_json(directory / "evaluation_only_config.json", config.to_dict())
    debugger = PipelineDebugger(directory, args.pause_at, args.quiet)
    print(f"OUTPUT_DIR {directory}", flush=True)
    try:
        delegate = None
        if args.advisor == "mock":
            delegate = MockAdvisor()
        elif args.advisor == "local":
            model_config = LocalModelConfig(str(args.model_path.resolve()), args.model_device)
            delegate = LocalModelAdvisor(model_config)
            metadata.update(status="loading_model", model_config=asdict(model_config))
            write_json(directory / "run_metadata.json", metadata)
            delegate.load()
            import torch
            import transformers
            metadata.update(torch_version=torch.__version__, transformers_version=transformers.__version__,
                            torch_package=torch.__file__, transformers_package=transformers.__file__,
                            actual_device=delegate.device, actual_dtype=str(delegate.model.dtype),
                            cuda_device=torch.cuda.get_device_name(0) if delegate.device == "cuda" else None,
                            model_load_seconds=delegate.load_seconds)
        advisor = RecordedAdvisor(delegate, directory) if delegate is not None else None
        metadata["status"] = "simulating"
        write_json(directory / "run_metadata.json", metadata)
        episode = simulate(config, advisor, initial_planner=args.planner, advisor_name=args.advisor,
                           debug_hook=debugger)
        write_json(directory / "episode.json", episode)
        write_json(directory / "metrics.json", episode["metrics"])
        (directory / "decision_log.txt").write_text(render_decision_log(episode), encoding="utf-8")
        metadata["status"] = "episode_saved"
        write_json(directory / "run_metadata.json", metadata)
        if args.plot or args.gif:
            from uav_em.visualization import render
            render(episode, directory / "summary.png")
            if args.gif:
                render(episode, directory / "replay.gif", animate=True,
                       playback_speed=1, decision_pause_seconds=2)
        metadata.update(status="complete", finished_local=datetime.now().astimezone().isoformat(),
                        elapsed_seconds=time.perf_counter() - started, debug_stage_count=debugger.sequence)
        write_json(directory / "run_metadata.json", metadata)
        print(json.dumps(episode["metrics"], ensure_ascii=False, indent=2), flush=True)
        return directory
    except BaseException as exception:
        metadata.update(status="interrupted" if isinstance(exception, KeyboardInterrupt) else "failed",
                        error=f"{type(exception).__name__}: {exception}",
                        elapsed_seconds=time.perf_counter() - started)
        write_json(directory / "run_metadata.json", metadata)
        raise


if __name__ == "__main__":
    main()
