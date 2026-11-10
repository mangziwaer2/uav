"""当前实验入口：配置、数值基线／模型咨询、结果与离线回放。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from uav_em.advisor import LocalModelAdvisor, LocalModelConfig, MockAdvisor
from uav_em.config import load_config
from uav_em.decision_log import render_decision_log
from uav_em.simulation import simulate
from uav_em.visualization import render


def main() -> None:
    """使用同一仿真闭环切换后端；none与mock不能用于证明真实模型能力。"""
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/local_power.json", help="研究配置文件")
    parser.add_argument("--advisor", choices=("none", "mock", "local"), default="none",
                        help="none为数值基线，mock为固定模拟响应，local才调用本地模型")
    parser.add_argument("--planner", choices=("astar", "rrt_star"), default="rrt_star",
                        help="初始及数值回退规划器；模型可建议已适配的规划器")
    parser.add_argument("--model-path", type=Path, default=root / "models/Qwen3-1.7B",
                        help="已有本地模型目录；不自动下载，也不是远程推理地址")
    parser.add_argument("--model-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model-max-tokens", type=int, default=1024,
                        help="最大输出词元数；阶段计划增加输出长度，超限不自动修复JSON")
    parser.add_argument("--model-timeout", type=float, default=None,
                        help="可选生成软时限，单位秒，必须为有限正数；默认不限时，不是硬实时期限")
    parser.add_argument("--max-calls", type=int, help="覆盖模型调用上限；耗尽后明确记录数值回退")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, help="新建独立实验目录；已有目录拒绝覆盖")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--gif-speed", type=float, default=1.0, help="GIF播放倍速，正数；默认1倍仿真时间")
    parser.add_argument("--gif-decision-pause", type=float, default=2.0, help="GIF决策帧额外停顿秒数，非负")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.max_calls is not None:
        config = replace(config, simulation=replace(config.simulation, max_advisor_calls=args.max_calls))
    config.validate()
    advisor = None
    if args.advisor == "mock":
        advisor = MockAdvisor()
    elif args.advisor == "local":
        advisor = LocalModelAdvisor(LocalModelConfig(
            str(args.model_path), args.model_device, max_new_tokens=args.model_max_tokens,
            generation_seconds=args.model_timeout))
        advisor.load()
    directory = args.output_dir or root / "outputs" / f"interference_{args.advisor}_{args.planner}_seed{config.seed}"
    directory.mkdir(parents=True, exist_ok=False)
    result = simulate(config, advisor, args.planner, args.advisor)
    (directory / "episode.json").write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (directory / "metrics.json").write_text(json.dumps(result["metrics"], ensure_ascii=False, indent=2,
                                                      allow_nan=False), encoding="utf-8")
    (directory / "decision_log.txt").write_text(render_decision_log(result), encoding="utf-8")
    if not args.no_plot:
        render(result, directory / "summary.png")
    if args.gif:
        render(result, directory / "replay.gif", animate=True, playback_speed=args.gif_speed,
               decision_pause_seconds=args.gif_decision_pause)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"Results: {directory.resolve()}")


if __name__ == "__main__":
    main()
