"""读取当前项目日志生成回放，无需重新加载模型或重新规划。"""

import argparse
from pathlib import Path

from uav_em.visualization import load_result, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1, help="抽帧间隔，正整数；保留全部决策帧，不改变倍速")
    parser.add_argument("--speed", type=float, default=1.0, help="播放倍速，必须为正数；1为仿真时间实时，2为两倍速")
    parser.add_argument("--decision-pause", type=float, default=2.0, help="决策帧额外停顿秒数，非负；不代表推理耗时")
    args = parser.parse_args()
    result = load_result(args.episode)
    render(result, args.output, animate=args.output.suffix.lower() == ".gif", frame_stride=args.stride,
           playback_speed=args.speed, decision_pause_seconds=args.decision_pause)
    print(f"Replay: {args.output.resolve()}")


if __name__ == "__main__":
    main()
