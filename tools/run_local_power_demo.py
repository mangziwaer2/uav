"""点功率→局部时空估计→数值重规划演示；不调用LLM，不作为算法收益实验。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_em.config import load_config
from uav_em.decision_log import render_decision_log
from uav_em.scenarios import ScenarioSettings, generate_scenario
from uav_em.simulation import simulate
from uav_em.visualization import render
from render_navigation_replays import render_replays


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/local_power.json")
    parser.add_argument("--output", type=Path, required=True, help="必须是新的独立实验目录")
    parser.add_argument("--random-city", action="store_true", help="复用现有有界随机建筑/发射源生成器；参数不是实测城市统计")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--planner", choices=("rrt_star", "astar"), default="rrt_star")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    generation = {"kind": "explicit_configuration"}
    if args.random_city:
        generated, generation = generate_scenario(config.seed, ScenarioSettings(
            columns=config.map.columns, rows=config.map.rows, resolution_m=config.map.resolution_m,
            building_count=6, min_sources=2, max_sources=3, horizon_seconds=config.simulation.max_seconds))
        # 保留点测/GP/任务配置，复用生成器的几何和源；不按飞行结果筛选场景。
        config = replace(config, map=generated.map, electromagnetic=replace(
            config.electromagnetic, fields=generated.electromagnetic.fields))
    config.validate()
    if config.sensor.mode != "power_only" or config.belief.estimation_method != "local_gp":
        parser.error("this demonstration requires power_only and local_gp")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("scenarios", "runs", "reports", "logs", "replay"):
        (output / name).mkdir()
    save(output / "scenarios/scene.json", config.to_dict())
    save(output / "reports/generation.json", generation)
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((ROOT / "src/uav_em").rglob("*.py"))}
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save(output / "protocol.json", {"kind": "sensing_estimation_smoke_demo_not_LLM_benchmark",
        "sources_sha256": sources, "scenario_sha256": hashlib.sha256((output / "scenarios/scene.json").read_bytes()).hexdigest(),
        "scene_chosen_before_flight": True, "random_city": args.random_city,
        "sensor": "power_only", "estimator": "local_gp", "planner": args.planner,
        "LLM_called": False})
    method = "numeric_rrt" if args.planner == "rrt_star" else "numeric_astar"
    run = output / "runs" / method
    run.mkdir()
    save(output / "status.json", {"status": "running"})
    def hook(stage, data):
        if stage in ("decision_executed", "finished"):
            print(stage, data["time_seconds"], flush=True)
    episode = simulate(config, initial_planner=args.planner, debug_hook=hook)
    save(run / "episode.json", episode)
    save(run / "metrics.json", episode["metrics"])
    (run / "decision_log.txt").write_text(render_decision_log(episode), encoding="utf-8")
    save(run / "status.json", {"status": "completed", "method": method, "scene": "local_power",
                               "seed": config.seed, **episode["metrics"]})
    if episode["decisions"]:
        save(output / "reports/first_visible_context.json", episode["decisions"][0]["context"])
    checks = {"no_directional_scan": all(e.get("directional_scan") is None for e in episode["events"]),
              "no_directional_history": all(not f["directional_observations"] for f in episode["frames"]),
              "zero_LLM_calls": episode["metrics"]["advisor_calls"] == 0,
              "observations_are_point_samples": True}
    save(output / "reports/interface_checks.json", checks)
    render(episode, run / "summary.png")
    if not args.no_gif:
        render_replays(output, frame_stride=4, playback_speed=2., decision_pause_seconds=.5)
    metrics = episode["metrics"]
    report = ("# 点功率与局部地图估计演示\n\n"
        "本次仅验证环境、点测、局部估计和数值规划闭环，没有调用LLM，不证明绕行优化有效。\n\n"
        f"- 到达：{metrics['success']}；终止：{metrics['termination']}。\n"
        f"- 航程：{metrics['executed_length_m']:.1f}m；飞行超参考积分：{metrics['interference_excess_integral_db_seconds']:.3f}dB·s。\n"
        f"- 飞行点测：{metrics['flight_observations']}；测向：0。\n\n"
        "左图为评价真值，右图只由点测估计。参考等值线不是信号截止边界或设备失效线。"
        "灰色区域未知；局部估计不是整个圆盘被测得。GP标准差未经实测校准，普通核可能跨建筑过度平滑。"
        "飞行定位、恒速与能量是研究简化，实际计算耗时不推进仿真。\n")
    (output / "RESULTS.md").write_text(report, encoding="utf-8")
    page = ('<!doctype html><meta charset="utf-8"><title>点功率与局部估计</title>'
        '<style>body{max-width:1100px;margin:30px auto;font:17px/1.8 sans-serif}img{max-width:100%}</style>'
        '<h1>点功率 → 局部时空估计 → 动态重规划</h1>'
        '<p>环境与接口演示，未调用LLM，不作为LLM路径优化结果。</p>'
        '<p><a href="../index.html">总入口</a> · <a href="replay/index.html">集中GIF回放</a> · '
        '<a href="reports/first_visible_context.json">决策可见输入</a> · '
        f'<a href="runs/{method}/decision_log.txt">可读日志</a> · <a href="RESULTS.md">结果说明</a></p>'
        f'<p>到达：{metrics["success"]}；航程：{metrics["executed_length_m"]:.1f}m；'
        f'超参考积分：{metrics["interference_excess_integral_db_seconds"]:.3f}dB·s；'
        f'飞行点测：{metrics["flight_observations"]}；测向：0。</p>'
        '<p>左侧是研究者真值；右侧是点测驱动的局部估计，灰色为未知。'
        '建筑与源为合成场景；黑色功率等值线只表示所选软参考，不是严格影响边界。</p>'
        f'<img src="runs/{method}/summary.png"><pre>{html.escape(report)}</pre>')
    (output / "index.html").write_text(page, encoding="utf-8")
    save(output / "status.json", {"status": "completed", "LLM_called": False, **metrics})
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(output / "index.html")


if __name__ == "__main__":
    main()
