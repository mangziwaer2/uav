# LLM辅助UAV路径重规划

目标：在局部带噪功率点测下，让LLM调用共同数值算子，检验相对强数值方法的实际全程T+Q增量。RP3六臂均合法到达，但平均代价上升8.75，尚无稳定优势。

- 接手只读[交接](docs/AGENT_HANDOFF.md)、[方法](docs/method.md)，约束见[AGENTS](AGENTS.md)。
- [最新三场回放](outputs/passive_repair_RP3_20261126/index.html) · [全部结果](outputs/index.html)。
- `src/uav_em/`：活动算法；`tests/`：回归/合成验收；`tools/`：维护；`docs/`：两页入口与冻结合同。
- `outputs/`：试飞证据；`outputs/development/`：静止夹具；`outputs/research_archive/`：默认不读的历史归档。
- `models/`、`datasets/`、`记录表/`、`docs/submissions/`、`temp/`：保留资产。

检查不调用模型或启动科学实验：

```bash
temp/uav_cloud_env/bin/python src/run_experiment.py describe
temp/uav_cloud_env/bin/python -m pytest -q -p no:cacheprovider
temp/uav_cloud_env/bin/python tools/check_workspace.py
```

各历史批次均已关闭，不补跑。检查器保留21个历史缺失，分别报告当前问题。旧稿合并为[一个ZIP](outputs/research_archive/handoff_compaction_20261126.zip)，不影响冻结合同与试飞原件。
