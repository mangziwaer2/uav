# LLM辅助UAV动态路径重规划

研究主线：在局部观测的复杂电磁环境中，由 LLM 调用路径修复算子生成、比较并提交未飞余程，检验能否在同信息、同工具、同资源下比强数值对照降低实际全程 `T+Q`。静止工具闭环已验证；全程收益尚未证明。

- [交接与下一步](docs/AGENT_HANDOFF.md)：接手只需先读此页。
- [方法](docs/method.md) · [维护约定](AGENTS.md)。
- [试飞回放入口](outputs/index.html) · [RP1 三场配对与双图](outputs/passive_repair_RP1_20261126/findings.md)：六次飞行合法到达，LLM 的全程 `T+Q` 在两场降低、一场升高。

**目录导航**

- `src/uav_em/`：活动实现；`tests/`：回归与合成验收；`tools/`：维护与核验工具。
- `docs/`：交接、方法和实验合同；[开发参考](docs/reference/README.md)：旧草案及诊断。
- `outputs/`：RP1 与 E1–E3 的试飞和双图；`outputs/development/`：静止验收和合成夹具；`outputs/research_archive/`：旧研究归档。
- `models/`、`datasets/`、`记录表/`、`docs/submissions/` 与 `temp/`：保留资产，不是活动算法入口。

RP1 实验已关闭。以下命令用于本地检查，无需调用模型或重跑科学实验。

```bash
temp/uav_cloud_env/bin/python src/run_experiment.py describe
temp/uav_cloud_env/bin/python -m pytest -q -p no:cacheprovider
temp/uav_cloud_env/bin/python tools/check_workspace.py
```

检查器分别报告当前完整性与21个历史原件缺失；不重跑实验。旧合同、文献和冻结代码集中于 `outputs/research_archive/`，默认检索忽略。
