# LLM辅助UAV动态路径重规划

复杂电磁环境是背景，研究核心是**LLM相对强数值方法的决策增量**，目前尚未证明。

- [交接与下一步](docs/AGENT_HANDOFF.md)：接手只需先读此页。
- [方法](docs/method.md) · [维护约定](AGENTS.md)。
- [回放与历史证据](outputs/index.html)：按需查看，不作为当前开发路线。

活动代码仅在 `src/uav_em/`；`tests/`保留其回归测试。旧实验入口已删除，当前不开放新科学运行命令。

```bash
temp/uav_cloud_env/bin/python src/run_experiment.py describe
temp/uav_cloud_env/bin/python -m pytest -q -p no:cacheprovider
temp/uav_cloud_env/bin/python tools/check_workspace.py
```

检查器分别报告当前完整性与21个历史原件缺失；不重跑实验。旧合同、文献和冻结代码集中于 `outputs/research_archive/`，默认检索忽略。
