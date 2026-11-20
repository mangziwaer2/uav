# LLM辅助复杂动态电磁环境下的UAV重规划

研究目标：仅凭飞行中的局部功率点测，提前判断前方干扰风险，借助LLM组织环境假设修订和路径搜索，使UAV绕开高风险区域并到达目标，兼顾航程与暴露。

当前目标流程：点测 → GP局部外推 → GP辅助等效场假设 → 前方预警 → LLM检验/修订及规划 → 数值复核 → 执行短段并持续更新。通信受干扰是应用动机，不另建断联/恢复研究线。

从这里开始：

- [下一位agent交接](docs/AGENT_HANDOFF.md)
- [唯一方法](docs/method.md) · [唯一实施顺序](docs/current_roadmap.md)
- [GP辅助等效场及预警设计](docs/gp_equivalent_trigger_design.md)
- [直接看现存回放与结果](outputs/index.html)
- [目录说明](docs/workspace_layout.md) · [文档索引](docs/README.md)

最新[D2-M0新replay](outputs/interference_source_only_d2_m0_20261120/replay/index.html)已移除主假设池的全图功率延续，完成2条数值机制飞行：早期源拟合T+Q=134.09，失配修订=139.96。源误差降低但几何航迹相同，修订未获任务收益；0次新决策LLM调用。[结果、同状态表示检查与局限](docs/source_only_results_20261120.md)。本轮预算结束，228项CPU测试通过。

当前D1已完成冻结GP/等效场联合预测、GP软形状拟合、前方预警及真实LLM闭环：[8飞行全部到达，3独立会话共18份真实回复](docs/d1_results_20261120.md)。实际T+Q：N/I=145.99、P=155.24、J=161.28，LLM尚未带来优化收益；独立GP参考143.32。可直接看[8条双图回放](outputs/interference_joint_gp_d1_20261119_0940/replay/index.html)。单个已见开发场景、同步等待冻结，不能作泛化/实机及时性结论。MLP、训练及进一步异步建设暂缓。

论文研究核心为LLM组织假设检验、修订与路径搜索的决策增量，见[待验证贡献](docs/paper_research_questions.md)。经典路径规划、GP新能力与工程模块数量不能直接充当LLM创新证据。

在项目根目录进行不启动飞行的检查：

```bash
temp/uav_cloud_env/bin/python src/run_experiment.py describe
temp/uav_cloud_env/bin/python -m pytest -q -p no:cacheprovider
temp/uav_cloud_env/bin/python tools/check_workspace.py
```

完整性检查已知报告21个时序诊断原件缺失；不是新增失败，不能修改旧清单掩盖。新研究须先冻结输入与预算，不直接续用旧demo合同。

[环境及文献依据](docs/local_power_environment.md) · [公开信息边界](docs/llm_information_contract.md) · [实践记录](记录表/README.md)
