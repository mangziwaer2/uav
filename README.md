# LLM辅助复杂动态电磁环境下的UAV重规划

研究目标：仅凭飞行中的局部功率点测，提前判断前方干扰风险，借助LLM组织环境假设修订和路径搜索，使UAV绕开高风险区域并到达目标，兼顾航程与暴露。

当前目标流程：点测 → GP局部外推 → GP辅助等效场假设 → 前方预警 → LLM检验/修订及规划 → 数值复核 → 执行短段并持续更新。通信受干扰是应用动机，不另建断联/恢复研究线。

从这里开始：

- [最新：共享工具v2修复、合成回归与361项测试](docs/shared_tools_v2_results_20261120.md)
- [最新：LLM方法与全流程审查、反例及修改顺序](docs/reviews/20261120_llm_decision_audit/README.md)
- [下一位agent交接](docs/AGENT_HANDOFF.md)
- [唯一方法](docs/method.md) · [唯一实施顺序](docs/current_roadmap.md)
- [GP辅助等效场及预警设计](docs/gp_equivalent_trigger_design.md)
- [直接看现存回放与结果](outputs/index.html)
- [目录说明](docs/workspace_layout.md) · [文档索引](docs/README.md)

本轮已完成审查中7类共享工具问题的修复：新基元竞争、合法进展保留、分腿搜索、GP支持及回执等；31模块、361测试通过，无新科学运行。当前进入统一任务接口的组件集成，旧实验不重跑。[修复结果与边界](docs/shared_tools_v2_results_20261120.md)。

此前完成文献与全流程审查，发现共享工具和任务接口问题，已留合成反例与修改顺序；生产源码/旧结果未改，315测试通过，没有新科学运行。[详细报告](docs/reviews/20261120_llm_decision_audit/README.md) · [文献与实际阅读范围](docs/references/llm_decision_review_20261120/README.md)。

此前[D2-M7交互诊断](outputs/interference_task_contrast_d2_m7_20261120/review/index.html)完成6项受限动态相容检验，3份独立真实回复均及时、0回退；六个受检剖面未找到排序翻转。0新飞行/点测/GP/真值查询，源池/权重/路线不变；不能据此证明全局排序确定或LLM任务收益。[结果和覆盖缺口](docs/task_contrast_results_20261120.md)。31模块、315测试，预算关闭。

此前[D2-M6约束工具复核回放](outputs/interference_constrained_geometry_d2_m6_20261120/replay/index.html)固定原提议重算5项几何作业：原LLM两项现在均合法接受，确认旧工具错失可行修订；但6条方法/状态路线和短段均未变，任务收益0。[结果及归因](docs/constrained_geometry_results_20261120.md)。0新决策LLM/飞行/点测，30模块、293测试，预算结束。

此前[D2-M5可切换假设源回放](outputs/interference_task_proposals_d2_m5_20261120/review/index.html)完成两阶段机制/区域/源数提议。1新独立会话6份真实回复全部及时、0回退；25秒数值/LLM冻结剩余T+Q=111.70/112.91，其余两状态相同，未获LLM优化收益。[结果及下一步](docs/task_proposals_results_20261120.md)。0新飞行/点测，29模块、277测试，预算结束；下一步先核验共享工具对半域约束的求解。

此前[D2-M4固定状态回放](outputs/interference_task_allocation_d2_m4_20261120/review/index.html)已引入1个全新独立子agent、3份真实决策回复，但仅1份及时执行、2份派发超时回退。三状态N/I路线均未变、冻结路线收益为0，尚不能评价LLM增量。[结果与完整回执](docs/task_allocation_results_20261120.md)。本轮0新飞行/点测，265测试通过，3次调用预算结束；下一轮先接稳独立会话连续服务，再检验任务相关假设/机制提议。

此前[D2-M3新replay](outputs/interference_source_geometry_d2_m3_20261120/replay/index.html)比较同预算幅度与中心/尺度细化：25秒后实际动作分岔，真实暴露Q93.33→86.78，但航程742.35→865.06m、T+Q144.82→146.45（+1.13%）。3个固定状态最优路线均未变；未获任务收益。[结果与下一步](docs/source_geometry_results_20261120.md)。0次新决策LLM调用，253项CPU测试通过，预算结束。下一步检验LLM选择任务相关修订/细化的组织增量。

此前[D2-M2新replay](outputs/interference_source_renewal_d2_m2_20261120/replay/index.html)检验失配后替换旧源：20.5秒暂停3个旧版本，但两条数值飞行实际动作/观测相同，T+Q均144.82，源预测RMSE3.37→3.88dB，3个同状态最优路线也未变。未获替换收益，仅保留可选消融。[结果与下一步](docs/source_renewal_results_20261120.md)。0次新决策LLM调用，243项CPU测试通过，本轮预算结束。

此前[D2-M1](outputs/interference_source_search_d2_m1_20261120/replay/index.html)补入侧向搜索：航程768.89→742.35m，但T+Q139.96→144.82，3个同状态中2次排序失误、1次真值改善；[结果](docs/source_search_results_20261120.md)保留。新主池继续排除全图功率延续。

此前[D2-M0](outputs/interference_source_only_d2_m0_20261120/replay/index.html)移除主池延续，早期源拟合/失配修订T+Q=134.09/139.96，源误差降低但几何航迹相同，修订未获任务收益；[原结果](docs/source_only_results_20261120.md)保留。

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
