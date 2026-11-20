# 文档入口

下一位agent按顺序阅读：

1. [交接与真实状态](AGENT_HANDOFF.md)
2. [唯一研究方法](method.md)
3. [唯一后续顺序](current_roadmap.md)
4. [GP辅助等效场及前方预警的具体设计](gp_equivalent_trigger_design.md)

| 材料 | 用途 |
|---|---|
| [共享工具v2规则](shared_tools_v2.md)、[修复结果](shared_tools_v2_results_20261120.md) | 已修F02–F08，361测试；活动源码更新，旧科研原件不改，无新科学运行；下一步统一任务接口 |
| [最新全流程审查](reviews/20261120_llm_decision_audit/README.md)、[新文献选读](references/llm_decision_review_20261120/README.md) | 共享工具/接口反例、有限反馈修订方法与唯一修改顺序；生产源码未改，315测试，无新科学运行 |
| [环境说明](local_power_environment.md)、[信息合同](llm_information_contract.md) | 仿真假设、文献和模型可见信息 |
| [现有等效场逐步说明](equivalent_field_walkthrough.md) | 已实现字典/拟合/证据及旧replay解释；新设计不冒充已有能力 |
| [核心评价设计](joint_inference_planning_v1.md) | N/I/P/J及GP/延续参考，尚未运行完整主实验 |
| [D1合同](joint_gp_pilot_v1.md)、[最终结果与诊断](d1_results_20261120.md) | 8飞行/18真实回复与真值评价回放完成，尚无LLM优化收益；旧[数值阶段](d1_progress_20261119.md)保留历史状态 |
| [D2-M0合同](source_only_pilot_v1.md)、[结果](source_only_results_20261120.md) | 无全图延续、早期源拟合/失配修订的2条数值飞行；未获修订任务收益 |
| [D2-M1合同](source_search_pilot_v1.md)、[结果](source_search_results_20261120.md) | 侧向搜索2飞行/3同状态诊断；路线改变但总代价上升，预测排序仍不稳定 |
| [D2-M2合同](source_renewal_pilot_v1.md)、[结果](source_renewal_results_20261120.md) | 失配旧源替换2飞行/3同状态诊断；活动池变化但行动/代价相同，未获收益 |
| [D2-M3合同](source_geometry_pilot_v1.md)、[结果](source_geometry_results_20261120.md) | 同预算源中心/尺度细化2飞行/3固定状态诊断；真实动作改变、暴露降低，但T+Q增加1.13%，未获任务收益 |
| [D2-M7合同](task_contrast_pilot_v1.md)、[最新结果](task_contrast_results_20261120.md) | 6个受限动态剖面未找到排序翻转，3真实回复均及时、0回退；无新飞行/真值/GP，见证不入池，近竞争覆盖仍有缺口 |
| [D2-M6合同](constrained_geometry_pilot_v1.md)、[最新结果](constrained_geometry_results_20261120.md) | 固定M5提议的5项约束工具复核；I两项均接受，但6条方法/状态路线均未变；0新决策/飞行/点测 |
| [D2-M5合同](task_proposals_pilot_v1.md)、[最新结果](task_proposals_results_20261120.md) | 3固定状态两阶段机制提议；6真实回复均及时、0回退，未获LLM收益；下一步核验共享区域约束工具 |
| [D2-M4合同](task_allocation_pilot_v1.md)、[最新结果](task_allocation_results_20261120.md) | 3固定状态工具分配；1独立会话3真实回复但仅1及时执行，2派发超时回退；路线均未变，不能评价LLM优劣 |
| [论文研究问题](paper_research_questions.md) | 核心为LLM组织假设修订与路径搜索的决策增量，贡献仍待检验 |
| [结果入口](../outputs/index.html) | 最新M7及旧M6/M5/M4/M3/M2/M1/M0/D1/C1原件；旧事件和时序诊断目录缺失 |
| [目录用途](workspace_layout.md) | 文件放哪里、哪些必须保留 |
| [文献索引](references/README.md) | 原文、来源、实际阅读范围与限制 |
| [历史讨论](history/README.md) | 被后续讨论替代的旧优先级/案例解释，不是当前待办 |
| [维护记录](maintenance/handoff_cleanup_20261119.json) | 本轮移动/删除/原件哈希及实际检查 |

event_demo_v1.md、event_demo_public_contract.md、async_replanning_v1.md保留为旧合同/工程依据，不能直接当下一轮实验授权或当前研究前置。旧清理/测试记录在maintenance/，不复用旧阶段计数。

[申请书](实践项目申请书.md)、submissions/和[实践记录](../记录表/README.md)保留原文，不作为当前代码状态或唯一待办。
