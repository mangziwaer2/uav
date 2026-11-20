# 文档入口

下一位agent按顺序阅读：

1. [交接与真实状态](AGENT_HANDOFF.md)
2. [唯一研究方法](method.md)
3. [唯一后续顺序](current_roadmap.md)
4. [GP辅助等效场及前方预警的具体设计](gp_equivalent_trigger_design.md)

| 材料 | 用途 |
|---|---|
| [环境说明](local_power_environment.md)、[信息合同](llm_information_contract.md) | 仿真假设、文献和模型可见信息 |
| [现有等效场逐步说明](equivalent_field_walkthrough.md) | 已实现字典/拟合/证据及旧replay解释；新设计不冒充已有能力 |
| [核心评价设计](joint_inference_planning_v1.md) | N/I/P/J及GP/延续参考，尚未运行完整主实验 |
| [D1合同](joint_gp_pilot_v1.md)、[最终结果与诊断](d1_results_20261120.md) | 8飞行/18真实回复与真值评价回放完成，尚无LLM优化收益；旧[数值阶段](d1_progress_20261119.md)保留历史状态 |
| [论文研究问题](paper_research_questions.md) | 核心为LLM组织假设修订与路径搜索的决策增量，贡献仍待检验 |
| [结果入口](../outputs/index.html) | 旧C1原件及完整D1；旧事件和时序诊断目录缺失 |
| [目录用途](workspace_layout.md) | 文件放哪里、哪些必须保留 |
| [文献索引](references/README.md) | 原文、来源、实际阅读范围与限制 |
| [历史讨论](history/README.md) | 被后续讨论替代的旧优先级/案例解释，不是当前待办 |
| [维护记录](maintenance/handoff_cleanup_20261119.json) | 本轮移动/删除/原件哈希及实际检查 |

event_demo_v1.md、event_demo_public_contract.md、async_replanning_v1.md保留为旧合同/工程依据，不能直接当下一轮实验授权或当前研究前置。旧清理/测试记录在maintenance/，不复用旧阶段计数。

[申请书](实践项目申请书.md)、submissions/和[实践记录](../记录表/README.md)保留原文，不作为当前代码状态或唯一待办。
