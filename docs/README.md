# 当前文档入口

更新：2026-11-09。主线只采用合成城市、隐藏动态源、当前位置功率点测、局部时空GP和LLM辅助重规划。不使用初始稀疏地图、完整地图重建、地图编码或训练。内部仍存在的兼容代码不作为当前实验能力。

先看[结果总入口](../outputs/index.html)；本轮[点功率在线子agent验证](../outputs/interference_point_agent_20261109/index.html)已完成，结果以该目录真实记录为准，不沿用旧测向实验结论。

| 文档 | 负责内容 |
|---|---|
| [AGENT_HANDOFF.md](AGENT_HANDOFF.md) | 新agent接手、当前任务和信息/清理边界 |
| [current_roadmap.md](current_roadmap.md) | 唯一研究顺序：基本可用性→具体方法→公平验证 |
| [first_version.md](first_version.md) | 当前已实现流程、决策/执行、评价和未实现能力 |
| [local_power_environment.md](local_power_environment.md) | 城市/源生成、传播公式、机载点测和文献依据 |
| [llm_information_contract.md](llm_information_contract.md) | 点测到GP的计算、公开字段来源、禁止信息和因果边界 |
| [research_method_20261109.md](research_method_20261109.md) | 两项候选贡献与具体拟议方法；不是完成声明 |
| [workspace_layout.md](workspace_layout.md) | 当前代码、保留资产与结果目录约定 |

根[README.md](../README.md)负责快速运行与结果入口，[AGENTS.md](../AGENTS.md)负责维护约定。当前配置含义见[configs/README.md](../configs/README.md)，工具职责见[tools/README.md](../tools/README.md)。

[实践项目申请书.md](实践项目申请书.md)及[submissions/](submissions/README.md)是用户原有正式材料，保留原件性质，不作为当前能力或研究待办清单。`models/`和`datasets/`中的旧研究资产同样不构成本轮依赖。

状态以交接文档和实际结果为准，阶段只维护在路线中，物理依据与信息合同各有一个入口。不得用旧测试计数、历史目录链接或未来设计冒充当前证据；所有模型与数值回退分别记账。
