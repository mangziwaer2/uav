# 文档索引

当前主线：无人机飞行中测量局部电磁干扰，更新地图与不确定性，由大语言模型（Large Language Model，LLM）辅助继续、重规划或主动补测。环境可随时间变化，但不以服务站或通信质量为核心。

| 文档 | 用途 |
|---|---|
| `AGENT_HANDOFF.md` | 新agent接手：研究边界、代码、运行和版本区别 |
| `CLOUD_HANDOFF.md` | 本次云端迁移：Linux复现、权重转移、真实失败证据及可复制的新agent任务 |
| `first_version.md` | 实际流程、信息隔离、指标、协议与回放 |
| `../temp/debug_full_pipeline.md` | 用户指定的完整流程调试入口：IDE／pdb断点、12阶段变量与真实模型环境说明 |
| `simulation_foundation.md` | 当前传播／360°范围感知的公式、硬件假设、信息边界、预警与未实现项 |
| `uav_sensing_reading.md` | 机载功率、阵列测向、区外干扰定位和无线电地图的实测论文、阅读入口及与代码的对应边界 |
| `current_roadmap.md` | 非代码决策→局部代码生成的四阶段路线、收敛条件与开发验收清单 |
| `electromagnetic_research_plan.md` | 干扰功率场、动态性与局部信息假设 |
| `实践项目申请书.md` | 正式长期方案，不是已实现功能清单 |
| `submissions/实践计划.docx` | 用户原始提交文件，未自动改写 |

参数见`../configs/README.md`，命令见根目录`README.md`，保留结果的场景版本见`../outputs/README.md`。原`llm_hierarchical_replanning_plan.md`内容已合并到`first_version.md`和`current_roadmap.md`，删除重复计划，不再维护另一份策略说明。

维护分工：入口与命令只在根目录；阶段与待办在路线文档；协议与代码流程在实现说明；环境假设及后期编码／代码方向在环境研究文档；接手文档仅汇总状态与入口。当前日志版本2；旧通信质量与历史天气结果不能当作本版实验，不恢复历史接口。用户原始提交文件保留不动。
