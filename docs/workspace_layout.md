# 项目目录用途

| 路径 | 用途 |
|---|---|
| README.md、AGENTS.md | 项目入口和维护边界 |
| src/run_experiment.py | 唯一活动入口；describe不飞行，event/reference/demo/joint为显式研究入口 |
| src/uav_em/ | 24个活动Python模块（含__init__）；D1新增joint_prediction与joint_experiment |
| configs/local_power.json | 当前合成参考配置；隐藏fields仅供仿真器，不直接发LLM |
| tests/、tools/check_workspace.py | 当前组件测试及原件/依赖/导航检查 |
| docs/AGENT_HANDOFF.md | 下一位agent先读的交接 |
| docs/method.md、current_roadmap.md | 唯一方法、唯一待办顺序 |
| docs/gp_equivalent_trigger_design.md | GP辅助等效场与预警总体设计，D1冻结参数及状态另见其合同/进度 |
| docs/history/ | 两份旧方法讨论/缓存解释，优先级已过时，不作并行计划 |
| docs/references/ | 已收文献及阅读记录，保留有效资料 |
| docs/maintenance/ | 清理、检查、哈希与旧工程证据 |
| outputs/index.html | 当前结果总入口 |
| outputs/interference_scenario_loop_20261118/ | 唯一现存旧C1研究，原件不改；GIF/显示衍生图在其replay/ |
| outputs/interference_joint_gp_d1_20261119_0940/ | 完整D1：合同/冻结源码、8飞行、18真实回复、评价/诊断和8条双图GIF；预算已结束 |
| models/、datasets/ | 用户模型/数据资产，保留；当前方法不把完整真图当先验 |
| 记录表/、docs/submissions/、申请书 | 用户实践和提交材料，保留原文 |
| temp/ | 现有虚拟环境和指定debug单入口转发，不存另一套算法 |

event_demo及timing_diagnostic目录当前缺失，不把目录表中的历史描述当现存文件。完整性检查仍报告既有缺失，不能更新旧哈希掩盖。

旧冻结源码和runtime_dependencies只属于C1来源见证，不是另一条活动实现。新实验独立outputs/interference_*并拒绝覆盖，GIF统一replay/；无需寻找resume目录。本轮不创建完整项目归档。
