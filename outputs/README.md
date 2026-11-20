# 当前可用结果

最新为[D2-M0去除全图延续后的两条新飞行](interference_source_only_d2_m0_20261120/replay/index.html)：均到达；早期源拟合T+Q=134.09，失配修订=139.96。源误差改善未转成路线收益，0次新决策LLM调用。见[结果与局限](../docs/source_only_results_20261120.md)。本轮新预算已结束，未追加D1。

目前存在旧C1及完整[D1双图回放](interference_joint_gp_d1_20261119_0940/replay/index.html)，从[index.html](index.html)进入。D1已完成8飞行、18份真实LLM回复与事后评价：N/I=145.99、P=155.24、J=161.28，未获LLM任务收益；GP参考143.32。详见[结果与诊断](../docs/d1_results_20261120.md)。D1预算已结束，原数值阶段报告仍保留。

[D1带源标记回放](interference_d1_source_markers_20261120/index.html)为独立缓存显示副本：左图真值源，右图可切换一个假设等效源，保留原权重，底图仍为原规划聚合场。8条标记GIF位于该目录replay/；原D1原件未改，没有新增场预测或实验预算。流程解释见[当前源假设流程](../docs/d1_source_pipeline_20261120.md)。

- [overview双图](interference_scenario_loop_20261118/replay/overview/index.html)：左真值、右单个有源替代解释，显示原权重；可切换全池混合。
- [full_mixture双图](interference_scenario_loop_20261118/replay/full_mixture/index.html)：完整池当前时刻功率期望，含方差，不是所有源同时存在；未来路线另按到达时刻评分。
- [原8条飞行GIF](interference_scenario_loop_20261118/replay/index.html)：C1数值推断＋LLM路径，全部到达但LLM未胜数值混合，等待冻结物理时间。

event_demo_20261119和timing_diagnostic_20261119目录当前缺失，只有历史叙述，不能称原件可核验；本轮不重建、不重跑。原C1全部文件保留，不把旧显示改标签冒充GP联合新方法。

后续研究独立新outputs/interference_*，GIF集中replay/，拒绝覆盖旧结果。待实现方向和顺序见[交接](../docs/AGENT_HANDOFF.md)及[路线](../docs/current_roadmap.md)。
