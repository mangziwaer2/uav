# 当前可用结果

最新[M7交互诊断](interference_task_contrast_d2_m7_20261120/review/index.html)：同一M6 I状态下N/I选择受限单基元动态检验，3真实回复全及时、0回退；六项剖面均未找到排序翻转。无新飞行/真值/GP查询，无新GIF；路线/等效假设和参数图仅来自公开缓存与池外见证。[结果与遗漏](../docs/task_contrast_results_20261120.md)。315测试，预算关闭。

此前[M6六图回放](interference_constrained_geometry_d2_m6_20261120/replay/index.html)：固定M5原计划，5项显式区域约束工具复核。原LLM两项0/2→2/2接受，但6条方法/状态路线全未变、任务收益0。0新决策/飞行/点测；30模块、293测试，预算结束。[结果](../docs/constrained_geometry_results_20261120.md)。

最新[M5假设源可切换回放](interference_task_proposals_d2_m5_20261120/review/index.html)：3固定状态两阶段提议，6独立真实回复均及时、0回退；25秒数值/LLM剩余T+Q=111.70/112.91，其他相同，未获LLM收益。0新飞行/点测，277测试，预算结束。[结果](../docs/task_proposals_results_20261120.md)。

最新为[D2-M4带回执的固定状态对照](interference_task_allocation_d2_m4_20261120/review/index.html)：1全新独立会话产生3真实回复，1及时执行、2派发超时keep回退，实际三状态路线不变、反事实收益0；不能评价LLM优劣。[结果与调用核对](../docs/task_allocation_results_20261120.md)。0新飞行/点测，265测试通过，3调用预算已结束。图/GIF为冻结状态反事实，标明回退；冻结原图在其replay/，带回执显示副本在review/，无新增场查询。

此前[D2-M3同预算源几何细化两条新飞行](interference_source_geometry_d2_m3_20261120/replay/index.html)：25秒后实际动作分岔，Q93.33→86.78，但航程742.35→865.06m、T+Q144.82→146.45（+1.13%）；3个固定状态最优路线均未变。未获任务收益，0次新决策LLM调用，253项测试通过，预算结束。见[结果与任务作用诊断](../docs/source_geometry_results_20261120.md)。

此前[D2-M2活动源替换两条新飞行](interference_source_renewal_d2_m2_20261120/replay/index.html)：20.5秒暂停3个失配旧源，但两飞行观测/实际动作相同，T+Q均144.82，源一步RMSE3.37→3.88dB；3个同状态最优路线均未变。未获替换收益，0次新决策LLM调用。见[结果与局限](../docs/source_renewal_results_20261120.md)。本轮预算结束，243项测试通过。

此前[D2-M1侧向搜索](interference_source_search_d2_m1_20261120/replay/index.html)：航程768.89→742.35m，但T+Q139.96→144.82。同状态3次对照有2次排序失误、1次真值改善；[结果](../docs/source_search_results_20261120.md)及原件保留。

此前[D2-M0去除全图延续](interference_source_only_d2_m0_20261120/replay/index.html)：早期源拟合/失配修订T+Q=134.09/139.96，源误差改善未转成路线收益，0次新决策LLM调用。[结果](../docs/source_only_results_20261120.md)及原件保留，预算亦结束。

目前存在旧C1及完整[D1双图回放](interference_joint_gp_d1_20261119_0940/replay/index.html)，从[index.html](index.html)进入。D1已完成8飞行、18份真实LLM回复与事后评价：N/I=145.99、P=155.24、J=161.28，未获LLM任务收益；GP参考143.32。详见[结果与诊断](../docs/d1_results_20261120.md)。D1预算已结束，原数值阶段报告仍保留。

[D1带源标记回放](interference_d1_source_markers_20261120/index.html)为独立缓存显示副本：左图真值源，右图可切换一个假设等效源，保留原权重，底图仍为原规划聚合场。8条标记GIF位于该目录replay/；原D1原件未改，没有新增场预测或实验预算。流程解释见[当前源假设流程](../docs/d1_source_pipeline_20261120.md)。

- [overview双图](interference_scenario_loop_20261118/replay/overview/index.html)：左真值、右单个有源替代解释，显示原权重；可切换全池混合。
- [full_mixture双图](interference_scenario_loop_20261118/replay/full_mixture/index.html)：完整池当前时刻功率期望，含方差，不是所有源同时存在；未来路线另按到达时刻评分。
- [原8条飞行GIF](interference_scenario_loop_20261118/replay/index.html)：C1数值推断＋LLM路径，全部到达但LLM未胜数值混合，等待冻结物理时间。

event_demo_20261119和timing_diagnostic_20261119目录当前缺失，只有历史叙述，不能称原件可核验；本轮不重建、不重跑。原C1全部文件保留，不把旧显示改标签冒充GP联合新方法。

后续研究独立新outputs/interference_*，GIF集中replay/，拒绝覆盖旧结果。待实现方向和顺序见[交接](../docs/AGENT_HANDOFF.md)及[路线](../docs/current_roadmap.md)。
