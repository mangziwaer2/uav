# 下一位agent交接（2026-11-19）

**后续最新D2-M0：** 用户明确全图功率延续不应进入主假设池，授权继续新replay。[新合同](source_only_pilot_v1.md)下2条数值机制飞行已完成：早期源拟合134.09、失配修订139.96；均到达、几何路线相同，误差改善未带来修订任务收益，0次新决策LLM调用。[结果](source_only_results_20261120.md)、[新replay](../outputs/interference_source_only_d2_m0_20261120/replay/index.html)。主池去掉延续，8秒拟合，修订版按所有源旧预测失配触发新拟合及同事件路线复核；源码26模块、228测试通过。C1/D1原件哈希保持，本轮预算结束。D1当前活动源码哈希自新开发起已变化，不能调用D1旧check声称当前源码仍复现旧合同；只核旧原件及其冻结源码字节。

**2026-11-20最新状态：** 用户明确论文核心为LLM辅助决策优化，并回复“继续”授权独立决策会话。D1已完成8飞行、3独立会话/18份真实回复、冻结后真值评价及8条双图GIF；I与N代价相同，P增加6.33%，J增加10.47%，没有LLM优化收益。详见[最终结果与诊断](d1_results_20261120.md)、[回放](../outputs/interference_joint_gp_d1_20261119_0940/replay/index.html)。冻结源码及C1字节保持一致，219项测试记录保留，21个旧原件缺失照实报告。下文保留交接当时状态，勿将“尚未实现/唯一C1”误当D1当前状态。

先读本页，再读[方法](method.md)、[唯一待办顺序](current_roadmap.md)和[具体设计](gp_equivalent_trigger_design.md)。用户本轮要求整理目录并换agent，未要求开始新实验。

## 用户刚确认的方向

研究核心：复杂动态电磁环境下，基于LLM辅助决策优化的UAV路径重规划。UAV应根据有限信息提前避让高风险干扰区域，兼顾航程与暴露并到达goal。通信受强干扰影响是背景动机；用户明确纠正：不要将其扩成断联执行、链路建模或通信恢复研究。

用户认可GP作为辅助：用局部GP场约束等效场，缺失信息在飞行后续新点测中持续更新，逐步检验和修正假设。边飞边更新和短段规划，不等源模型完美，也不承诺轨迹不足时能唯一恢复真源。

目标链：点测 → GP局部估计 → 多等效场解释 → 前方有支持的预测越线时咨询 → LLM组织检验/细化/路径搜索 → 同一冻结预测下数值复核 → 执行短段 → 用新点测验证并更新。

咨询与换路分开：不必先找到有收益绕路才咨询；真正执行的路线再比较T+Q与硬约束。任务预警线须明示开发设定，不把−80dBm归一化值当设备失效线。MLP暂缓，若主线证明规则不足再考虑；不先训练网络/RL/本地LLM，不扩建异步架构或RAG。

## 当前真实实现

| 环节 | 已实现情况 / 尚未实现 |
|---|---|
| 环境 | 合成已知2.5D建筑、隐藏动态源、有限一次反射/等效穿楼；当前位置带噪总功率点测 |
| GP | 局部常均值时空GP，Matérn-3/2空间×指数时间；主配置60m/20s尺度、80m支持、35s年龄、64点上限；方差未校准 |
| 源解释 | 当前任务圆域160m内80m字典中心、40/80/160m尺度、有限静止/轴向平移/强度变化；贪心选至多2基元，仅连续拟合幅度；最近最多12条读数 |
| 证据池 | 新生声明质量，后续真实点测先预测后计分；拟合历史不再当出生后证据 |
| GP＋源 | 现在分开，mixture评分不含GP；GP只在独立参考与路径LLM局部提示中。场辅助拟合和联合预测均未实现 |
| 触发 | 旧fixed及power_rise：每点累计升6dB、冷却2秒、最多4事件；前方预测阈值规则未实现 |
| LLM | 同步推断pilot/refine与路径advisor接口；数值工具求解参数、共同接受器复核。原C1实际为数值推断＋LLM路径 |
| 路径 | 直连/中间点/受阻时有预算RRT*；只修未飞路线，共同航时＋功率积分及几何/航程/能量约束 |
| 时间 | 回调等待冻结仿真，固定0.5秒悬停；实际墙钟未映射物理时间，不称实机实时 |
| 可选资产 | 异步路径模块有组件检查，联合异步飞行未集成；神经预测器保留未部署；MLP/attention提议器未训练 |

## 现存证据与缺失项

唯一现存研究目录：`outputs/interference_scenario_loop_20261118/`（C1）。[结果入口](../outputs/index.html)。静态/动态×四方法共8飞行全到达，2独立决策会话/8真实回复；LLM相对数值混合平均T+Q为+6.08%，没有优化收益，不能写成联合方法已成功。

旧C1动态20秒接受浅南偏中间点；24.5秒池已退回近常值延续，仍执行20秒路线。当前切片不等于决策冻结预测，旧抽帧跳过决策后状态，详见[历史缓存说明](history/source_gp_case_memory_faq.md)。

回放选择：

- [overview双图](../outputs/interference_scenario_loop_20261118/replay/overview/index.html)：左真值，右单个有源替代解释，显示原权重；可切换混合。
- [full_mixture双图](../outputs/interference_scenario_loop_20261118/replay/full_mixture/index.html)：全池当前时刻功率期望，含方差；可能近同色，不是所有源同时叠加，也不是未来路线全程切片。
- 原8条GIF留在C1的replay/，不要覆盖或重画冒充新飞行。

`interference_event_demo_20261119`和`interference_timing_diagnostic_20261119`目前缺失。前者历史叙述为3飞行/9请求、联合LLM三次keep未增益；后者有时序影子诊断历史记录。只能称历史记载，不能称原件可核验，也不重建冒充恢复。旧预算已结束。

## 代码位置与继续方法

唯一入口src/run_experiment.py；22个活动Python模块（含__init__）在src/uav_em：

- point_gp.py：因果GP和支持边界。
- equivalent_field.py、event_inference.py：基元、拟合、池及出生后计分。
- event_llm_inference.py：LLM有限工具作业及反馈。
- scenario_planning.py、feasible_scenario.py、repair_search.py：冻结预测、整路评分/接受及预算搜索。
- decision_timing.py、scenario_loop.py：当前触发与同步飞行。
- replay_overview.py：旧缓存显示，非新研究结果。

下一步严格见[current_roadmap](current_roadmap.md)：先落小版本公开接口/融合与预警参数/预算合同，再接GP联合预测，单独检验GP场辅助拟合和简单前方预警，随后有限闭环与replay。不要求先完成整个四方法矩阵；但N/I/P/J及GP/延续参考必须保留为核心评价设计，不能被门控或外推工程替代。

真实决策agent必须独立新会话，仅给public allowlist及本任务已发生反馈；不能给完整配置中的隐藏源或事后真图。新预测/路线先冻再评价，不追加旧额度，不读封存城市。新后续点测先评价旧冻结版本，再更新GP和源；修订模型不继承重复历史成绩。

## 本轮整理与验证

本轮仅合并当前文档、移动两份旧讨论到docs/history、删冗余导航stub和允许删除的缓存，更新describe状态文字；没有改变规划/拟合/物理场算法，也没有新研究飞行、真实决策LLM或训练。C1全目录文件清单及哈希前后核对，模型/数据/记录表/提交材料/虚拟环境保留。

验证、移除文件哈希和移动清单见[整理记录](maintenance/handoff_cleanup_20261119.json)。代码测试与检查日志放同目录handoff_tests_20261119.log、handoff_workspace_check_20261119.json；结果以这些实际文件为准。完整性检查整理前已有21个时序诊断原件缺失，必须保留报告，不改旧哈希令其假通过。

```bash
temp/uav_cloud_env/bin/python src/run_experiment.py describe
temp/uav_cloud_env/bin/python -m pytest -q -p no:cacheprovider
temp/uav_cloud_env/bin/python tools/check_workspace.py
```

不自动git提交/推送。仓库有大量早先重构的未提交改动，别重置或归因于本轮；冻结实验源码只供原件见证，不作为另一条活动代码线。
