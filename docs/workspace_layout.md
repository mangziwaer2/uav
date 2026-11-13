# 工作区与结果目录

最新[离线LLM结构表达式生成v2](offline_structure_generation_v2.md)已完成4个独立真实设计会话（2初稿+2修订）、4份合法程序、16同状态求解及16次新闭环，全部到达，运行时0模型调用/策略错误，564测试通过。生成8个有限标量表达式，控制入口、前向接回点、优先级、样本/边额度、停止与直连/C-RRT*；不是任意Python或新GP/RRT*底层实现。38个原模块未改，四数值参考的16飞行/16配对明确复用并核验，未重新运行。draft_02飞行T+Q347.146→338.697（−2.43%）、总超参考94.664→91.101dB·s（−3.76%），但dev613的T+Q约+9.34%、超参考0→6.394，违反每城≤105%的预声明条件；修订未保留总体收益。**四产物均未通过，无胜者，本轮4份额度用尽，不追加第5份。** 16GIF均为新实际闭环；未见验证/测试继续封存。下一步先另冻dev613首次分岔与入口/接回/调度的因果归因协议，必要时研究对所有方法一致的预测/接受改进，不盲目追加生成或私改GP/C。

以下为既有阶段摘要，其中“下一步”是当时记录；当前待办仅以current_roadmap第15项为准。

前一[多接回点与分批竞争](reconnection_search_v1.md)已完成16同状态求解/16真实闭环，全部到达，0模型/新生成，544测试通过。新结构支持固定剩余路线上的多个接回点、最多6批独立C-RRT*和共享台账硬边切片；37原有模块不变。四组为单次/同点分批/多点均分/多点反馈，初始池、GP/C、首批随机流及700追加采样/2100总样本/10000边上限一致。本轮新配对流不拿上轮数值当分母。主反馈组真实飞行T+Q347.146→348.977（+0.528%）、超参考94.664→106.158dB·s（+12.142%），未获任务收益；均分T+Q−1.432%但超参考+6.772%，不是稳定避强或LLM收益。113决策GP重建、完整候选/批额度/几何能量核验，16GIF为实际闭环。本阶段已由最新结构生成研究接续，后续顺序以current_roadmap为准。

前一[C一致搜索原语](aligned_search_v1.md)已完成16同状态求解、8实际闭环，全部到达，0模型/新生成提交，536测试通过。新原语由研究者编写，让池后RRT*选父/重连/终点连接使用共同C；原GP/传播/接受器及36旧模块不变，共同池仍为旧J。两组同初始池、空搜索缓存、配对随机流和2100样本/10000边总账。8组状态比较7组当前C下降，但其中6组真实联合分变差；闭环合计飞行T+Q354.936→354.459（-0.134%），含悬停377.292→377.442，不改善。dev619超参考18.482→15.585dB·s，dev617为92.035→93.690而退步；总边+3.58%，无稳定避强/效率或LLM收益。8GIF为实际闭环。下一步进入多接回点、分批竞争搜索的结构设计，再冻结新离线生成合同/预算；不重复旧profile选择或追加旧生成名额，验证测试继续封存。

前一[因果可靠性反馈与在线选择](causal_feedback_v1.md)已完成：4个独立fork-none会话、7真实响应、16同快照求解，0错误/新飞行/新离线产物，529测试通过。原GP/传播/v3接受器及35个旧模块不变，各组初始公开状态与已付共同池资源一致。LLM在25.5s选择local_pad1→stop，真实联合分101.573→101.165（−0.401%）、超参考91.778→90.914dB·s（−0.941%）；其余路线不变。四项求解总边39748→34110（−14.18%），主要来自45.5s直接停止；模型等待/咨询92.9—217.4s/任务，无总时延优势。前三状态同一操作可被固定过程解释，尚无固定local_pad1对照/去可靠性字段消融或宏观避强证据。4GIF是实际选择后的开环反事实，无新点测/重规划、无真值最优参考。当时建议的固定操作对照现由C一致搜索研究接续，后续以current_roadmap为准；不追加本轮7响应/16任务或旧生成/飞行预算，验证测试继续封存。 下文旧阶段的“最新/下一步”均指当时，以current_roadmap为准。

更新：2026-11-11。当前研究为仅点功率感知下的LLM辅助绕行与动态重规划；一切运行结果从[outputs/index.html](../outputs/index.html)进入。基础记录为[interference_point_agent_20261109](../outputs/interference_point_agent_20261109/index.html)；第一版区段修复记录为[interference_route_repair_20261110](../outputs/interference_route_repair_20261110/index.html)，均已完成。此前[直接参数/预设算子程序比较](../outputs/interference_program_comparison_20261110/index.html)也已完成，写入独立目录，不覆盖上述证据。

```text
src/uav_em/                   仿真、传播、点测、局部估计、数值规划和决策接口
src/run_experiment.py          通用实验入口
src/replay_experiment.py       已保存episode的回放入口
tools/                        当前演示、在线agent实验和集中回放工具
configs/local_power.json      点功率/局部时空GP主配置
tests/                        当前保留能力的测试
docs/                         接手、当前方法、环境依据与信息合同
outputs/index.html            唯一结果总入口
outputs/interference_.../     每项新研究的独立完整记录
models/                       用户模型与既有最终权重，保留但不作为当前依赖
datasets/                     原始/派生数据，保留但不作为当前依赖
docs/submissions/             用户提交原件
temp/debug_full_pipeline.py   指定调试入口
temp/uav_cloud_env/            当前Python环境，保留原位避免路径失效
```

场景生成器的正式位置为`src/uav_em/scenarios.py`，不再挂靠地图编码训练包。当前不使用旧地图训练、初始稀疏输入、外部功率图或Sionna流程。测向/源假设内部兼容分支仍存在；主配置和本轮公共协议不启用，不据此宣称已清除源码每一个历史分支。

用户已授权删除旧实验、临时文件及废弃代码/文档，不创建归档。保留模型、原始数据、提交材料和来源不明的外部缓存；实际清理清单在本轮reports/cleanup_manifest.json；旧报告已删除。原始提交材料中的历史路径不代表当前存在。

每项研究使用一个新的`outputs/interference_<主题>_<日期>/`，已有记录不覆盖：

```text
index.html       人可读结果入口
RESULTS.md       结论、比较条件与限制
protocol.json    场景/方法/信息协议及运行前冻结信息
status.json      当前执行状态
scenarios/       仿真配置；隐藏源不可提供给决策agent
public/          决策agent允许读取的协议、控制、请求和自己的响应
runs/            按方法/场景/种子分组的episode、metrics、日志和静态图
reports/         汇总、审计及必要的冻结源码
replay/          集中GIF、index.html与来源清单
logs/            运行和验证日志
paired/          仅程序比较研究：因果估计快照与各方法配对结果；不提供给决策agent
```

共享目录不是信息隔离沙箱。决策agent只能读公共协议明确允许的文件，不可读隐藏场景或完整episode；模型输入、原始响应、接受/拒绝、实际执行和回退必须保留。

GIF集中在`replay/`，不在每个run散放。每次收尾更新研究入口和总入口，使用户无需搜索日志目录就能看表现。失败/追加尝试在同一研究下建清楚标注的新run，不新增含糊的resume根目录，不篡改旧响应或指标。

最新诊断入口`tools/run_repair_diagnostics.py`；`src/uav_em/repair_diagnostics.py`只构造因果候选，`diagnostic_evaluation.py`只做离线真值评价。新结果`outputs/interference_repair_diagnostics_20261110/`分cases/candidates/evaluations/figures/replay/reports/literature；不生成正常闭环episode。候选冻结前不评价未来真值，该阶段最小算子过程接口当时仅为草案，现已由有限过程v1接续。

最新过程入口`tools/run_process_study.py`，核心新模块`src/uav_em/repair_process.py`。`outputs/interference_process_baselines_20261110/`中programs存人工JSON过程，cases为精确快照，paired/evaluations分开保存因果结果与事后评价，runs为实际闭环，replay集中6个GIF；没有public决策目录，因为本轮无模型。

最新入口`tools/run_task_readiness.py`，新模块`src/uav_em/task_readiness.py`只评分当前GP候选，不改变控制。`outputs/interference_task_readiness_20261111/`保存4个新开发城市、13个真实因果快照、880候选/633可行；evaluations与冻结candidates分开，4个原控制器闭环GIF集中replay。reports含实验前设计、冻结源码、原序列化失败与修正、逐位GP重建和审计。当前437项测试通过，开发机会门槛0/4未通过；三种备选评分仅离线诊断。

最新共同目标v2入口`tools/run_objective_v2.py`，独立模块`src/uav_em/decision_objective.py`、测试`tests/test_decision_objective.py`。`outputs/interference_objective_v2_20261111/`中scenarios原样复制4个开发城市，runs为两数值方法8个真实闭环，run_manifest先冻结全部输出，reports保存协议/冻结设计/源码/预算几何能量核查/选路误判诊断，8GIF集中replay。无LLM/public请求或主动补测；448测试通过。旧v1、传播和GP保持不变，不把新数值目标收益算作模型贡献。

最新数值过程入口`tools/run_evidence_budget.py`，组合模块`src/uav_em/evidence_budget.py`、测试`tests/test_evidence_budget.py`。`outputs/interference_evidence_budget_20261111/`原样复制4开发城市，runs保存四人工方法16项真实闭环，4进程独立运行；run_manifest先冻结全部飞行，reports保存源码/设计、状态选择复算、拒绝分解、旧对照轨迹复现与GP风险重建，16GIF集中replay。459测试通过，主方法无优化收益，原27共有核心源码未变；无LLM/public请求或新补测。

最新新增`src/uav_em/offline_process.py`、`tests/test_offline_process.py`与单入口`tools/run_offline_contract.py`。`outputs/interference_offline_contract_20261111/`保存6个因果快照、24配对核验、冻结源码/合同和公开生成接口/人工参考；public状态仅含因果白名单。0生成调用/新飞行，无新replay/GIF，不把配对求解当闭环收益。

最新`tools/run_offline_generation.py`是离线过程生成单入口；`outputs/interference_offline_generation_20261111/`内public按四独立会话划分（合同/请求/原始响应、修订标量反馈），submissions分别保存解析产物/接受与错误，runs是28项新飞行，reports冻结源码/设计/分阶段manifest/审计/真实程序例子，28GIF集中replay。488测试通过，四过程均未通过收益门槛，运行时无模型咨询。

最新归因入口`tools/run_process_attribution.py`，`src/uav_em/process_attribution.py`只截断/核验，`process_attribution_evaluation.py`只事后真值评价。`outputs/interference_process_attribution_20261111/`中programs为诊断副本，cases为6因果快照，paired为30求解，source_pools为原16冻结池，runs为10新飞行，evaluations为固定10s反事实，replay集中10实际飞行GIF。不是第5份生成提交。

最新入口`tools/run_balanced_objective.py`，新模块`balanced_objective.py`和仅事后`balanced_evaluation.py`，旧31核心模块不改。`outputs/interference_balanced_objective_20261111/`保存冻结设计/源码、6因果状态36配对、4开发城24实际飞行、精确执行折线、连续真值事后评价和24集中GIF；先冻结再评价，不是第五份生成提交。

当前结构研究入口`tools/run_aligned_search.py`，实现`src/uav_em/aligned_search.py`。新结果`outputs/interference_aligned_search_20261111/`，16配对与8飞行分别冻结，replay集中8个实际闭环GIF；无public模型请求。

当前新增模块`structural_policy.py`为有限AST解释器、`generated_structure.py`为生成策略执行器；研究入口`tools/run_structure_generation.py`。最新结果`outputs/interference_structure_generation_20261111/`的public/submissions分别留请求与原始/解析程序，paired与runs分别为同状态求解和16新飞行，reports保留冻结、审计与全部失败，replay集中GIF；reference_paired为明确复用的旧参考。
