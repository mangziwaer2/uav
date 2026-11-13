# 离线LLM结构表达式生成v2：协议与已完成结果

2026-11-11。用户继续授权新的离线结构生成。本轮独立于旧4份有限过程预算，旧预算不追加。先做真实离线生成，运行时0模型；尚不测试在线LLM选择或本地模型迁移。

## 生成什么

LLM生成8个有限标量表达式：entry_score、entry_offset、exit_score、priority、sample_quota、edge_quota、stop、operation，以及1..3个接回点额度。表达式可含算术、min/max/abs/sqrt/log1p、比较、布尔和条件分支；单式≤1200字符/160 AST节点/24层，无变量赋值、循环、导入、属性、文件、网络、下标、exec/eval。AST由手写解释器计算，不执行任意Python。数值常数不是设备标定。

这生成新的入口/接回/调度评分与预算规则，不再只选择旧六profile；但不声称生成传播/GP/RRT*实现或任意源代码。实现src/uav_em/structural_policy.py、generated_structure.py。每轮固定共同C选出的剩余路线，entry_score选段、entry_offset在该段前移0..3个锚点；对后续全部锚点计算公开几何/当前GP特征，exit_score选至多3个接回点。priority按同快照数值反馈选候选；operation选择C-RRT*或直连（都复核）。最多6步；stop可提前停止。未知仍用原先验，不新增测向、真源或未来场。

每决策新增搜索累计≤700样本，连同已付共同池在2100样本/10000边全账内。候选弦几何检查计边，积分/解释器另计墙钟。生成表达式预算按整数向下取整并裁剪到剩余额度；负数、非有限、类型/算术错误记录policy_error、停止并保留共同池可行路线，不另送搜索预算。LLM不改C、GP、1.25D0全程约束或能量。双方对应原入口/接回点/重复次数沿用已冻结配对种子；新入口有独立键，禁止模型选择随机种子。

## 新冻结预算

- 2个独立fork-none初稿会话（draft_01/02），各一次原始提交；随后每稿最多1个独立fork-none修订（revision_01/02），最多4会话/4提交，无额外重试。拒绝提交亦占额度，不由研究者修好后冒充模型产物。
- 每个合法产物4既有开发城市611/613/617/619各1次真实闭环，共最多16新飞行；每产物4个既有dev617因果快照各求解1次，共最多16新配对。失败不补seed，不读验证701/709及测试种子。
- 4组数值参考完整复用上一轮冻结的16飞行/16配对，保留其来源哈希、源码/场景一致性和原始记录链接，不冒充新运行。共有38原模块（含reconnection_search）保持不变；生成执行器另立，不改变参考实现。计算墙钟来自不同批次，不能据此做严格时延优越结论。
- 每阶段全部产物/候选/实际运行先冻结，再衍生真值评价。初稿评价完成后，修订只得到自身父版本及四数值参考的逐开发场景标量反馈、解析/执行错误；无路径、真源、真图、未来脚本、别的生成产物或验证测试。反馈包含未来评价的开发标量是离线适应度，不能进入运行时决策。

## 会话信息与证据

每会话只能读自己public子目录的合同、请求、当前GP因果特征示例及（修订时）自身父提交/标量反馈；不得读项目AGENTS/源码/配置/其他实验或交叉交流。共享FS是流程allowlist，不是OS沙箱。请求/原始响应/解析接受/冻结表达式/运行错误/实际选择与候选分别存储；代理标识、时长和不可得词元null留证。

## 评价与预声明选择

固定任务效用是实际飞行航时+归一化线性功率积分（λ=1，P0=−80dBm），并单报固定决策悬停、超参考dB·s、航程、能量、搜索及模型成本。通过开发候选门槛须：四场景到达且硬约束/执行审计通过、0策略错误；相对single_C合计飞行T+Q至少改善1%，含悬停合计T+Q改善，合计超参考不增加，每场景T+Q不超过single_C的105%。这些是研究筛选标准，不是设备安全或干扰容忍阈值。通过者取飞行T+Q最小，随后超参考、航程破同值；全失败则无胜者，不改门槛。

四数值参考全部逐场景呈现，multi_equal/single_batched/multi_feedback不因结果被排除。开发选择不证明泛化或LLM必要；C改善不等于未来暴露改善。若固定C本身阻挡有效绕行，须另立对所有方法一致的预测/评价研究，不能私改共同接受器。GIF只画最多16新实际飞行，集中replay，参考GIF从原记录链接访问。

入口tools/run_structure_generation.py，新建outputs/interference_structure_generation_20261111并拒绝覆盖；prepare/submit/initial/initial_summary/request/revisions/evaluate/report分阶段，脚本不伪造LLM响应。

## 执行结果（前瞻协议之后追加）

原前瞻文本已在运行前冻结为[reports/frozen_design.md](../outputs/interference_structure_generation_20261111/reports/frozen_design.md)，本文追加结果后不声称与冻结文档哈希相同。实验源码、请求和运行结果的独立哈希仍须核验。

真实完成4独立会话/4合法程序、16同状态求解/16新闭环，全部到达，运行时0模型/策略错误；四数值参考旧16飞行/16配对明确复用、0新参考飞行。新增20项测试、全量564通过（1第三方弃用警告），原38模块未改。

single_C合计飞行T+Q347.146、超参考94.664dB·s；draft_01为346.138/94.721，draft_02为338.697/91.101，revision_01为352.683/102.529，revision_02为349.018/100.225。draft_02总分约−2.43%、总超参考约−3.76%，但dev613约+9.34%、0→6.394dB·s，违反每城≤105%条件；其他产物也不满足完整门槛。**全部失败，无胜者，不追加第5份，不改变门槛。** 修订只得到自身标量反馈，恢复了dev613零超参考，却没有保留总收益。

[研究入口](../outputs/interference_structure_generation_20261111/index.html) · [重点GIF](../outputs/interference_structure_generation_20261111/replay/focus.html) · [完整结果/逐门槛判定](../outputs/interference_structure_generation_20261111/reports/findings.md) · [真实表达式与执行](../outputs/interference_structure_generation_20261111/reports/generated_examples.md) · [审计](../outputs/interference_structure_generation_20261111/reports/evidence_verification.json)。16GIF均为新实际闭环；paired未来指标是固定路线反事实。开发适应度不证明稳定避强/在线必要或泛化。下一步按current_roadmap另冻分岔归因协议；若改共同预测/接受器，对所有方法一致，不追加本轮生成。

最终证据核验：114次新飞行决策的因果GP重建、2953项候选C检查、383项内部C检查、527项表达式执行步骤复算通过；16GIF共179帧逐帧解码通过。源码/输入/会话/原始响应与阶段结果清单均一致，核验未调用新规划或模型。
