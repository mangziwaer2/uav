# 当前工具

- `run_local_power_demo.py`：随机/指定城市的点测→局部GP→数值规划闭环，不调用LLM。`--output`必须是新目录。
- `run_point_agent_study.py`：准备冻结的基础点测场景与公共合同，然后通过文件桥接器接受真实在线决策agent响应。程序不会自动召唤LLM；无外部决策进程时会等待并最终记为失败。
- `summarize_point_agent_study.py`：从已有episode/请求/响应生成对照表、调用审计、静态图及集中GIF，不重新做策略决策。
- `reproduce_point_agent_study.py`：用已保存的真实动作在当前源码上重放闭环，核验轨迹和指标；这属于记录动作重放，不是新的模型调用或独立重复实验。
- `render_navigation_replays.py --root <研究目录>`：只从已保存episode重建`replay/`画廊，绝不重新规划。

每项的准确参数用`--help`查看。旧训练、地图编码、LoRA、菜单研究和外部射线工具已删除；模型/原始数据保留不意味着当前还依赖它们。

- `run_repair_study.py`：第一版固定目标/总预算下的区段修复与RRT*采样分配。prepare冻结新场景，repair_numeric/repair_single/repair_feedback分别运行，report生成汇总及GIF；LLM方法需独立文件决策进程。

- `reproduce_route_repair.py`：用已保存的真实提议在当前代码重放三项修复任务，核验轨迹、指标及预算；不调用新模型、不补造原运行缺失的总耗时。

- `run_program_study.py`：直接参数LLM与预设算子短程序LLM的比较。prepare冻结两个既有场景；baseline采集4个因果同快照任务；direct_values/operator_program各处理配对任务和s02闭环；report生成审计和集中GIF。两模型组需独立文件决策进程。程序只解释预设函数调用，不执行任意Python；同库数值对照单列。

- `run_repair_diagnostics.py`：读取保留的4个精确GP快照，prepare复制/冻结输入，propose仅据当前估计生成全部候选并保存哈希，evaluate才用真值事后评价，report生成诊断图和集中GIF。all顺序完成四阶段且拒绝覆盖新目录。0模型调用，固定路线开环反事实，不是新闭环方法；literature独立下载5篇全文与来源，不把下载冒充方法已审阅。

- `run_process_study.py`：有限过程数值对照，prepare冻结合同/两份人工JSON过程/源码/既有快照；paired运行36同条件任务并冻结结果，evaluate事后评价最终固定路线，flights运行6闭环，report生成审计/指标和集中GIF；all按序完成。三组都不调用模型，0生成代码，最多6个前向节点/共享预算。

- `run_task_readiness.py`：新开发城市与目标一致性检查。all顺序prepare/scenarios/collect/propose/evaluate/report；新目录拒绝覆盖，4个预声明开发种子，不读取验证/测试。原library_numeric真实飞行采集13快照，固定几何池及三评分选择先冻结后评价未来真值；备选评分不进控制，0模型调用，GIF是真实基线飞行。结果未通过机会门槛，不把它当作新LLM方法。

- `run_objective_v2.py`：共同候选池的旧J与10s风险v2数值闭环。all顺序prepare/flights/evaluate/report，8飞行全部保存并冻结后才做衍生评价；新目录拒绝覆盖。v2使用全程航程上限、显式未知代价，同原GP/传播，无模型/补测；悬停积分事后单列。合计改善但dev619显著退步，门槛未通过；人工目标不能当LLM优化证据。

- `run_evidence_budget.py`：四人工过程×四已用开发城市，prepare/flights/evaluate/report，all顺序执行；4进程、16真实闭环、新目录拒绝覆盖，全部冻结后再评价。共用原v2候选与预算，分别旧J/即时R10/证据条件/证据+额度+支持条件，无模型/补测。最新主方法复现旧J、未通过收益门槛；额外拒绝分解不冒充新闭环消融。

- `run_offline_contract.py`：all顺序prepare/verify，新输出拒绝覆盖；仅从既有已收到点测重建6状态，24同状态修复、共同初态池/风险/资源/硬约束及提交接口等价核验。0模型/新飞行，公开contract.json冻结最多4份离线生成的下一阶段协议。

- `run_offline_generation.py`：真实离线子agent经public提交2初稿+2修订，prepare/references/submit/initial/initial_summary/request/revisions/evaluate/report分阶段；共同数值参考12飞行、生成产物最多16飞行，每产物一次评价，拒绝覆盖。脚本不召唤或伪造LLM；冻结全部阶段结果再生成标量反馈，最终28GIF集中replay。最新四产物均未通过门槛，不追加名额。

- `run_process_attribution.py`：prepare/paired/flights/evaluate/report，all按序；6状态×5模式30求解、两城10新飞行，0模型/新生成提交。保留全部6节点/额度后缀，只截断跳转，前缀逐位核验；冻结候选后才评价10s真值。新目录拒绝覆盖，GIF集中replay。

- `run_balanced_objective.py`：prepare/paired/flights/evaluate/report或all，6组×6状态36配对、4城24新飞行，无模型/新生成。整条剩余路线航时与连续功率共同评价，配对随机种子，保留旧J搜索；全部冻结后复算精确执行折线功率与初段进度，24GIF集中replay。主lambda1，0.25/4只敏感性，不根据结果挑权重。

- `run_risk_diagnostics.py`：prepare/capture/propose/evaluate/report或all。仅重建dev617四个已发生状态，复算原J/R10/v3、保留原池并有限扩大；所有候选先冻结再评价。48独立诊断搜索总成本单列，不是在线共享额度；0模型/新飞行/生成，4开环反事实GIF集中replay，新输出拒绝覆盖。

- `run_causal_feedback.py`：prepare冻结4既有快照/原共同池/公开合同，numeric完成12参考，model --task task01..04分别等待四个独立决策会话（最多2响应/任务）；freeze冻结16结果再evaluate/report。共同GP/v3接受器不变，新增历史读数前预测检验和可辨识性反馈，在线仅选六profile/停止；0新飞行，4开环GIF集中replay，不产生事后真值参考路线。实际任务需另行新建独立会话，脚本不伪造模型响应。

- `run_aligned_search.py`：prepare/paired/flights/evaluate/report或all，新建目录拒绝覆盖。16同状态求解、8真实闭环；共同池仍旧J，固定local_pad1池后原语比较旧J/C内部搜索，接受器与GP不变。0模型/新生成。冻结全部结果后派生评价；8GIF集中replay。研究者编写，未证稳定避强。

- `run_reconnection_search.py`：prepare/paired/flights/evaluate/report或all；新建独立目录，16同状态求解/16实际闭环。四组单次/同点分批/多点均分/多点反馈，固定同轮原路线与接回点；C-RRT*分批硬边额度，700追加搜索样本及原总账。所有运行冻结后逐决策重建GP/候选C/能量几何核验再评价，16GIF集中replay。0模型/新生成，不记作LLM收益。

- `run_structure_generation.py`：新的有限结构表达式离线生成研究。prepare/submit/initial/initial_summary/request/revisions/evaluate/report分阶段；真实独立会话提交2初稿+2修订，脚本不召唤/伪造LLM。最多16新配对/16新闭环，四组数值参考明确复用上一轮冻结记录并检查来源；共同GP/C与预算不变。先冻结运行再评价，16GIF集中replay，原始响应/解析/逐步表达式和所有失败分别保存。当前四份均未通过门槛，无胜者；额度已耗尽，不追加新提交或重复试飞。后处理`reports/publish_findings.py`/`verify_evidence.py`只读既有证据。
