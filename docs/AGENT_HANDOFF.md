# 当前交接（2026-11-24）

目标是验证复杂动态电磁环境下，LLM 能否在同信息、同资源下改善 UAV 未飞路线的决策以降低真实全程 `T+Q`。**E2 已修正开发场景的可绕性，但尚未证明 LLM 任务收益。**

## 当前证据

[E3 新试飞双图](../outputs/passive_branch_event_E3_20261124/replay.html) · [结果与异常](../outputs/passive_branch_event_E3_20261124/findings.md)。新开发场景 `43821`，强 N 在第 6 秒改道、第 7 秒持续预警时因原方向变成备选再次复核，实际 `T+Q=73.56`，合规到达。LLM 独立请求人工转交时有一行数据不符，回执未执行；该臂在第 6 秒停止，不重试、不用 N 代选、无配对差值。下一轮先落实公开包原样交付和送达哈希验证，再检查分岔去重。E3 预算已关闭。

[E2 局部干扰双图](../outputs/passive_local_hotspots_E2_20261124/index.html) · [结果](../outputs/passive_local_hotspots_E2_20261124/findings.md) · [已关闭合同](passive_local_hotspots_E2_contract.md)。固定开发场景 `43711` 的预警覆盖峰值为可飞网格点的 15.67%，且存在有绕行成本的低暴露合法路线。两臂第 6 秒同选原路，第 25 秒同选绕行，均合法到达，真实 `T+Q` 同为 354.36；未显示 LLM 增量。第 6 秒的菜单已含低暴露绕行，事后核算好于原路，但有限窗内共同 GP 上尾排序相反。

[E1 入口](../outputs/index.html) · [双图 43621](../outputs/passive_event_choice_E1_20261124/replay_43621.html) · [双图 43641](../outputs/passive_event_choice_E1_20261124/replay_43641.html) · [结果](../outputs/passive_event_choice_E1_20261124/findings.md) · [已关闭合同](passive_event_choice_E1_contract.md)。

- 43621 在 6/9/12 秒咨询，43641 在 1 秒咨询。四份独立回执有效，LLM 均主动明确选择原路并实际执行，无弃权或数值回退。两场都与同菜单强数值 N 同轨到达，配对真实 `T+Q` 分别为 253.00、426.57。
- 两场暴露了不同触发问题：43621 在 28 秒新预警前已用完预算；43641 持续预警未重触发，46 秒再越线时仅有一条合法余程。四次比较均只有 4.5 秒共同支持（6 秒窗）。每次删点重连都生成了新合法候选，但未被选中。咨询时仿真暂停，不能读作实时效果。

## 接手顺序

讨论未知段是否应计入 Q、全图预测或 MLP 是否值得做，先读[研究备忘录：未知路段怎么选](method.md#研究备忘录未知路段怎么选)。其中的选项和待办可随证据修订，不是冻结合同。

**先把 LLM 决策链调通，再做增量对照。** 开发无需每次跑 N；验证增量时主配对为强 N 与 LLM。不限 LLM 调用次数、不继承旧三次复核上限、不让 N 代选；合法回复应实际执行。优先做[持续预警中的新分岔触发和分岔风险证据](method.md#优先改进的两处)，不先换 GP。规则见[下一轮的决策权与预算](method.md#下一轮的决策权与预算)。

1. [分岔证据表](method.md#e1e2-分岔证据表)已完成。E2 第 6 秒两路立刻分开，线性期望 Q 已翻序；后来实测只验证原路预测偏低。E1 43621 在预算耗尽后出现了不同的合法走法；43641 在持续预警期间出现新首分岔，等到再次越线却已只剩原路。未触发帧只核几何和资源，风险可比性仍未知。不在已见场调参重跑，不把固定秒数搬到别的场景。
2. 将“场景本身缺少可绕走廊”“走廊存在、但候选生成漏掉”“菜单有路、但在线证据不足”分开。双图看起来接近全图覆盖只是观察，不是不可绕的证明。后续独立合同预先定义场景分层、判据和种子，保留有绕行代价和困难场；真场只供独立事后诊断。
3. **新动作列入计划：允许 LLM 在同一次必选回执中指定一个图外中间目标点。** 共同规划器从当前点经该点补全到终点并核硬约束。不能合法补全才用 LLM 同时指定的合法路线 ID；不可比或数值得分不占优不构成拒绝。强 N 同样可使用图外点算子，细节见[方法](method.md#计划图外中间目标)。
4. [复现顺序和原始文献](method.md#优先复现的已有方法)：先做多通道几何生成和删段修复的强 N，再检验 LLM 调度同一组算子。新的证据表示、候选生成、实时触发和 LLM 效应分别比较；先冻结独立合同、公开输入、强 N、总资源及停止条件，再开新科学运行或独立 LLM 会话。E2、E1 及旧预算均已关闭，图外动作未实现。

## 实现与入口

- [`passive_branch_event.py`](../src/uav_em/passive_branch_event.py) / [`passive_branch_event_pilot.py`](../src/uav_em/passive_branch_event_pilot.py)：E3 持续预警中的新分岔触发、可比证据分解及独立回执执行；不能用已关闭合同重新起飞。
- [`passive_local_hotspots_pilot.py`](../src/uav_em/passive_local_hotspots_pilot.py)：E2 局部场、完整剩余 T 加共同窗 Q 的同菜单强 N、冻结与配对执行。
- [`passive_event_choice.py`](../src/uav_em/passive_event_choice.py)：风险预警、路线菜单、窗口评分。
- [`passive_event_choice_pilot.py`](../src/uav_em/passive_event_choice_pilot.py)：E1 冻结、必选回执与配对执行。
- [`passive_decision_replay.py`](../src/uav_em/passive_decision_replay.py) / [`passive_decision_replay.js`](../src/uav_em/passive_decision_replay.js)：P7 样式双图和决策分岔。沿用左真场右公开预测的布局；不以较简回放替代。
- [`passive_route_actions.py`](../src/uav_em/passive_route_actions.py)：旧顶点 ID 接口，只映射既有路径，不能创建图外路线。E1 仍复用一些 P1–P7 辅助模块，不按实验名删源码或测试。

先读[维护约定](../AGENTS.md)和[方法](method.md)，按 [README](../README.md)验证。旧证据见[归档入口](../outputs/research_archive/passive_pre_e1/index.html)，不照旧命令重跑。检查器保留 21 项已知历史缺失；这是历史原件问题，不等于当前 E1 链接或测试失败。不读隐藏 `fields` 或封存场景。
