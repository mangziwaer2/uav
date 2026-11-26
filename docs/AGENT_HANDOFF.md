# 交接 · 2026-11-26

先读本页与[方法](method.md)，约束见[AGENTS](../AGENTS.md)。本轮只压缩文档、清目录，未改算法、试飞或调用实验模型。旧稿已[压缩归档](../outputs/research_archive/handoff_compaction_20261126.zip)，按需查看，不沿用其中“下一步”。

## 当前结论

主线是 **LLM 调用现有分岔修复工具并调度搜索**；完整余程优化、等效源预测尚未接入飞行。RP3 已关闭，三个预抽随机场六臂均合法到达，平均代价上升，未证明稳定 LLM 增量。

| [同期回放](../outputs/passive_repair_RP3_20261126/index.html) | N T+Q | LLM T+Q | LLM−N |
|---|---:|---:|---:|
| 单侧建筑 | 35.17 | 33.62 | −1.55 |
| 横跨单楼 | 43.84 | 18.39 | −25.45 |
| 错列双楼 | 21.54 | 74.79 | +53.25 |

平均差 **+8.75**。详情只看[本批诊断](../outputs/passive_repair_RP3_20261126/findings.md)与[审计](../outputs/passive_repair_RP3_20261126/audit.json)。第三场暴露两个问题：

- t=3 相同观测下，N 扫描16个引导点找到南侧大绕行；LLM 只试三个较小偏移，遗漏更有利报价。主要是搜索覆盖问题。
- t=13 前视重新越线却被 `persistent` 抑制复核；只缓解一个点，未完成两点重新布防。主要 Q 已在3–12秒累积，不能把全部损失归于此。

## 接下来按顺序做

1. **P1：静止修预警状态。** 覆盖短暂缓解后重新越线、持续高风险不重复复核、未知支持；从 `passive_repair_alert.py` 和 `test_passive_repair_feedback.py` 开始。双方共用，收益不计为LLM增量。
2. **P2：搜索覆盖与对照。** 保留完整 N，双方共享公开走廊建议、参数域和资源；增加读取同样反馈的数值自适应调度，再比较LLM的细化、换方向和停止。不要先认定单引导点不够用。
3. **P3：新合同后再试飞。** 用全新随机场，冻结公开输入、强N、版本、总资源和停止条件；全场保留并交付同期双图，按合法到达与全程T+Q评价。当前批次不补跑。

并行备选见[方法](method.md)：先校核积分再做完整余程优化；等效源先验证未来预测，再接规划。LLM改算子源码仍后置。

## 接手入口

- 工具：`src/uav_em/passive_repair_tools.py`；生成域：`passive_repair_geometry.py`；完整N：`passive_repair_tool_policy.py`。
- 执行/回放：`passive_repair_flight.py`、`passive_repair_pair.py`、`passive_repair_replay.py`；RP3入口复用RP2同步接力驱动，均在同一源码目录。
- [LLM实际配置](method.md#llm配置) · [RP3合同](passive_repair_RP3_contract.md) · [冻结清单](../outputs/passive_repair_RP3_20261126/manifest.json) · [全部回放](../outputs/index.html)。
- 最新全量 **918 passed**，见[原日志](../outputs/passive_repair_RP3_20261126/postflight_tests.log)。维护检查保留 **21个历史缺失**，不能改成全绿；命令见[README](../README.md)。

RP1–RP3、EP1、MC1及更早批次均已关闭。RP2曾因接收器误拒 `final_answer` 停飞，修正后另立RP3；[停飞原件](../outputs/passive_repair_RP2_20261126/RP2-01/flight/replay.html)保留。EP1无模型通道，MC1因旧网关额度不足零有效回答；恢复旧网关、旧校准及RC1传输验收均不是当前门槛。不要读封存场或配置隐藏 `fields`，不自动提交/推送。
