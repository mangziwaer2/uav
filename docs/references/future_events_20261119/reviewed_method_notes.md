# 未来事件方法定向阅读

2026-11-19。来源、下载时间及HTML/文本哈希见sources.json。只核对摘要和下列方法段，没有复现实验或进行系统综述。

- **Transformer Hawkes Process，2002.09291：**§3.1输入为(t_j,k_j)已观测事件时间/类型，时间编码＋类型嵌入、attention；§3.2连续时间条件强度；§3.3似然/事件预测训练。借鉴时间/类型联合预测；我们输入混合功率而非已知事件，需要单独处理隐事件与空间属性，不能直接称原模型适用。
- **Self-Attentive Hawkes Process，1907.07561：**§2事件与历史定义；§3.1 marked temporal point process及条件强度；§4自注意力条件化动态强度。区分事件发生率与电磁场功率；连续值轨迹/源位置并非原事件类型预测自动输出。
- **MOTR，2105.03247：**§3.2持续track query与新生detect query；§3.3 tracklet-aware assignment；§3.4—3.5更新旧轨迹/新生终止。可借鉴保持旧基元表示而非每点重造；其对象可见性、框标签和分配条件不成立于仅标量功率，等效源ID不是物理跟踪ID。

全文方法转化、源组合语义和及时调度设计见../../future_events_and_timely_replanning.md。没有新训练/拟合/飞行/真值查询或实际决策LLM调用。
