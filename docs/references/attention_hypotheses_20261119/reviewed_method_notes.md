# 定向方法阅读记录

2026-11-19。5篇新全文HTML已下载，来源/时间/哈希见sources.json；阅读范围是摘要及下列指定方法段，不冒充系统综述或实验复现。

- **Set Transformer，1810.00825：**§3.1 ISAB用诱导点降低集合attention复杂度；§3.2 PMA用k个seed输出k个相关向量，后接SAB建模输出相关；§5.3为混合高斯摊销聚类。用于观测集合到基元集合的结构参考，不能把总功率点测误当互斥源簇。
- **Slot Attention，2006.15055：**§2.1随机同分布初始化槽、竞争注意力、多轮更新；§2.2图像解码RGB加alpha，alpha跨槽softmax；§2.3集合预测处理槽交换。借鉴可交换参数槽；不直接借用独占观测分配或alpha物理功率合成，不把随机槽初始化称已校准多环境后验。
- **BayesSim，1906.01728：**§IV-A定义模拟器参数、仿真轨迹和观测；§IV-B—E条件混合密度网络与后验恢复；§IV-F轨迹统计量。支持离线仿真到条件参数分布的摊销思路；等效场参数与物理模拟器参数并非同一监督对象。
- **The frontier of simulation-based inference，1911.01429：**§1.2推断，§3.1—3.4直接模拟/代理/前后处理/建议，核对工作流而非只从“有模拟器”推导后验自动可靠。本文用作方法背景，不据此声称当前高斯基元具备正确似然。
- **ReAct，2210.03629：**§2交错推理、动作和反馈；工具帮助修订计划。借鉴LLM组织拟合和路径算子；当前工具反馈不是新的物理观测，原文任务也不是功率源反演。

复用上轮原文并重新查看指定段：

- **ANP，1901.05761：**../llm_equivalent_review_20261118/attentive_neural_processes.html。§2.1全局z对应一个函数实现，正文指出context consistency限制；§3 context self-attention和target cross-attention。每个场景共享z，不能每个路线点独立换一个世界；普通条件网络不自动得到顺序贝叶斯一致性。
- **Task-based learning，1703.04529：**上轮task_based_learning原文；这里只复用已核对的任务目标对齐思想，不新增全文精读声明。
- **无线地图自编码、事件触发MPC：**依据和原文仍见../llm_equivalent_review_20261118/与../../llm_equivalent_research_review.md；具体控制稳定性/触发阈值不能直接移植为本系统保证。

研究转化与现有代码的边界见../../hypothesis_pool_attention_review.md。未新增训练/拟合/LLM决策/飞行/真值评价。
