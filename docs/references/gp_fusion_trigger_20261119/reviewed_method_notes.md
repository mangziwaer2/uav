# GP组合与路线影响触发：检索及阅读记录

日期：2026-11-19。检索原始返回在3份`*_search.xml`，查询及摘要结构化记录在[search_results.json](search_results.json)。搜索结果不等于正文证据；没有把未下载条目写成已读。

下载URL、实际地址、原文件字节数和SHA256见[downloads.json](downloads.json)。下载了2篇arXiv完整HTML及GPML第2章PDF；HTML旁的txt为标准库解析生成的阅读副本，数学排版以HTML为准。未安装新依赖。

## 1. UAV群事件触发DMPC

Alexander Gräfe, Joram Eickhoff, Sebastian Trimpe. **Event-triggered and distributed model predictive control for guaranteed collision avoidance in UAV swarms**. arXiv:2206.11020v1，2022；正文标注被IFAC NecSys 2022接受。

- 来源：https://arxiv.org/abs/2206.11020v1
- [完整HTML](uav_event_mpc.html) / [提取文本](uav_event_mpc.txt)
- 本次实际选读：摘要、引言贡献、§4末尾Theorem 2之后的事件触发独立性说明及“Priority based trigger rule”，式(9)和后续变量说明；未复核完整证明或复现数值。
- 原文做法：有限计算节点每轮只重规划部分UAV；优先级结合距目标距离、距上次规划的时间、目标方向锥内其他UAV的位置。该例的拥堵项用于降低不容易取得进展的重规划优先级，不应误读成我们的干扰风险阈值。
- 借鉴：触发可依据当前轨迹与任务相关性/有限计算资源，而不必对每个状态变化均完整重算。
- 边界：碰撞保证依赖线性动力学、初始可行轨迹及后备轨迹构造；正文明确Theorem 2不依赖所选ET。不能把原文碰撞保证归给优先级公式，更不能转移到未知电磁场。本文不是单UAV电磁暴露触发器，式(9)不直接移植。

## 2. GP学习动力学的事件触发MPC

Yuga Onoue, Kazumune Hashimoto, Akifumi Wachi. **Learning-based Event-triggered MPC with Gaussian processes under terminal constraints**. arXiv:2110.12214v3，版本日期2024-01-01。

- 来源：https://arxiv.org/abs/2110.12214v3
- [完整HTML](gp_event_mpc.html) / [提取文本](gp_event_mpc.txt)
- 本次实际选读：摘要、引言方法概述；§III-C式(19)触发条件及式(21)–(23)后对阈值/递归可行性的说明；结论关于RKHS和误差界的条件。未逐行验证§IV证明、附录或仿真。
- 原文做法：GP学习未知系统动力学，预测状态与实测状态的距离超出规划相关阈值时重解控制优化；阈值由保持递归可行的条件推导，相关辅助优化为几何规划。
- 借鉴：把“新观测是否使原计划依据失效”作为更新线索，而非仅看测量值本身上升/下降。
- 边界：我们GP估计外部干扰场，不是状态动力学；没有采用其终端集、RKHS误差界和控制证明，不能宣称继承收敛/安全保证或照抄阈值。本文不直接推出我们的T+Q收益触发式，那是本项目待验证的工程转化。

## 3. GPML第2章：仅保存，未完成审读

Carl Edward Rasmussen, Christopher K. I. Williams. **Gaussian Processes for Machine Learning**, Chapter 2: Regression，MIT Press，2006。

- 下载：https://gaussianprocess.org/gpml/chapters/RW2.pdf
- [PDF原文](gpml_ch2.pdf)
- 本次状态：已下载PDF，字节以`%PDF`开头；本机没有pdftotext，正文尚未提取/选读。不能把该下载写成已全文审读，当前设计不把它作为已核对公式来源。

## 4. 复用此前已选读的无线地图方法

Shweta Sagari, Larry Greenstein, Wade Trappe. **Practical Interpolation for Spectrum Cartography through Local Path Loss Modeling**. arXiv:1611.03725v1，2016。

- 来源：https://arxiv.org/abs/1611.03725
- [先前保存的完整HTML](../hypothesis_refinement_20261119/radio_local_pathloss.html)
- 先前实际选读范围：§III、V-A及VI-A实验条件；原哈希和阅读记录在同目录。此次重新核对引言的单源/已知位置起始条件及附录A趋势＋残余估计表达；未复核全部推导。
- 原方法用路径损耗趋势和空间残余估计接收功率，支持把结构模型与局部数据修正结合。主实验含固定传感器和已知源位置条件，不能当未知时变多源定位可辨识的依据。
- 本项目中的“GP软形状正则”和“局部GP/等效场平滑预测组合”是明确标注的拟议实现，不能称原论文已给出相同算法或证明本任务收益。

## 本次失败与未做事项

- 第一次解析尝试因当前虚拟环境缺少bs4失败；后改标准库HTMLParser下载/提取，无安装。
- PDF提取尝试因未安装pdftotext失败，保留原PDF并标注未读。
- 搜索命中中的机会约束通信、在线GP更新、计算延迟和蒸馏事件模型等未选读全文，不作为具体算法证据。
- 没有新GP预测、拟合、飞行、决策LLM、真值查询、训练或RL。方法安排见[设计](../../gp_equivalent_trigger_design.md)，不在文献目录另开实验线。
