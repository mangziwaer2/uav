# 预测失败反馈与等效场修订：定向阅读

2026-11-19。本轮实际联网检索、下载及指定段落阅读。全文落盘不等于完整精读；没有运行论文算法或证明本项目收益。搜索原始响应、题名核对及错误说明见sources.json与search_results*.json。

1. **Literature Meets Data: A Synergistic Approach to Hypothesis Generation**，arXiv:2410.17309v3。读摘要、§2.2/2.3：数据驱动流程收集错误预测样本，触发新假设生成；HypoRefine把失败数据与文献摘要用于交替细化。原文的奖励是分类准确率与UCB式探索，不能当电磁观测似然，更不能用LLM自评作为物理可信度。项目转化是把已发预测的残差作为失败证据，让LLM选择受限数值修正算子；最终靠未用于修订的未来点测检验。

2. **Online Prediction Under Model Uncertainty via Dynamic Model Averaging: Application to a Cold Rolling Mill**，Raftery、Kárný、Ettler，Technometrics 52(1), 2010，DOI 10.1198/TECH.2009.08104。读摘要与§3.1–3.3选段：参数状态与模型身份均可变化；遗忘/预测似然用于在线更新，静态BMA为特殊情形。当前项目源解释有真实预测偏差，不能先靠遗忘把错误模型权重抬高。仅在参数修正后另行检验非平稳证据是否限制恢复。尚未实现DMA。

3. **Bayesian Online Changepoint Detection**，Adams、MacKay，arXiv:0710.3742。读摘要、§1–2.1：对当前机制持续时间维护分布，递推预测与变化概率。原文分段独立同分布条件不直接适用于运动接收机的功率序列；不能把所有功率上升都解释成源开启。可借鉴“延续/变化都保留”而非预知突现源，当前不替换已有触发规则。

4. **Practical Interpolation for Spectrum Cartography through Local Path Loss Modeling**，Sagari、Greenstein、Trappe，arXiv:1611.03725。读摘要、§III、V-A及VI-A条件：局部最小二乘估计趋势，处理剩余阴影变化；实验主条件包含已知发射机位置及固定传感器布局。可支持“粗趋势＋局部修正”的思路，不能宣称其已解决未知动态多源反演，也不据此增加本项目传感能力。本轮不实现残差GP。

项目第一步及归因对照见../../history/hypothesis_refinement_plan.md。单纯使用LLM反馈循环、贝叶斯模型平均或最小二乘不是本项目独有创新；候选贡献要通过同工具/同信息/同资源的数值与LLM对照证明。
