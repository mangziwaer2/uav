# 任务排序相容见证：原文选读记录（2026-11-20）

本目录仅为方法设计文献记录，不是实验输出，不含仿真真值、评价标签、拟合或预测运行。原始公开 PDF、来源页面和获取日志均已保存；只对下列段落作了实际选读，不声称通读全文。检索命中、下载成功和 PDF 自动抽取均不单独算阅读证据。

## 来源与实际阅读范围

| 原文 | 来源与保存件 | 本次实际选读 | 支持的有限结论 |
| --- | --- | --- | --- |
| Charles Marx, Flavio Calmon, Berk Ustun (2020), *Predictive Multiplicity in Classification*, ICML, PMLR 119:6765–6774 | [PMLR 原始页面](https://proceedings.mlr.press/v119/marx20a.html)；[原 PDF](predictive_multiplicity_original.pdf) | 摘要；PDF 第 3 页 §2 的 competing models、Definition 1/2；第 4 页 §3.1 的 MIP Solvers；第 5 页 §3.2 的问题 (1) 与 Bounds 段。详见 [选读原文](predictive_multiplicity_selected.txt)。 | 近似相同的经验误差允许冲突预测；该文以误差水平集明确“相容”的范围；有可行解与有全局最优证书的结论不同。 |
| Adam N. Elmachtoub, Paul Grigas, *Smart “Predict, then Optimize”*, arXiv:1710.08005v5，2020-11-19 修订 | [arXiv 来源](https://arxiv.org/abs/1710.08005)；[原 PDF](spo_original.pdf) | PDF 第 2–3 页引言中预测误差与决策误差的区别；第 10–12 页 §3 的动机、Figure 1 说明、Definition 1/2 及并列最优解讨论。详见 [选读原文](spo_selected.txt)。 | 同样的预测误差可引出不同决策/决策损失；真正 SPO 损失需要实际实现的成本向量，不能用同一假设下的自评分代替。 |
| Bryan Wilder, Bistra Dilkina, Milind Tambe, *Melding the Data-Decisions Pipeline: Decision-Focused Learning for Combinatorial Optimization*, AAAI 2019 全文版，arXiv:1809.05504v2，2018-11-21 修订 | [arXiv 来源](https://arxiv.org/abs/1809.05504)；[原 PDF](decision_focused_original.pdf) | PDF 第 1–2 页摘要、引言、Problem description 与式 (1)。详见 [选读原文](decision_focused_selected.txt)。 | 预测准确率不能完整代表优化效用；该文效用由决策在真实但未知参数下的目标值定义。 |

原件 SHA-256：

```text
3f59e8fc11776173fc77f5363227419fc4fe9a53fc22d74e2a3e3244efd1be57  predictive_multiplicity_original.pdf
56381283da01e0a21a861cc8d1bfda97b377095b64051d7d68156b94e848cc3b  spo_original.pdf
7ba5ab2a93ad6fe1d33360c4707696e73d6b3ecfd3240198f0a646fc4460c83b  decision_focused_original.pdf
```

完整 URL、重定向后 URL、时间、HTTP 状态、字节数及原件/抽取件哈希见 [retrieval_manifest.json](retrieval_manifest.json)。本目录其余文件哈希见 `SHA256SUMS`。

## 对本轮见证的推论：存在性与全局界必须分开

以下是本项目的方法推论，不是三篇论文对 UAV、GP 或 LLM 的既有证明。预先定义相容集

`H = {θ ∈ Θ : L_real(θ) ≤ B_real，L_GP(θ) ≤ B_GP}`，

其中参数域 Θ、两个损失上界、路线与到达时刻在调用前冻结。GP 派生约束共享原始读数，不计为额外独立观测。两个损失分开约束可防止用一种误差的下降掩盖另一种误差的上升。容差只是开发设定；未校准时不称置信集、真源覆盖率或某种可信概率。Marx 等的误差水平集只提供结构上的参考，分类误差的容差不能直接移植成 dBm 拟合阈值。

令 `ΔC(θ) = C(route_b; θ) − C(route_a; θ)`，仍使用原 T+Q、各自实际到达时刻及原线性功率积分。若独立检查得到两个可行见证 θ₁、θ₂，且 `ΔC(θ₁) < −τ`、`ΔC(θ₂) > τ`，只能据此证明：在声明的相容集、路线对和数值容差 τ 内，存在相反的成对风险排序。它没有识别真实世界，也没有证明整池最优路线或下一 15 米不同；这些须分别复核。

局部求解获得的最小值 a 和最大值 b 满足 `inf_H ΔC ≤ a`、`b ≤ sup_H ΔC`。所以 `[a,b]` 是已达到的范围，不能冒充包含所有相容结果的外界。若 a、b 异号，存在性见证成立；若它们同号、求解失败或预算耗尽，不能推出整个 H 的符号不变。Marx 等 §3.1–3.2 明确区分可行模型、上下界和二者吻合时的全局证书；本轮有限局部非凸搜索没有其全局证书。

比较对象也应明确：成对排序翻转不自动等于全池 argmin 翻转；全池 argmin 翻转不自动等于立即动作翻转；条件预测代价下降不自动等于实际任务收益。

## 为什么与任务有关，以及为何不证明 LLM 收益

见证目标是路线风险差的极值，真实点测与 GP 形状是相容约束；不是继续最小化历史拟合误差，也不通过改变 T+Q、信任权重或功率尺度制造绕行。SPO Figure 1 与 Wilder 等的目标定义支持关注下游决策的动机，但二者研究的是监督式模型训练，本文工具不是已实现的 SPO/决策导向学习算法，也不据此启动训练。

本轮新见证保持池外：不修改旧世界权重，不继承旧模型的出生后成绩，不把相容性当成更高后验概率。仅替换单个世界进行混合代价敏感性代数时，应标成“固定权重下的条件分析”，不能宣布旧证据已支持该替代世界。

N/I 同信息、同工具、同总预算下，可比较它们是否找到经独立检查的排序冲突、是否提出实际不同的短段、耗用了多少计算。即使 I 更早发现冲突，也仅支持该开发实例下的组织效果；若没有独立的后续预测/实际任务评价，不能称 LLM 任务收益。SPO 的损失定义需要 realized cost，Wilder 等式 (1) 也以真实参数下的效用为目标；用自己改过的预测重新评分自身路线不满足这个要求。

## 检索与未读范围

实际进行了 Crossref 联合关键词检索 `predictive multiplicity decision focused learning`，保存为 [search_crossref.json](search_crossref.json)。前五条结果未直接提供本轮所需基础原文，未把它们作为全文证据。随后依据已知基础文献题名直接访问对应 PMLR/arXiv 页面，核对标题、作者、版本并获取原件；没有把这个直接访问过程记为另一次关键词搜索。

本次没有系统综述，不声称覆盖全部相关工作。未阅读三篇论文的完整实验、全部证明或补充材料；没有复现实验，也没有把其统计一致性或分类问题的全局结论迁移为本项目结论。下载的其余页保留用于原件核验，不算已读。

环境没有 `pdftotext`、`pdfinfo` 或 PDF 读取 Python 包。本次仅在本目录保存并直接导入 `pypdf==5.9.0` 的 wheel 来抽取文本，没有修改项目依赖或活动源码。抽取器提示部分 PDF 对象偏移异常；布局模式另有旋转文字警告，最终采用默认文本抽取。原 PDF 完整保留，抽取公式有缺字/字符映射问题，公式细节以原 PDF 为准。本记录的关键论证依靠可读定义与 Bounds/MIP Solvers 段，不以损坏的公式转录作为算法规范。
