# 本轮针对性阅读记录

日期：2026-11-18。以下行号对应同目录下载文本，不是PDF页码。短引文只标示核对锚点；中文解释是本项目判断，不是原文承诺。完整综述/创新性检索尚未完成。

## deep_completion

- 阅读位置：III-B; III-E1/2/3; equations (9), (10)。
- 下载文本行号：[(2287, 2325), (4740, 4845), (4998, 5034)]。
- 短引文：“This would clearly render the estimator useless.”
- 用途和限制：只拟合输入位置可能退化；论文用测量掩码、受限表示，以及输入/目标子集和仿真教师训练。子集在原文不要求必然不相交；本项目建议更严格的因果/空间留出，不把改造冒充原文结论。

## attentive_neural_processes

- 阅读位置：Sections 2 and 3; latent path and context consistency discussion。
- 下载文本行号：[(326, 345), (445, 452), (547, 568), (830, 845)]。
- 短引文：“each sample of” / “would correspond to one realisation of the stochastic process.”
- 用途和限制：全局隐变量生成连贯函数样本，注意力利用相关观测；论文也讨论条件一致性的局限。等效基元解码和动态校准尚需本项目验证。

## task_based_learning

- 阅读位置：Abstract; Introduction; differentiation through optimization discussion。
- 下载文本行号：[(45, 60), (87, 123), (860, 875)]。
- 短引文：“the criteria by which we train these algorithms often differ from the ultimate criteria on which we evaluate them.”
- 用途和限制：任务损失与预测损失可能不一致；本文不由此声称RRT*可微或本项目已实现端到端训练。

## sayplan

- 阅读位置：Publisher abstract and bibliographic metadata only。
- 下载文本行号：仅出版社页面。
- 短引文：“reduce the planning horizon for the LLM by integrating a classical path planner”
- 用途和限制：仅核对出版社摘要：任务相关搜索、经典路径规划器与反馈修订。arXiv全文HTML404，未称全文精读。

## code_as_policies

- 阅读位置：Section III and Section V Discussion and Limitations。
- 下载文本行号：[(150, 182), (1036, 1065)]。
- 短引文：“This fits into systems with factorized perception and control”
- 用途和限制：用代码组合API和反馈；可用感知/控制接口限制能力，不能事先保证响应正确。本项目只建议受限计划/AST。

## event_triggered_mpc

- 阅读位置：Abstract; Introduction contributions; discussion following event mechanism。
- 下载文本行号：[(18, 28), (196, 201), (1448, 1455)]。
- 短引文：“the average of errors between two consecutive event-triggered instants is taken into consideration”
- 用途和限制：累计预测误差可用于事件触发设计；该动力系统中的稳定性/间隔证明不能转为本项目保证。

## inner_monologue

- 阅读位置：Introduction; Section III feedback formulation。
- 下载文本行号：[(113, 124), (364, 386)]。
- 短引文：“continually injecting information from the various sources of feedback”
- 用途和限制：执行/环境反馈可帮助LLM修订计划；不是新增测量能力或电磁推断保证。

