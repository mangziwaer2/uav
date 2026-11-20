# LLM辅助决策优化：定向原文选读

日期：2026-11-20。范围是“有限点测→GP辅助多等效场→LLM组织工具检验与路径搜索”，本轮仅文献研究，没有决策模型实验、飞行、点测、训练或评价真值查询。未读取outputs或隐藏场景。结论是设计候选，不是已经获得UAV收益。

建议优先借用 **LLM-Modulo的提案—外部批评—有限修订**，把ReAct的工具反馈接进该流程；任务相关上下文和共享数值工具应先准备完整。ToT/LATS只借小规模分支与回溯结构，限于可复制的冻结预测状态。完整方法判断见[reviewed_method_notes.md](reviewed_method_notes.md)。

本轮实际选读5篇原始论文的方法/局限段落：LLM-Modulo立场论文、其后续实证论文、ReAct、LATS、ToT；不是5篇全文通读。另核对SayPlan出版社摘要，并选读既存ReEvo原文用于说明与离线启发式演化的区别。出处、版本、原件/文本哈希和精确文本行范围见[reading_manifest.json](reading_manifest.json)。ReAct、ReEvo、SayPlan摘要引用既存缓存，未修改或重复下载。

新下载原件包括4份PDF和对应arXiv摘要页；PDF使用既存`../task_contrast_20261120/pypdf-5.9.0-py3-none-any.whl`提取，未安装依赖。提取复现脚本为[extract_pdf.py](extract_pdf.py)。文本内`=== PDF PAGE n ===`是所存PDF的页序，不是期刊页号。

检索/获取的失败没有当成全文证据：

- 首轮`urllib(timeout=20)`只限制socket，不保证整体请求20秒内结束；两个SayPlan请求持续过久，已终止。其他成功原件先落盘而响应元数据未及时持久化，因此`retrieval.json`明确保留“HTTP状态/最终URL未留存”，不补造200或重定向版本。
- 之后使用curl的20秒总网络时限；SayPlan出版社/arXiv全文都因15MB文件传输未完成而超时。部分字节已丢弃，失败与字节数留在`retrieval_bounded.json`。
- 最后对出版社PDF使用58个分别限20秒的HTTP Range请求，范围字节均收到，但ETag出现两种值。未将其拼接为版本一致原件，未读取拼接文本；范围头/哈希与放弃原因保留在`sayplan_ranges_retrieval.json`和`sayplan_assembly_failure.json`，片段负载已丢弃。SayPlan仍仅摘要级阅读，不称全文阅读。

`retrieve.py`、`retrieve_additional.py`保留最初获取过程，存在上述总时限局限，不建议直接重跑；`retrieve_bounded.py`记录后续总时限做法。搜索命中`search_modulo.xml`只是元数据，只有下载并实际选读的原文进入方法证据。
