# 地图连续嵌入接口：尚未训练对齐

更新：2026-11-05。代码src/uav_em/map_learning/embedding_bridge.py提供投影与拼接张量接口；另由llm_bridge.py完成本地Qwen3-1.7B一次无梯度forward。最新已接独立测向及投影训练入口，但本地真实尝试OOM、优化器步数0，见map_sensor_alignment.md及CLOUD_ALIGNMENT_HANDOFF.md。未接正式simulate或Advisor；随机初始化不能称为LLM已理解地图。

## 1. 输入与输出

MapEmbeddingProjector由目标LLM的输入隐藏维度llm_dim构造，encoder_dim默认96、latent_tokens默认32、hidden_dim默认256。地图路径为LayerNorm→Linear(96,256)→GELU→Linear(256,llm_dim)。同一投影器对所有时刻共享参数。

输入为tokens[B,T,32,96]、valid[B,T]、offset_seconds[B,T]。有效时间必须有限、非正且从旧到新严格递增；末帧必须有效、时间偏移为0。无效位可含占位值，投影前屏蔽并在输出再次清零，不能当作已测零干扰。

接口另有小时间MLP，将实际offset_seconds／35映射至llm_dim，加到对应帧32个地图嵌入上。35s只是时间归一化尺度，不改变有效观测定义。时间MLP尚未训练，不能据此宣称时序理解。此处选择连续时间编码，不额外增加帧头位置；后续文字仍须清楚说明地图格式和任务，不能用LLM自己的序号替代实际时间。

输出embeddings[B,T×32,llm_dim]及有效位置mask[B,T×32]；4帧地图占128个位置，缺失位由mask屏蔽，未进一步压缩时间维。LLM的文本、任务、状态和回答仍需其他位置，总长度不等于128。

## 2. 拼接位置

insert_map_embeddings接收已经经过目标LLM词嵌入层的text_embeddings、文字有效位、投影后的地图向量及有效位，在指定文字边界插入地图，返回inputs_embeds、attention_mask、position_ids；可选返回labels。

这不是把向量转成小数文本，也不是把地图向量再送一次文字embedding查表。文字和地图必须显式匹配批量、隐藏维度、dtype与device。position_ids按有效位置累计；地图填充和文字填充被attention_mask屏蔽。地图所有位置的labels=-100，不作为词表预测目标；调用方还必须先把问题／系统提示等文字标签设为-100，仅答案保留监督。

拼接函数不猜测聊天模板边界、不下载tokenizer、不调用generate；具体LLM的模板、最大长度、位置机制、生成缓存与特殊token边界仍需后续实际集成测试。不能把张量单元测试等同真实模型兼容性验收。

## 3. 使用顺序

1. 先确定通过保真核查的编码器权重与哈希，固定它，不混用旧token缓存。
2. 选择具体LLM，由它的词嵌入层确定llm_dim；建立投影器和时间MLP。不要只根据模型名字硬编码维度。
3. 用已知可见区域还原／任务B数据构造问答，先冻结编码器与LLM主体训练投影器／时间MLP；这是后续训练阶段，本轮未执行。
4. 如果仅训练接口不足，再另行考虑少量LLM适配参数并检查文字能力保留；不声称冻结主体就足够。
5. 推理和训练均只能读取允许的地图tokens及公开时间／位置等。导出文件的onboard_target、重建图、标准答案只用于诊断或监督，不能作为压缩输入旁路。

后续需将投影器参数、编码器checkpoint哈希、目标LLM准确标识／权重版本、输入格式／时间尺度一起保存；相同维度不代表语义可互换。目前没有训练出的投影器权重，没有LLM微调／RL；只有随机投影器的真实LLM前向接线记录，没有完整地图回答或理解效果结论。
