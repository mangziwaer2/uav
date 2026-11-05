# 地图区域问答与本地LLM接线

更新：2026-11-05。本文件记录区域问答准备及一次无梯度forward阶段；当时未训练。后续已独立接入测向及冻结LLM的投影训练入口，真实本地尝试显存不足、0优化器步，最新状态见map_sensor_alignment.md及CLOUD_ALIGNMENT_HANDOFF.md。没有LLM主体微调、RL、完整回答验收或在线simulate改动。

## 1. 固定哪版编码器

本轮显式固定原空间v2第39轮作为接口基线：outputs/interference_map_encoder_20261104_174105_494050/best.pt，SHA256为24e8edce45137ae765102ee88a527ba32e0c195e7ed37d13943536454c5ac2f5。不是自动择优或安全验收。

理由是保持输入版本稳定，停止为了单个指标反复改编码器；新的时序候选改善召回／变化格差值，但假变化增加，没有充分理由全面替换。两版都保留原始误差报告。LLM区域还原监督仍从原始机载地图生成，不从有误差的解码图生成，避免把解码误差当标准答案；不能据此消除编码信息损失。现有3dB变化指标是诊断尺度，不作为规则替LLM判源或行动。

后续对齐可先研究区域还原与估计变化理解，把剩余误差作为限制记录；不能宣称当前编码已支持可靠源移动判断。以后替换编码器必须新建数据／缓存版本并重新对齐，不混用旧tokens。

## 2. 正式派生数据

目录：datasets/interference_map_alignment_v1。名称：UAV机载地图编码对齐数据v1。原始数据datasets/interference_onboard_maps_v1保持不变，manifest哈希不变。

- train：16,144道问答，来自84场景、504条episode、25,410帧。
- validation：3,462道问答，来自18场景、108条episode、5,433帧。
- 不导出test问答或轨迹，不打开evaluation_only。读取共享原任务JSONL时按split过滤，不用test标签调参或评价。
- 每道原时序题派生两类：current_region还原当前指定ROI的六通道；temporal_region比较原历史窗口最早有效帧与当前帧。
- ROI来自原数据题目，不是模型自主选择；两类同源题保持原场景组，不拆分到不同split。

文件：manifest.json记录源数据／编码器／问题文件／各token缓存哈希；train.jsonl与validation.jsonl每行一题；tokens/每episode一份缓存；samples/各一条可读题；DATASET_CARD.md说明边界。

每条问答包括id、episode_id、scenario_id、split、kind、inputs、question、answer。
inputs记录token_path、因果frame_indices、valid、实际时间偏移、当时UAV位置、ROI及公开栅格尺寸／分辨率／高度。question中的占位符只标记连续嵌入插入点，不是地图小数文本。
answer独立保存，不进入问题。current_region含supported、power_dbm、uncertainty_db、age_seconds、blocked、building_height_m和含义标记；temporal_region保留原任务B标签。数值保留3位小数，功率来自原机载张量[-120,-30]dBm归一化／截断语义，不是额外真实功率标签。未知功率／年龄为null。

tokens缓存只含独立编码的tokens和时间，没有原图、解码图、答案或隐藏源。虽整条episode缓存有后续时刻，AlignmentExamples.model_input仅按已验证的历史索引取窗，严禁把整条缓存输入当前决策。无效位填零并屏蔽，不是已测零干扰。

## 3. 构建与审核

```bash
python tools/build_map_alignment_dataset.py --dataset datasets/interference_onboard_maps_v1 --checkpoint outputs/interference_map_encoder_20261104_174105_494050/best.pt --output datasets/interference_map_alignment_v1 --device cuda
python tools/audit_map_alignment_dataset.py --dataset datasets/interference_map_alignment_v1 --source datasets/interference_onboard_maps_v1
```

输出目录／报告必须是新的；示例对应现存目录，重跑须换名。生成器校验源文件与原始任务标签，固定编码器无梯度encode；不训练LLM。
审核逐条取因果输入、核对哈希／场景组／完整题覆盖，并由原机载帧重算六通道及变化答案；不是只检查目录存在或条数。报告见数据目录quality_report.json，不替代原数据的物理重放审核，也不证明LLM理解。

## 4. 本地真实模型接线

工具tools/check_map_llm_bridge.py，代码src/uav_em/map_learning/llm_bridge.py。只从本地加载模型／tokenizer，不下载、不启用remote code，不运行优化器。用真实聊天模板关闭thinking；渲染后的单个地图占位符拆成前后文字，分别tokenize，再用连续地图向量替换占位符，避免把坐标矩阵或向量逐数值写成文本。

```bash
python tools/check_map_llm_bridge.py --dataset datasets/interference_map_alignment_v1 --model-path models/Qwen3-1.7B --sample-index 6 --device cuda
```

实际输出：outputs/interference_map_llm_bridge_20261105_122625_826523。样本scene_00003_goal_000_12_current_region；文字316个位置，地图128个有效位置，总输入[1,444,2048]，logits为[1,444,151936]，末位置logits有限。CUDA已分配峰值约3.35GiB，只是该次前向记录，不代表训练显存或通用性能基准。

投影器与时间MLP随机初始化，只有一次无梯度forward；没有生成完整区域JSON。下一token不作正确性评价，也不是决策证据。记录prompt模板／前后文字ID、模型配置／tokenizer／权重文件哈希、编码器哈希、库版本、预算和形状；不把渲染模板中的占位符冒充真实文字地图输入。

这个结果只证明本机这份Qwen3-1.7B权重与当前张量接线兼容。其他LLM、更改模板、生成缓存、长回答teacher-forcing等仍需分别验证。超出max_positions直接拒绝，不截断地图或证据。

## 5. 下一阶段边界

接线成功不等于语义对齐。随后用户已授权并完成答案token监督与提示／地图labels=-100、完整长度预算及冻结检查，训练入口见map_sensor_alignment.md；本地真实尝试显存不足，云端按专门交接继续。因果teacher-forcing的答案只作为assistant监督后缀，不能放入user问题或自由生成输入降低难度，也不能只测复制模板。

先验证指定区域还原／时序任务，再考虑少量LLM适配参数和动态重规划。当前数据不能监督隐藏源坐标或唯一源数，也没有训练LLM自主选ROI。需要源推测时仍需保留带时间点测／测向等证据，而不是假定地图编码包含所有传感器信息。

## 6. 本轮代码检查

2026-11-05新增12项对齐数据／LLM桥接测试全部通过；小型Qwen3检查真实forward及冻结LLM下投影器梯度链路，不执行优化器。全量435通过、2失败（122.41s）；失败仍为既有可选夹具datasets/example_power_grid.json和datasets/example_buildings_local.geojson缺失，未补造或跳过。命令为`python -m pytest -q -p no:cacheprovider --basetemp temp/interference_map_alignment_full_20261105_02`。现存真实1.7B前向记录与单元测试分别记账，不能把单元测试通过当作语义对齐。
