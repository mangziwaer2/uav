# 云端接手：地图＋测向投影训练

更新：2026-11-05。本地已实际触及显存边界，停止真实GPU重试。新agent先读AGENTS.md、AGENT_HANDOFF.md、current_roadmap.md及CLOUD_HANDOFF.md；详细输入／训练协议见map_sensor_alignment.md。当前只推进冻结LLM主体的投影／时间层训练，不自动启动LLM主体微调、RL、源定位评价或在线代码执行。

## 1. 当前位置

- 原空间v2第39轮、每帧32×96编码保持冻结，4帧投影成128个连续地图位置。时序候选保留作对照，不替换缓存。
- 独立测向已接入：最近60秒全部已到点测／角峰及误差信息以紧凑文本输入，不给源坐标，不采未来，不把无检测当安全。地图编码器无需重训。
- 问答train 16,144条、validation 3,462条，当前指定ROI还原／同区域变化；没有源坐标、真实运动标签或自主ROI选择监督。
- 完整长度审核全部通过，最长7,885位置，包含问题、地图、完整答案和EOS；预算8,192，没有截断。
- 训练入口、因果teacher-forcing屏蔽、冻结检查、保存／恢复已实现。小型真实Qwen3测试证实梯度能到空间投影和时间层，LLM参数不变；不以此证明1.7B理解了地图。

## 2. 本地真实失败现场

目录：`outputs/interference_map_alignment_training_20261105_135229_972149/`。

RTX 3060 Laptop GPU、6GB，torch2.14.1+cu126、transformers5.20.0，Qwen3-1.7B BF16，microbatch=1、累积8、SDPA、非重入梯度检查点。真实训练出现CUDA显存不足并退出，尚无优化器更新；last.pt实查optimizer_steps=0、sample_cursor=0，history.jsonl为空。该恢复点仅初始化，不是训练成果。

当时异常为torch.AcceleratorError而非旧捕获器预期的torch.cuda.OutOfMemoryError，所以该次退出码1，没有自动report.json／oom.json。POSTMORTEM.md及terminal_failure.txt是事后说明和终端摘录，不冒充运行时自动报告；精确失败microbatch、训练峰值没有记录，不推测。first_training_example.json仅是首题。

最新代码已补充严格的CUDA OOM异常分类并加入测试；其他CUDA错误仍抛出。又修正跨BF16／FP16恢复时空scaler状态处理。未在本机重新进行真实训练，原run_config及checkpoint未重写，源码哈希与运行时版本差异如实保留。下一次OOM会写详细诊断并退出码2，不反复重试、截断、CPU回退或自动换模型。

## 3. 需要单独传输的内容

源码与文档仍未commit／push，先审查实际工作区后再按用户要求发布；不能假设GitHub已含新增代码。权重、数据和普通输出均被Git忽略，clone不等于收齐以下文件。

| 内容 | 用途 |
|---|---|
| datasets/interference_map_alignment_v1/完整目录 | 19,606条问答、冻结tokens、manifest及审核报告 |
| datasets/interference_sensor_evidence_v1/完整目录 | 独立观测、manifest、样本、质量报告 |
| models/Qwen3-1.7B/完整目录 | config、tokenizer、词表、权重分片及索引；云端位置可不同，文件内容须相同 |
| 上述本地失败目录 | last.pt、run_config、token_budget、首题、初始化层统计及事后说明 |
| src、tools、configs、tests、docs、requirements等源码 | 尤其sensor_evidence.py、alignment_training.py、两份新工具和configs/map_alignment.json |

只继续训练时不需要原始机载数据或地图编码器权重；如果要重建／完整重审核，另传datasets/interference_onboard_maps_v1/和outputs/interference_map_encoder_20261104_174105_494050/best.pt。原始evaluation_only不得进入训练或模型输入。不要搬Windows解释器、缓存或第三方原始资源，不删除本地原件。

关键SHA256：

```text
map alignment manifest: 7c8adb4540c97e15b0bbdf5e8d21068953c4bc15f2ba84358d26190662ec3ff1
sensor manifest:        70644ce00a836a97e3c44d87b81fbc125944ed9aec3fa85e08c5602bfa7687fb
last.pt (step 0):       9459881e51e8a1ea719c06c97f89aae13aa757894ae244b65fc4cf5067394adc
```

模型文件哈希在原run_config.identity.model_files_sha256内，恢复时自动逐项核对。数据需保持原始字节，不重排JSON或改变换行以免哈希改变。

## 4. 云端启动

使用独立CUDA可用环境，不混用本机Windows toolkit路径。先检查现成PyTorch与驱动，按服务商条件安装合适CUDA版，不盲目重装已可用环境：

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory, torch.cuda.is_bf16_supported())"
python -m pip install -r requirements.txt -r requirements-map.txt -r requirements-llm.txt
python -m pip check
python -m pytest -q -p no:cacheprovider
```

仍有两个既有可选夹具缺失失败，见主交接；不补造或跳过。依赖是范围而非锁文件，本机版本只作复现线索。先记录云端实际环境；Tesla型号与容量必须实查，不能只凭名称判断支持BF16或能容纳全部长样本。microbatch已经为1，累积不会解决单条样本OOM，不承诺16GB卡一定够。

恢复训练，所有命令从项目根目录执行；按实际位置替换LLM／checkpoint路径，默认新建输出：

```bash
python tools/train_map_alignment.py --dataset datasets/interference_map_alignment_v1 --evidence-dataset datasets/interference_sensor_evidence_v1 --model-path /path/to/Qwen3-1.7B --config configs/map_alignment.json --device cuda --resume /path/to/interference_map_alignment_training_20261105_135229_972149/last.pt
```

如果不带last.pt，可去掉--resume，从头初始化；必须明确记为新训练，不声称续上已训练权重。需要先重查长度时加--budget-only，会创建独立输出且不加载LLM训练。默认训练先完整审核长度，可能花数分钟，不是挂起。

配置epochs=1、gradient_accumulation=8、learning_rate=0.0002、max_positions=8192、history_seconds=60、checkpoint_every=25、validation_examples=16、max_steps=null。恢复可调整max_steps、保存间隔和validation样本数；其余设置和数据／模型身份必须一致。FP16会自动启用scaler；从本次BF16空scaler状态迁移会初始化新scaler，不保证跨硬件逐位复现。

新agent先检查第一批能完成forward、backward和优化器更新，再观察更长题；不要只看较短首题或step数就宣布全量显存适配。若云端仍OOM，保存新诊断并报告，不默默删历史、截答案、读取隐藏真值或换成已知源坐标。后续是否采用等价显存优化或更大显存设备根据现场判断，不将梯度累积当作单样本内存方案。

## 5. 完成标准与后续边界

- optimizer_steps须大于0；完整epoch还须full_training_schedule_completed=true，step_limit_reached只表示短训练上限。
- trainable_parameters中LLM可训练参数应为0，只训练1,078,208个投影／时间参数；地图编码器不参与训练。检查loss／梯度有限、checkpoint可恢复、数据与模型哈希正确。
- validation.json只是固定样本的teacher-forced loss，没有自由生成正确性结论。后续需在未见场景验证完整区域JSON、未知mask、实际数值和时序变化，再检验测向证据是否被利用。
- 当前题目不能监督唯一源位置、源数或移动；LLM可能忽略测向，不能因增加通道就声称已学会定位。源假设与补测／重规划能力仍需后续阶段研究。
- 未接正式simulate，不自动开启RL、LLM主体微调、代码生成、门控或更复杂传播；继续遵循current_roadmap。
