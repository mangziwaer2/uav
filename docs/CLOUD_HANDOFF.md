# 云端迁移与接手

更新：2026-11-05。本文件按当前目录重写，不沿用已经缺失的旧日志／目录清单。研究路线只在`current_roadmap.md`维护，先读`AGENT_HANDOFF.md`和`first_version.md`。

**当前优先：** 本地已接入独立测向并实际尝试冻结LLM的投影训练，6GB显存不足、完成优化器步数0，已停止本机真实重试。专门交接见`CLOUD_ALIGNMENT_HANDOFF.md`，训练／证据协议见`map_sensor_alignment.md`；按该文迁移两个派生数据目录、相同LLM及step=0初始化恢复点。不要把下方较早444位置无梯度接线当长答案训练显存验收。最新全量测试448通过、2个既有缺失夹具失败（122.11s）。

## 1. 当前交付边界

- 主环境是普通建筑＋本地时变干扰源＋有限一次反射；主配置为`configs/demo.json`与`demo_25d.json`，不需要真实地图、外部电磁图、Sionna或GPU生成环境。
- GPU用于真实LLM运行及可选小地图编解码器训练。当前模型接口只加载本地目录，不是远程API客户端；云端直接运行仓库，不等于本机已能调用云端推理服务。
- 当前含工作区整理与独立地图学习新增代码，未提交或推送Git。用户需要先核对未提交更改、文件删除及忽略规则，再提交／上传；不能假定云端已收到本轮代码或忽略的本地冒烟数据。
- 旧三份精选日志和此前RT输出当前缺失；不要依赖旧文档路径、恢复成“现存证据”或将其作为新环境结果。

## 2. 迁移内容

**带走：** 根README／AGENTS、requirements文件、pytest.ini、src、tools、configs、tests、docs、temp中指定调试入口及md，以及明确选定的当前数值验收记录。以Git实际候选清单为准，不复制整个工作区。

**单独传输（需要时）：** 完整模型权重目录。模型需有config、tokenizer及权重／分片索引，不是只传一个文件；放置位置由云端使用者决定，启动时指定--model-path。

**不需要／不默认上传：** RadioMapSeer及ZIP、其他原始数据、temp中的缓存和Windows虚拟环境、.idea、原始提交文件、可重建GIF及其他输出。主配置不依赖这些内容。不要为了瘦身删除本地原件。

当前实际输出以`../outputs/README.md`为准；2026-11-04复查时旧none记录及上午地图冒烟输出已缺失，不依赖这些历史路径。正式首版地图数据／编码器也被Git忽略，需要单独传输，不是拉取源码后自动存在。

正式地图数据统一在`datasets/interference_onboard_maps_v1/`，单独传输整个目录（含DATASET_CARD、manifest、episodes、temporal_tasks、quality_report及仅供离线复现的evaluation_only），核对报告的manifest哈希；训练和LLM适配不得读evaluation_only。目录搬迁不改数据版本和划分，旧运行元数据中的原路径不重写。当前推荐`configs/map_encoding_v2.json`，方法／结果见map_codec_v2；先`--require-coverage`审核，再`--require-quality`训练。v1配置仍保留供基线复现，不能把其旧权重按v2解释。checkpoint只保存模型，不支持恢复优化器；云端重训另开输出目录。当前仍未授权执行LLM微调或RL。

时序核查已完成，见map_temporal_codec：固定v2对1,731道validation题检查，支持增失较好，但功率变化存在较多误报；随后已完成下述时序候选微调，不直接以单帧IoU作为LLM适配达标。新工具audit_map_temporal_codec默认不检查test。新历史窗口导出window_schema_version=2增加每个有效位的重建；valid、实际时间偏移及权重哈希不可丢失，也不能将诊断原图作为LLM压缩旁路。

2026-11-05已完成一版时序约束微调，配方map_encoding_temporal.json、入口train_map_temporal_autoencoder；12轮选第8轮，权重在outputs/interference_map_encoder_temporal_20261105_114745_542222/best.pt，需单独传输。变化格差值及召回改善，但假变化增加，详见map_temporal_training；候选不自动替代v2，未检查test。连续嵌入接口见map_embedding_bridge：4帧128位置＋实际时间，插入文字词嵌入之后。本轮已完成本地Qwen3-1.7B一次冻结无梯度前向，尚无投影器对齐、LLM训练或完整生成验收。

### 地图对齐数据与接线

派生数据`datasets/interference_map_alignment_v1/`已完整生成并逐题审核：train 16,144题、validation 3,462题；原场景划分不变，不导出test。两类题是当前指定ROI六通道还原和同ROI时序比较，答案来自当时原始机载图而不是解码重建。详见`map_alignment_data.md`。原始与派生数据均被Git忽略，云端需单独传输两个完整目录，并携带原v2权重`outputs/interference_map_encoder_20261104_174105_494050/best.pt`；该权重仅固定为本轮接口基线，不声称时序误报已解决。派生manifest记录权重／源数据／问答／缓存哈希，不能混用时序候选tokens。

云端安装地图和LLM可选依赖后，先复核数据；若选择重新导出，必须指定不存在的新目录，不能覆盖现存数据：

```bash
python -m pip install -r requirements-map.txt -r requirements-llm.txt
mkdir outputs/interference_alignment_cloud_audit
python tools/audit_map_alignment_dataset.py --dataset datasets/interference_map_alignment_v1 --source datasets/interference_onboard_maps_v1 --output outputs/interference_alignment_cloud_audit/quality_report.json
python tools/check_map_llm_bridge.py --dataset datasets/interference_map_alignment_v1 --model-path /path/to/Qwen3-1.7B --sample-index 6 --device cuda
```

审核报告目标也须不存在。重建命令见`map_alignment_data.md`，需要源数据和相同哈希的v2权重；只迁移派生数据时可检查前向，但无法重算原图监督审核。读取器严格按frame_indices和valid抽取因果窗口，绝不能把episode缓存的后续帧或answer拼入模型输入。

较早接线记录`outputs/interference_map_llm_bridge_20261105_122625_826523/`：316文字＋128地图＝444位置，inputs_embeds为[1,444,2048]，末位置logits有限；随机投影／时间层，0优化器步，不代表LLM理解地图。torch2.14.1+cu126、transformers5.20.0，该次只核查forward。随后已完成独立测向、完整监督／预算和冻结训练入口，实际长输入训练遇本地OOM、未更新参数；云端以本文件开头专门交接为准，不自动启动LLM主体微调或RL。

## 3. 创建云端环境

在克隆／解包后的项目根目录，使用独立Python>=3.10环境。服务商已有合适环境时可直接使用，但所有命令必须来自同一解释器。

```bash
python --version
python -m pip --version
python -m pip install -r requirements.txt
python -m pip check
python -m pytest -q -p no:cacheprovider
python temp/debug_full_pipeline.py --config configs/demo_25d.json --advisor none --plot --gif
```

requirements是范围约束，不是锁文件；先记录实际安装版本。可用`python -m pip freeze`另存于新的实验目录。不要照抄本机E盘路径、从其他环境挂接site-packages或复制Windows环境；根environment.yml不再维护。

已知测试缺口是两个缺失外部夹具：`datasets/example_power_grid.json`和`datasets/example_buildings_local.geojson`。如果迁移副本同样缺失，应如实保留对应失败，不当作主仿真失效，也不自动下载不明替代或跳过测试。新的其他失败应单独调查。

## 4. 启用真实模型前

先按云端驱动／CUDA条件安装或核查PyTorch，不盲目替换服务商可用的GPU环境。再处理可选requirements-llm约束，满足已有版本时无需重装：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
python -m pip install -r requirements-llm.txt
python -m pip check
```

基础仿真不需要torch。可选依赖范围包含当前查询到的transformers5.x，但这不是新环境真实LLM运行成功保证；先做短接口检查，并把实际torch／transformers／CUDA／精度记录在实验元数据中。不要把auto选择CPU当作GPU测速，正式核查使用--model-device cuda。

实际权重路径示例（替换成云端位置）：

```bash
python temp/debug_full_pipeline.py --config configs/demo_25d.json --advisor local --model-path /data/models/Qwen3-1.7B --model-device cuda --max-calls 3 --plot
```

3次调用只用于加载、格式与日志诊断，同一决策点源细化就可能耗尽预算；之后数值回退到达不是模型成功。全程行为观察应使用足够预算并新建实验目录，不能把短诊断当同预算效果评价。

默认生成不限墙钟时间；输入／输出词元和咨询预算仍有限。不要为解决显存不足简单提高输入上限，不截掉未知、时序或关键观测后声称语义未变。需要调整上下文或精度时独立记录和验证，不悄悄修改模型原始响应。

## 5. 必须保存的日志

调试入口每次创建新的outputs/interference_debug_后端_时间戳目录：

- episode、metrics、evaluation_only_config、run_metadata：配置、计数、版本、源码哈希和终止状态。
- decision_log.txt、debug_stages.jsonl：中文输入／输出／执行记录及12阶段流程。
- 真实调用额外保存call编号的context、prompt、原始response、含聊天模板的model_input，以及calls.jsonl；错误原样保留。
- PNG／GIF为离线显示；阅读停顿不是生成时间，左真值不得回灌模型。

主实验入口run_experiment会复用默认输出目录，若使用它必须显式指定新--output-dir。不要覆盖入库证据，不因源拟合通过就宣称真实定位或飞行安全。

## 6. 接手后的研究任务

最新授权继续冻结主体的地图投影／时间层训练，独立输入已有点测／测向；当前训练入口及保存恢复已实现，但本地实际尝试显存不足、尚无参数更新，云端按CLOUD_ALIGNMENT_HANDOFF继续。原数据覆盖及空间／时序核查已做，时序候选仍有误报权衡；不重训地图编码器，不把重建、任务B标签或通道接入当源定位／策略成功。未启动LLM主体微调或RL。

原有真实模型诊断仍保留：先复现新的无模型环境，再产生新的真实模型上下文，检查：

1. 是否根据功率／方向／时序提出可检验源假设，或明确说明歧义并选择补测。
2. 是否理解反射来波不一定指向源，以及未知不等于低干扰。
3. 数值拒绝反馈后能否作有依据的修订；后续新点是否检验了旧预测。
4. 阶段计划、实际执行、解析失败、覆盖和回退是否分别可追溯。

本轮授权独立地图数据与编解码器，不擅自启动LLM微调／RL、门控、在线代码生成或完整电磁平台开发。后续局部代码生成和独立模型评价按current_roadmap规定前提开展，不以一次到达或一条短航路证明收益。

## 7. 可选离线工具

建筑导入：`tools/import_buildings.py`＋`configs/optional/import_25d_template.json`。静态功率图示例也在configs/optional，只支持power_only，当前示例数据缺失。

Sionna：`tools/generate_interference_rt.py`＋`configs/optional/sionna_rt_*.json`；仅在明确需要独立引擎核查时另建Python>=3.11（推荐3.12）环境并安装requirements-rt。它不提供在线测向，不是默认环境或LLM实验前提，不复制本机temp里的试装环境。详见`sionna_rt_environment.md`。

## 8. 给新agent的接手提示词

```text
先读AGENTS.md、docs/AGENT_HANDOFF.md、docs/current_roadmap.md、docs/first_version.md及docs/CLOUD_HANDOFF.md。
主环境是普通建筑＋本地时变干扰源＋有限一次反射，configs/demo_25d.json，不以Sionna或真实数据为前提。
核心src/uav_em和temp/debug_full_pipeline.py未迁移；离线工具在tools，可选配置在configs/optional。
先核对当前工作区／提交、独立Python环境、GPU和本地权重，再跑基础测试与无模型闭环；已知两项外部夹具缺失失败如实报告。
按current_roadmap第6节先检查随机多场景时序地图与小编解码器；命令见docs/map_encoding.md，再准备LLM适配与动态重规划RL，不误称已实现。
旧真实诊断目录当前缺失，不恢复或当作现存证据；none/mock/真实模型/回退分别记账，到达不等于模型有效。
当前仅离线地图训练授权，不擅自启动LLM微调／RL，不扩展传播平台、门控、在线代码执行、中断穿越或高度优化。新实验只写新的outputs/interference_目录，不动权重、原始数据或缓存。
```
