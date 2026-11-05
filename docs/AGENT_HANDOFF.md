# Agent接手：当前状态

更新：2026-11-05。本文件只记录现状和接手顺序，不累积历史试跑时间线。先读根`AGENTS.md`，再读本文件、`current_roadmap.md`和`first_version.md`；云端另读`CLOUD_HANDOFF.md`。

## 1. 用户最新要求

- 研究核心：未知／不确定电磁环境中的**非代码LLM辅助决策**，模型根据观测提出假设、补测和绕行，传统算子规划与复核。
- 主环境回到普通建筑＋本地时变干扰源，完善公式、允许合理简化；不以真实城市、真实设备、Sionna或完整电磁仿真作为研究前提。
- 当前只声称仿真研究。2026-11-04授权先实现随机场景、多行为时序地图数据与小编解码器，再适配LLM；2026-11-05进一步授权独立测向输入及冻结LLM的投影训练，显存不足后转云端。真实尝试已OOM、优化器步数为0；未微调LLM主体或启动RL、门控、在线代码执行、中断穿越或三维航路优化。

### 新增离线地图学习入口

`src/uav_em/map_learning/`、`configs/map_encoding.json`、`requirements-map.txt`及tools下generate_map_dataset／train_map_autoencoder／inspect_map_encoding。完整协议与命令见`map_encoding.md`。随机场景验几何连通、净空、能量／时长、源轨迹及少量有限功率；数据按场景分组，三类行为不依赖隐藏源选路。原始历史帧不事后重算，任务B仅监督估计／支持／年龄变化，不标注源移动。已有时序候选、区域问答、短forward及独立测向／冻结主体投影训练入口，真实训练尝试OOM、0优化器步，详见下节。正式simulate未接编码器，无LLM主体微调或RL。

## 2. 本轮目录整理

### 当前优先交接：投影训练OOM

先读`CLOUD_ALIGNMENT_HANDOFF.md`和`map_sensor_alignment.md`。独立证据在`../datasets/interference_sensor_evidence_v1/`：612条轨迹、30,843次点测／扫描、16,878次有角峰；全部19,606题因果窗口检查通过。地图tokens不变，读最近60秒实际测向，不提供源坐标、未来数据或答案。

本地真实训练目录`../outputs/interference_map_alignment_training_20261105_135229_972149/`，6GB设备在完整输入／答案训练时出现CUDA显存不足；已停止，不继续本机重试。last.pt实查optimizer_steps=0、sample_cursor=0，仅初始化恢复点。原torch.AcceleratorError未被旧OOM捕获器覆盖，所以该目录没有自动report.json／oom.json；事后说明见POSTMORTEM.md，不补造运行时字段。最新源码已修正该异常分类及跨精度scaler恢复，使用小模型测试验证，不重复真实GPU运行。没有完成对齐或完整答案生成验收。

| 位置 | 当前职责 |
|---|---|
| `../src/uav_em/` | 唯一仿真与决策实现；保持位置和算法不变 |
| `../src/run_experiment.py`、`replay_experiment.py` | 正式实验／离线回放入口，位置不变 |
| `../temp/debug_full_pipeline.py`及同名md | 用户指定完整调试入口，位置不变，不复制simulate |
| `../configs/demo.json`、`demo_25d.json` | 两份主配置，均显式single_reflection |
| `../tools/` | 从src移入的可选建筑导入、Sionna导出工具；新路径见其README |
| `../configs/optional/` | 从configs移入的4份外部／离线配置，不作为主实验前提 |
| `../requirements*.txt`、`../pytest.ini` | 按当前代码重建依赖入口及测试收集范围；不恢复旧environment.yml |
| `../outputs/README.md`、`../datasets/README.md` | 当前证据、缺失文件和用户数据边界 |

旧离线脚本路径没有兼容包装器。相对数据路径、测试和文档已随移动更新。没有清空或搬动用户模型、RadioMapSeer、ZIP、第三方缓存、独立Python环境或原始提交文件；没有恢复旧项目或创建归档。

## 3. 当前能力与近似

- 固定高度／恒速二维航路；可选逐栋多边形楼高、庭院和飞越净空。
- 主JSON：三维自由空间衰减、每栋一次等效穿透损耗、镜像法有效一次墙／地面Fresnel反射。有限墙段／高度和反射两段遮挡均检查。
- 源和路径功率在mW域叠加，背景只加一次，最后转dBm；不是服务覆盖或通信质量。−80dBm是软参考，不是禁入、失联或损伤阈值。
- 总功率与360°含误差波束使用同一组传播路径。反射角峰可能指向墙，角峰数不是源数；没有测距、圆盘空间扫描或传感器直接给源坐标。
- 点测每次更新belief；innovation是更新前测量与估计的偏差，用于触发复核，不是地图更新开关。40m是估计支持半径，35s后失效，不是探测范围。
- LLM提议继续／重规划／补测、profile、采样区域、阶段计划及可选传播源假设。最近观测检验拟合一致性，多轮反馈不产生额外测量；新点先检验旧预测再更新。
- 默认RRT*，A*仅对照。几何／能量硬复核，干扰软代价；没有RRTX或安全最优保证。
- 未标定统一材料、等权极化功率、等效整栋损耗；无屋顶反射、多次反射、绕射、相干相位、阵元IQ或真实平台故障模型。
- 为保留历史语义，省略新增字段的配置及ProjectConfig仍为log_distance／2.2；两份主JSON显式single_reflection／2，不混作同环境效果对照。

公式：`simulation_foundation.md`；逐字段协议：`../configs/README.md`；实际流程和模型协议：`first_version.md`。

## 4. 信息边界

真值只供环境生成测量、暴露评价和左图。模型和规划器只能用机载估计、未知标记、历史观测和公开几何／传播参数；不能访问隐藏源、反射点、未观测全图或未来脚本。源假设是模型提议，不是事实；拟合通过不代表正确定位。

none＝数值基线，mock＝固定工程夹具，local＝真实模型。分别记录原始输出、解析、数值拒绝、实际执行、失败和回退；回退到达不等于LLM有效。

## 5. 当前可核查记录

2026-11-04本轮复查时，先前的`../outputs/interference_debug_none_20261103_144015_801006/`及上午地图冒烟输出已不在工作区；不推断删除原因，不恢复或继续称其为现存证据。当前实际输出见`../outputs/README.md`。本轮地图采集不是正式simulate实验，不混为导航或LLM收益证据。

旧三份精选记录、较早真实LLM失败日志及此前RT输出已不在工作区。历史曾有OOM、输入超限、未提源等描述，但缺少现存原始文件，不再当作可复现证据或新环境结论，不从Git历史恢复它们。

## 6. 验证与已知问题

2026-11-05本轮最终代码全量测试：**448通过、2失败（122.11s）**。新增13项独立测向／投影训练测试全部通过，覆盖因果窗口、缺测／无角峰区分、无隐藏字段、完整预算／答案屏蔽、非重入检查点下真实小Qwen3反向、投影／时间层更新而主体不变、完整保存恢复一致、两类OOM异常及跨精度空scaler恢复；此前区域问答、接口、时序训练及地图数据测试均保留。仍仅下述两个既有外部夹具缺失失败，没有修补、跳过或掩盖。命令另加`--basetemp temp/interference_sensor_alignment_full_20261105_02`。

正式数据命名为UAV电磁干扰机载地图数据集v1，唯一入口为`../datasets/interference_onboard_maps_v1/`：120场景、720轨迹、36,321帧、11,549道时序题。原随机test组缺单源，严格审核拒绝；按源数分层整场景另存后完整审核通过。已整体迁入datasets，manifest哈希不变，迁入后720条轨迹再次严格审核零错误；旧运行元数据保留原路径。数据说明见其中DATASET_CARD.md。

v1训练30轮、选第2轮；在其validation诊断后完成v2训练40轮、选第39轮。v2保持32×96瓶颈，增加局部卷积／patch解码、边界损失与训练翻转；旧结构／权重继续支持。test支持IoU由历史0.536至0.976，强干扰功率MAE由2.914至2.370dB，但整体功率MAE约2.48dB未改善，且参数增加。当前v2权重在`../outputs/interference_map_encoder_20261104_174105_494050/best.pt`，完整方法／局限见`map_codec_v2.md`。不把同一test的再次观察称为新盲测，不把逐帧重建当时序理解或LLM收益。

原v2的固定时序核查：108条validation轨迹、5,433帧、1,731道原始任务B；变化格差值MAE 4.513dB、变化精度0.357／召回0.638。历史报告保留在`../outputs/interference_map_temporal_codec_20261104_181909_806135/`，方法见`map_temporal_codec.md`。

本轮从v2微调空间编解码器，保留单帧目标并增加共支持功率差、年龄差、支持增失；train按可见变化分层采样，validation自然分布选模，保留epoch0基线及空间退化限制。完成12轮、选第8轮，候选权重在`../outputs/interference_map_encoder_temporal_20261105_114745_542222/best.pt`；未读test或隐藏源。核查在`../outputs/interference_map_temporal_codec_20261105_115226_042619/`：变化格差值MAE降至3.922dB，召回升至0.704，但精度降至0.344、假变化由1,345增至1,571，总体差值MAE略升。单帧支持IoU与功率MAE改善，但时序误报未解决；仅保留候选，不自动替换原v2。详见`map_temporal_training.md`，不能只报告改善项。

连续嵌入接口MapEmbeddingProjector支持[B,4,32,96]→[B,128,llm_dim]及实际时间MLP；insert_map_embeddings在文字词嵌入后拼接，返回attention_mask／position_ids，地图labels=-100。已通过张量单测及本地Qwen3-1.7B一次无梯度forward，没有训练参数、完整生成或对齐；详见`map_embedding_bridge.md`与`map_alignment_data.md`。后续权重需绑定编码器哈希与准确LLM版本，不把相同维度当语义兼容。

为停止无限改编码器，本轮显式固定原v2第39轮作接线基线，时序候选继续保留对照；这不等于消除误报或源移动可辨识。派生数据`../datasets/interference_map_alignment_v1/`已生成train 16,144题／validation 3,462题，每原时序题派生当前ROI六通道还原及同ROI时序比较两类，保留原场景组；612条轨迹、30,843帧缓存。完整审核逐题从原机载图重算答案已通过。输入question／tokens与answer分离；缓存只取因果窗口，不能读整条未来帧或答案旁路，不导出test／隐藏源。

实际接线记录`../outputs/interference_map_llm_bridge_20261105_122625_826523/`：本地Qwen3-1.7B，随机投影器，文字316位置＋地图128＝444位置，inputs_embeds形状[1,444,2048]，末位置logits有限；0优化器步，不评价输出语义。记录模型／编码器哈希、真实聊天模板及前后文字ID；仅适用本机该版本的forward，不代表长答案训练或生成缓存兼容。后续先准备监督拼接／预算／冻结检查再执行对齐；本轮未运行LLM训练。

inspect_map_encoding的新window_schema_version=2增加全部窗口位的reconstructed_window，保留旧anchor字段；无效位为零，必须结合valid。当前仍是逐帧空间编码，不是已训练时序模块。新导出在`../outputs/interference_map_inspect_20261104_181924_344676/`，后续LLM不能读取其中诊断原图作为tokens旁路。

原始样本预览现在正式数据的`previews/goal/`与`previews/revisit/`，有6通道／历史窗口／任务B PNG、1倍速GIF、JSON及栅格／点测CSV；旧outputs预览不删除。工具`tools/preview_map_dataset.py`无需torch或checkpoint且不读取隐藏真值。另见`../outputs/interference_map_codec_comparison_20261104_175339_533173/README.md`，两条validation样本对比原图／v1／v2；预览与模型重建分开保存。

上午1轮冒烟曾出现support IoU为0，只是链路检查，原输出现已缺失，不能拿来开展LLM适配或声称保真。正式训练使用验证集选择checkpoint，test仅在最终模型选定后检查；未知、支持mask、强干扰与年龄都要看，不只看平均loss。新输出必须另开目录。

目录整理最终验收：`python -m pytest -q -p no:cacheprovider`为**335通过、2失败（94.37s）**；本轮新增7项目录／工具入口测试通过，定向的目录＋RT适配共26项通过。与整理前328通过相比没有新增失败。两项失败只因缺少`datasets/example_power_grid.json`和`datasets/example_buildings_local.geojson`，是可选适配夹具，不是主流程依赖；未补造、恢复或跳过。

此前目录整理核查：可选脚本从非根工作目录运行--help成功；旧命令路径只在tools/README迁移对照中保留；功率图相对引用仍指向原datasets目标；git diff --check通过。现存episode的全部运行时源码／配置哈希仍匹配；Git忽略核查确认权重、数据ZIP、虚拟环境、可重建GIF被忽略，源码、可选配置、数据说明和选定episode可列入候选。该次整理没有运行真实模型或射线引擎；本轮另行完成了上述本地LLM前向接线，未改写原始实验日志。

基础命令：`python -m pytest -q -p no:cacheprovider`。已重建pytest.ini限定tests，不能在temp虚拟环境和缓存中递归搜测试；本轮另加`--basetemp temp/interference_map_alignment_full_20261105_02`，避免沙箱外系统临时路径。依赖范围不是锁文件；整理前本机元数据：Python3.10.22、NumPy2.2.6、Matplotlib3.10.9、Pillow12.3.0、pytest9.1.1。本轮真实LLM前向使用torch2.14.1+cu126、transformers5.20.0；safetensors0.8.0是已安装元数据。前向通过不保证训练／生成或其他版本兼容。

Windows基础解释器`E:\Miniforge\envs\uav\python.exe`。云端不要混用Windows toolkit路径，也不复制temp中的旧Python环境。可选RT另建Python>=3.11（推荐3.12）环境，不能只以pip安装成功或--help成功代替GPU实算。

## 7. 接下来做什么

唯一详细研究清单在`current_roadmap.md`第6节。下一步在云端以原v2缓存、独立测向证据和相同Qwen3-1.7B继续冻结主体的投影／时间层训练，先读CLOUD_ALIGNMENT_HANDOFF；本地已触及显存边界，不重复重试。当前last.pt是step=0，不把接口、预算通过或OOM尝试称为理解成功。保留时序候选与误报限制，不无限重训编码器；后续再验证自由生成、源假设／行动及动态重规划RL，不扩展传播阶数或恢复旧证据。

研究顺序仍为非代码闭环 → 按前提局部代码生成；模型评价／调优可独立安排，不以代码生成或门控成功为条件。未来方向不代表本轮执行授权。

## 8. Git与输出规则

当前包括目录整理和新增地图学习工作，均未git add／commit／push。存在未提交新增、修改和旧文件删除；上传前逐项审查，不把完成编辑当作已上GitHub。

仅源码、配置、测试、依赖、维护文档、指定调试入口和当前显式选定的小部分证据列为Git候选。权重、原始数据、ZIP、缓存、虚拟环境和原始提交文件保持忽略；当前GIF可从episode重建，不默认上传。根environment.yml保持已删除状态，以requirements入口为准。

新实验必须用新的outputs/interference_目录。优先调试入口自动时间戳；正式run_experiment默认会复用目录，需显式新--output-dir，绝不覆盖证据。当前episode和run_metadata中的路径／哈希属于运行时记录，不因整理而改写成新路径。
