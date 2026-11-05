# 本地数据与缺失夹具

主流程使用`configs/demo.json`／`demo_25d.json`中的合成源与建筑，无需下载数据。所有实验只声称电磁环境仿真研究，不声称真实无人机或电磁设备验证。

## 当前正式训练数据

`interference_onboard_maps_v1/`：**UAV电磁干扰机载地图数据集v1**，当前唯一正式地图训练入口。120场景、720轨迹、36,321帧、11,549道时序题；按场景分组，已完整审核。先打开其中DATASET_CARD.md，预览在previews/goal与previews/revisit。二进制npz、公开JSON、时序JSONL和质量报告均保留；evaluation_only只供离线复现，模型不能读取。数据未入Git，云端需单独传输。

2026-11-04从outputs中的时间戳目录整体迁入，manifest哈希未变且迁移后重新审核通过。旧实验日志中的运行时原路径不改写。数据版本不随编解码器v1／v2变动；后续变更原始数据、标签或分组必须另建数据版本。

## 编码对齐派生数据

`interference_map_alignment_v1/`：**UAV机载地图编码对齐数据v1**，用于后续区域还原／时序问答，不替代上面的原始机载数据。固定原v2第39轮，train 16,144题、validation 3,462题，保留原场景划分，不导出test。tokens只缓存连续编码及时间，question／inputs和answer分开；读取器只取因果历史窗，不能把整条缓存或答案当输入。先看DATASET_CARD.md及samples/current_region.json、samples/temporal_region.json；完整逐题原图重算审核已通过，见quality_report.json。方法、实际本地LLM前向与局限见`../docs/map_alignment_data.md`。原v2仅作固定接线基线，不声称已经消除变化误报；尚无LLM训练。数据被Git忽略，云端须单独传输。

## 独立测向证据

`interference_sensor_evidence_v1/`：配合地图对齐数据的独立点测／测向包，612条train／validation轨迹，30,843条点测和扫描，16,878次扫描检测到角峰。最近60秒的因果证据独立作为文字输入，不改原六通道地图或tokens、不提供源坐标；全部19,606题窗口检查通过，见quality_report.json。先看samples/train.json及samples/validation.json，包含展开观测和实际紧凑问题；没有标准答案。原数据与地图对齐数据保持不变，不导出test，不读隐藏真值。训练时需要本目录和interference_map_alignment_v1完整目录，二者均被Git忽略；继续云端训练见`../docs/CLOUD_ALIGNMENT_HANDOFF.md`。

## 用户资源与缺失夹具

当前用户资源为`RadioMapSeer/`和`RadioMapSeer.zip`。本轮不移动、解压、改写或删除，也没有把它们接入默认干扰环境；数据适用性与转换须另行核查，不能把路径损耗或服务覆盖直接叫作总干扰功率。

当前缺少`example_power_grid.json`、`example_buildings_local.geojson`两个旧小夹具，因此对应两项可选适配测试会失败。没有自动恢复、合成替代或跳过测试。用户外部数据可放`local/`或仓库外，必须记录来源、许可、坐标、频率、高度和量纲。

Git仅放本说明；原始数据和ZIP默认忽略。后续补入小夹具应显式标记为合成测试数据，与用户数据分开，不以修复这些夹具作为LLM实验前提。
