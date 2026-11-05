# 当前输出与复现入口

更新：2026-11-05。这里只列核查时实际存在的记录，不从Git恢复缺失的历史实验，也不把地图采集当LLM导航实验。

## 本轮地图数据

- `interference_map_data_20261104_155420_499056/`：正式配方的6场景试生成，36条轨迹，完整性审核通过；因每个留出组只有1场景，未作完整覆盖验收，不用于正式训练。
- `interference_map_data_20261104_155905_340342/`：120场景／720轨迹的原始完整采集；严格审核发现原随机test组无单源案例，报告保留为失败，不用于正式训练。
- 原`interference_map_data_stratified_20261104_163312_927539/`已按用户要求整体迁入`../datasets/interference_onboard_maps_v1/`，这是唯一正式训练入口。720轨迹、36,321帧、11,549道时序题，迁入后严格审核再次通过；旧目录不另留重复副本，旧实验元数据不事后改写。
- `interference_map_encoder_20261104_163654_653599/`：用户要求停止的早先尝试。第1轮未完成，只有训练配置及空history，没有checkpoint或最终指标；保留该暂停记录，已在下列新目录重训。
- `interference_map_encoder_20261104_170403_750336/`：v1基线实际完成30轮CUDA训练；按验证loss选第2轮，test支持功率MAE约2.477dB，支持mask IoU约0.536。边界偏宽，旧权重保留，不以训练完成宣称适配达标。
- `interference_map_codec_diagnostic_20261104_172932_498897/`：v1的validation分通道诊断；用于确定支持边界与局部结构改进，不使用test选修改方案。
- `interference_map_encoder_20261104_174105_494050/`：v2完成40轮CUDA训练，按validation选择第39轮。test支持IoU约0.976、强干扰功率MAE约2.370dB；整体支持功率MAE约2.480dB，未改善。best.pt及完整metrics.json在此；方法、参数增加和非盲测限制见`../docs/map_codec_v2.md`。
- `interference_map_codec_comparison_20261104_175339_533173/`：两条validation样本的原图／v1／v2对照，先看README.md及goal／revisit下comparison.png；输入信息边界与瓶颈维度相同，不代表等算力消融或导航收益。
- `interference_map_samples_20261104_171354_475695/`：早先原始数据预览保留。当前推荐打开正式数据`../datasets/interference_onboard_maps_v1/previews/`，有两条样本README、6通道／4帧窗口／时序题PNG、1倍速GIF、JSON和CSV，不读取隐藏源。
- `interference_map_inspect_20261104_171433_536036/`：已训练checkpoint的验证样本重建图与4帧tokens，可与原始数据预览区分。
- `interference_map_temporal_codec_20261104_181909_806135/`：固定v2第39轮对108条validation轨迹、1,731道原始任务B的同区域时序核查。新增／失效支持召回约0.985／0.961；3dB功率变化精度仅0.357，变化格功率差MAE 4.513dB，未达到可以忽略动态误差的程度。README、整体／分行为JSON、逐题CSV及四类对照PNG均保留；不是LLM或源运动实验。较早181530版本保留，当前版本增加平稳区域示例，整体指标一致。
- `interference_map_inspect_20261104_181924_344676/`：v2新格式4帧窗口，window_schema_version=2，含每个有效历史位的reconstructed_window；无效位须结合valid处理，旧anchor字段继续保留；不是时序模型训练结果。
- `interference_map_encoder_temporal_20261105_114745_542222/`：保留v2单帧目标并增加时序辅助约束；12轮CUDA微调选第8轮，包含基线、逐轮记录、配置／源码／父权重哈希、best.pt及metrics。只用train／validation，不运行test；epoch是本次微调轮数。候选不自动替换v2，混合结果见review.md及`../docs/map_temporal_training.md`。
- `interference_map_temporal_codec_20261105_115226_042619/`：上列候选的同一1,731道validation题复查，变化格差值MAE从4.513降至3.922dB、变化召回0.638至0.704，但假变化1,345至1,571，精度下降。不隐藏退步项，不作盲测、源定位或LLM收益结论；图例及四类对照见目录README。

详细分布、审核中发现的问题与复现入口见`../docs/map_dataset_run_20261104.md`。没有启动LLM微调或RL。

## 对齐数据与真实模型接线

- `../datasets/interference_map_alignment_v1/`：本轮完整导出的区域问答，train 16,144题、validation 3,462题，共612条轨迹、30,843帧缓存；不导出test。已用原始机载帧逐题重算答案，quality_report.json通过；原数据未改，编码器固定原v2哈希，仅作接口基线，不宣称解决时序误报。先看samples下两条问答及`../docs/map_alignment_data.md`。
- `interference_map_llm_bridge_20261105_122625_826523/`：本地Qwen3-1.7B一次真实无梯度forward，316文字位置＋128地图位置＝444，输入[1,444,2048]、末位置logits有限。含run_config、prompt、report和README；随机投影／时间层、0优化器步，没有完整答案生成，不是语义对齐或导航效果验收。记录模型／编码器哈希及实际库版本，数据和模型权重需单独迁移。

配方`../configs/map_encoding_full.json`，生成、审核和训练命令见`../docs/map_encoding.md`。数据、隐藏复现配置及权重被Git忽略，云端使用须单独复制完整目录；不得只复制npz而丢失manifest和公开元数据。

## 独立测向与投影训练

- `../datasets/interference_sensor_evidence_v1/`：612条train／validation轨迹的独立点测／测向，30,843次扫描、16,878次有角峰；全部19,606题因果窗口审核通过。地图编码和缓存不改，不给源坐标、test或未来观测。方法见`../docs/map_sensor_alignment.md`。
- `interference_map_alignment_training_20261105_134844_225657/`：真实Qwen3-1.7B tokenizer的全量位置预算，train最长7,885、validation最长7,808，均低于8,192；含完整测向问题、地图、完整答案及EOS，不截断、不加载模型训练。
- `interference_map_alignment_training_20261105_135229_972149/`：真实6GB设备的冻结主体投影训练尝试，CUDA OOM结束、0优化器步；last.pt实查为初始化恢复点，不是训练成果。原AcceleratorError未被旧捕获器覆盖，该目录没有自动report／oom；POSTMORTEM.md和terminal_failure.txt为明确标记的事后说明，原配置／权重不重写。最新代码已修正分类并测试，不重复本机GPU尝试；云端见`../docs/CLOUD_ALIGNMENT_HANDOFF.md`。此目录与数据被Git忽略，需单独传输。

## 历史记录边界

本次核查时，旧三份精选记录、`interference_debug_none_20261103_144015_801006/`及此前上午的地图冒烟目录均已不在outputs下；不推断删除原因，不恢复或冒充现存证据。交接文档中的历史数值不是本轮可复现结果。

新实验必须使用新的`interference_`目录，不覆盖记录。源码与本索引可入Git，数据、checkpoint、缓存及普通输出不默认上传；当前没有选择新的导航实验记录入库。
