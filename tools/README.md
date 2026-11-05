# 可选离线工具

在线闭环和主运行入口仍在`src/`；本目录处理离线环境准备与地图学习，不是LLM可调用的真值工具。所有命令从项目根目录运行。

## 地图数据与编解码器

`generate_map_dataset.py`生成随机可用场景并采集三类机载地图轨迹，支持`--workers`并行；`split_map_dataset.py`按源数分层划分整场景，另存新目录不改原数据；`audit_map_dataset.py`逐帧因果重放和覆盖审核；`train_map_autoencoder.py`独立训练小空间编解码器；`inspect_map_encoding.py`导出历史窗口tokens及重建PNG。小配方`configs/map_encoding.json`，正式首版配方`configs/map_encoding_full.json`（120场景／720轨迹），正式训练先分层、`--require-coverage`审核再`--require-quality`训练。完整命令、任务B数据、信息边界及未实现能力见`../docs/map_encoding.md`。生成、分组与审核不依赖torch，训练／编码检查使用`requirements-map.txt`；不启动LLM或RL。

原始地图样本预览：`preview_map_dataset.py --dataset <数据目录> --episode <manifest中的id>`，生成6通道与时序题PNG、慢速GIF及可用Excel打开的CSV；不需要checkpoint／torch，不读隐藏真值。实际两条样本路径和训练数值见`../docs/map_dataset_run_20261104.md`。

同区域时序保真：`audit_map_temporal_codec.py --dataset <数据目录> --checkpoint <best.pt>`，默认固定权重检查validation的原始任务B，导出事件计数、功率／年龄差、分行为报告、CSV和对照PNG；不训练LLM，不读隐藏源。新历史窗口导出另含全部有效帧重建及有效位，不再只有末帧可检查。实际结果和边界见`../docs/map_temporal_codec.md`。

时序约束训练：`train_map_temporal_autoencoder.py --dataset <数据目录> --checkpoint <v2权重> --require-quality`，默认配方configs/map_encoding_temporal.json；保留单帧重建并加入共支持功率差、年龄差及支持增失，train按可见任务变化分层采样，validation自然分布选模且保留epoch0基线；不打开test轨迹、隐藏源或启动LLM训练。命令与选模限制见`../docs/map_temporal_training.md`。

LLM对齐准备：`build_map_alignment_dataset.py --dataset <原始机载数据> --checkpoint <固定编码器> --output <新目录>`导出train／validation连续tokens及两类区域问答，`audit_map_alignment_dataset.py --dataset <派生目录> --source <原始目录>`逐题核对因果窗口和重算答案，不导出test、不读隐藏源。`check_map_llm_bridge.py --dataset <派生目录> --model-path <本地LLM> --device cuda`只做随机投影器的一次无梯度forward，记录模板／形状／哈希，不训练、不生成完整答案或证明理解。实际路径和命令见`../docs/map_alignment_data.md`。

## 独立测向与冻结投影训练

`build_map_sensor_evidence.py --dataset <地图对齐目录> --source <原始机载目录> --output <新目录>`从已有公开观测派生独立测向包，逐题检查因果窗口，不重训编码器、不读test／隐藏源。正式目录datasets/interference_sensor_evidence_v1。

`train_map_alignment.py --dataset <地图对齐目录> --evidence-dataset <独立证据目录> --model-path <本地LLM> --config configs/map_alignment.json --device cuda`只训练投影／时间层；`--budget-only`先核查完整问题／地图／答案／EOS位置预算，`--resume <last.pt>`恢复相同数据／模型与设置。microbatch固定1、SDPA＋检查点，CUDA OOM写诊断并停止，不截断或回退CPU。所有新输出另开目录；本机已遇真实OOM、0优化器步，专门交接见`../docs/CLOUD_ALIGNMENT_HANDOFF.md`，协议见`../docs/map_sensor_alignment.md`。

## 建筑导入

`import_buildings.py`将用户提供的GeoJSON转换为自包含配置，不联网、不自动推断缺失楼高。配置模板为`configs/optional/import_25d_template.json`。

```bash
python tools/import_buildings.py --help
```

实际导入时提供自己的`input`、`--coordinates`、`--source`、`--license`及未存在的`--output`；缺失高度必须显式决定`--default-height`。协议见`../docs/environment_25d.md`。原文提及的两个datasets小夹具目前缺失，不能把示例命令当作已可直接运行。

## Sionna RT核查

`generate_interference_rt.py`是可选离线导出器，默认读取`configs/optional/sionna_rt_demo.json`和`sionna_rt_sources.json`，输出新的`outputs/interference_rt_.../`。它不影响主配置，也不进入LLM咨询。

```bash
python tools/generate_interference_rt.py --help
```

实际求解须另建Python 3.11以上（推荐3.12）环境并安装根目录`requirements-rt.txt`。`--help`不导入射线引擎；不能用此成功冒充GPU求解验收。详见`../docs/sionna_rt_environment.md`。旧本地安装和缓存不移动、不复制到云端。

## 本轮路径变更

- `src/import_buildings.py` → `tools/import_buildings.py`。
- `src/generate_interference_rt.py` → `tools/generate_interference_rt.py`。
- 四份离线配置移至`configs/optional/`；主配置与`src/uav_em/`位置不变。

没有保留旧入口包装器或复制算法；新文档与测试统一使用新路径。
