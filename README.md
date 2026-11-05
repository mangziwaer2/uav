# 未知电磁干扰环境中的LLM辅助无人机决策

更新：2026-11-05。研究核心是**地图不确定性下的非代码LLM辅助动态重规划**。新增离线地图编码基础，不将编解码本身当作研究终点，也不扩展完整电磁求解器或无模型路径算法竞赛。

## 当前状态

- 普通建筑、30m定高、未知多干扰源；支持源出现、消失、平移和功率叠加。
- 主配置采用三维自由空间衰减、等效穿楼损耗、一次墙／地面Fresnel反射；不依赖Sionna、外部数据或GPU。
- 无人机在当前位置接收总干扰功率和含误差360°波束；反射来波不一定指向源，没有测距或空间圆盘扫描。
- 每次有效观测更新belief，LLM可建议继续、重规划、补测及源假设；传统RRT*求解并复核，A*仅作对照。
- 环境量为总接收干扰功率dBm，越大越强。−80dBm只是软参考，不是禁飞、失联或损伤阈值。
- 当前新传播环境只验收了无模型闭环；没有完成新环境下真实LLM效果实验，不将到达或回退成功当作模型收益。

## 目录

```text
AGENTS.md                    agent维护约定
requirements.txt             基础仿真、绘图和测试依赖
requirements-llm.txt         可选真实本地LLM依赖，不是锁文件
requirements-rt.txt          可选离线Sionna依赖，须用独立环境
requirements-map.txt         可选小地图编解码器训练依赖
pytest.ini                   只收集tests，避免扫描temp中的环境和缓存
configs/
  demo.json                  平面矩形建筑主配置，显式single_reflection
  demo_25d.json              按逐栋楼高判断通行的定高主配置
  map_encoding.json          随机场景、三类行为采集及地图训练配置
  optional/                  外部功率图、地图导入与Sionna离线配置
src/
  run_experiment.py          正式实验入口
  replay_experiment.py       从episode离线绘图／回放
  uav_em/                    唯一正式仿真、感知、belief、模型和规划实现
tools/                       离线建筑／RT工具及地图数据、训练、编码检查入口
tests/                       基础及接口测试，不加载真实LLM或射线引擎
docs/                        当前交接、路线、协议、公式和论文资料
temp/debug_full_pipeline.py  保留给逐步调试的入口，直接调用正式simulate
outputs/                     实验记录；只显式选定的小部分证据列入Git候选
datasets/                    正式机载地图数据interference_onboard_maps_v1，以及用户RadioMapSeer
models/                      用户本地权重，不上传、不移动、不删除
```

不要扫描、清空或迁移整个`temp/`：其中包含用户／第三方缓存及独立Python环境。旧项目不恢复、不新建归档。

## 快速开始

从项目根目录使用Python 3.10以上的独立环境：

```bash
python -m pip install -r requirements.txt
python temp/debug_full_pipeline.py --config configs/demo_25d.json --advisor none --plot --gif
python -m pytest -q -p no:cacheprovider
```

Windows本机已有基础解释器：`E:\Miniforge\envs\uav\python.exe`。云端创建自己的环境，不复制该路径、Windows虚拟环境或toolkit混用设置。

调试入口每次自动创建新的`outputs/interference_debug_后端_时间戳/`，不会覆盖旧结果。正式实验入口的默认输出目录会复用；使用它时必须显式指定未使用过的`--output-dir`，不要覆盖证据。具体参数可用`--help`查看。

逐步调试：

```bash
python temp/debug_full_pipeline.py --config configs/demo_25d.json --advisor none --pause-at observation_received --pause-at source_evaluated
```

IDE断点、12个阶段和字段说明见`temp/debug_full_pipeline.md`。`--advisor mock`只是固定响应夹具，`--advisor local`才加载真实模型；本地默认不限制生成墙钟时间，调用、输入和输出词元预算仍有限。

## 运行闭环与信息边界

```text
隐藏时变源＋建筑 → 传播路径 → 当前点功率／含误差波束
 → 更新前innovation与旧源预测检查 → belief更新
 → 周期／观测事件复核 → 可见上下文 → 数值策略或LLM提议
 → 源假设检验／可选反馈修订 → RRT*／可选A*＋几何能量复核
 → 执行短航段 → 再测量
```

仿真器知道真值以生成测量；模型和规划器不能读取隐藏源、未来脚本或未观测全图。左图是研究者真值，右图是机载估计；40m是局部估计支持半径，不是传感器扫描半径。每次有效测量都更新地图，innovation用于复核而不是地图更新开关。

## 当前可用记录与已知缺口

当前实际存在的数据、审核和训练结果见`outputs/README.md`。2026-11-04复查时此前none导航记录及上午地图冒烟输出已缺失，不继续把历史路径列为现存证据，也不从Git恢复。

当前测试结果及地图管线冒烟限制见`docs/AGENT_HANDOFF.md`。仍有两个既有可选夹具缺失：`datasets/example_power_grid.json`和`datasets/example_buildings_local.geojson`；主配置不依赖它们，没有跳过失败或补造数据。

旧三份精选诊断和此前RT输出已不在当前工作区，不当作现存证据，也未从Git历史恢复。Git当前存在未提交改动与删除；本轮目录整理不等于已经提交／上传。

## 阅读顺序

1. 新agent先读`docs/AGENT_HANDOFF.md`、`docs/current_roadmap.md`、`docs/first_version.md`。
2. 云端接手读`docs/CLOUD_HANDOFF.md`；只按当前路线继续非代码LLM源反馈检查。
3. 公式与近似读`docs/simulation_foundation.md`，参数读`configs/README.md`。
4. 可选工具读`tools/README.md`，其他资料读`docs/README.md`。

## 地图编码基础

随机2.5D场景＋goal／coverage／revisit三类独立采集，保存当时可见地图并按场景分组；小Transformer编解码器训练与时序token检查入口见`docs/map_encoding.md`。正式首版配方`configs/map_encoding_full.json`为120场景／720轨迹，先使用`tools/audit_map_dataset.py --require-coverage`逐帧审核，再用`tools/train_map_autoencoder.py --require-quality`训练。生成支持`--workers`，所有工具提供`--help`。

当前实现独立地图／时序约束训练、任务B、区域问答、独立测向证据及冻结LLM主体的地图投影／时间层训练。此前Qwen3-1.7B短forward接线通过；本轮完整输入／答案训练在6GB设备上遇真实CUDA OOM，完成优化器步数0，尚无完整地图回答或RL，不改变正式在线闭环。云端继续见`docs/CLOUD_ALIGNMENT_HANDOFF.md`；不启用代码生成、门控、中断穿越或三维航路优化。

正式数据统一命名为`datasets/interference_onboard_maps_v1/`，先读其中DATASET_CARD.md；`previews/goal/`与`previews/revisit/`有6通道／4帧历史／任务B图片、1倍速GIF及Excel可读CSV。预览不需要模型，其他轨迹用`tools/preview_map_dataset.py`。原始数据及v1训练记录见`docs/map_dataset_run_20261104.md`；当前编解码改进配置为`configs/map_encoding_v2.json`，方法与实际结果见`docs/map_codec_v2.md`，不变更原始数据版本或划分。

同一区域时序保真工具为`tools/audit_map_temporal_codec.py`，默认固定权重检查validation，输出支持增失／功率变化、功率差／年龄差及原图／重建对照。实际1,731道任务核查与剩余功率变化误报见`docs/map_temporal_codec.md`。历史窗口导出新格式包含全部有效位重建，仍是逐帧空间编码，不代表已训练时序模块或LLM。

时序约束微调入口`tools/train_map_temporal_autoencoder.py`，配方`configs/map_encoding_temporal.json`，已完成12轮、选第8轮；变化格差值与召回改善，但误报增加，候选不自动取代v2，结果见`docs/map_temporal_training.md`。连续嵌入接口见`docs/map_embedding_bridge.md`：4帧映射至128个地图向量位置，实际时间单独编码；短前向已接通，完整对齐尚未完成。

独立点测／测向位于`datasets/interference_sensor_evidence_v1/`，612条轨迹、30,843次扫描，所有19,606个问答窗口的因果检查通过，不重训六通道编码器。先看其中`samples/validation.json`。训练入口`tools/train_map_alignment.py`、配方`configs/map_alignment.json`及协议`docs/map_sensor_alignment.md`包含完整答案预算、提示／地图标签屏蔽、冻结检查与恢复；真实本地OOM现场和step=0恢复点已单独保留。

本轮固定原v2作为接口基线，已生成并逐题审核`datasets/interference_map_alignment_v1/`：训练16,144题／验证3,462题，六通道区域还原与时序比较两类。先看其中DATASET_CARD及samples；输入和答案分离。真实模型接线结果与复现命令见`docs/map_alignment_data.md`，只证明一次forward可运行，不证明随机投影器已经理解地图。
