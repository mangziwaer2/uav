# 文档索引

更新：2026-11-05。主线是普通建筑、未知时变电磁干扰中的非代码LLM辅助动态重规划。主环境不依赖真实设备、外部数据或Sionna；不要把可选工具、未来方向或历史试跑描述当当前能力。

## 接手必读

| 顺序 | 文档 | 负责内容 |
|---|---|---|
| 1 | `AGENT_HANDOFF.md` | 当前目录、能力、现存证据、测试缺口和下一步入口；不保留长历史时间线 |
| 2 | `current_roadmap.md` | 唯一研究阶段与待办入口；后续代码生成、可选分支和模型评价的前提 |
| 3 | `first_version.md` | 实际simulate流程、模型协议、源检验、预算和结果字段 |
| 云端 | `CLOUD_HANDOFF.md` | 文件迁移、独立环境、GPU／权重检查、新实验和可复制命令 |
| 当前训练 | `CLOUD_ALIGNMENT_HANDOFF.md` | 本地6GB真实OOM、step=0恢复点、地图＋测向数据迁移及云端续训命令 |

根`../README.md`负责快速运行、目录与入口；`../AGENTS.md`负责维护约定。

## 当前实现依据

| 文档 | 用途 |
|---|---|
| `simulation_foundation.md` | 传播公式、一次Fresnel反射、波束、误差、信息边界与未实现物理机制 |
| `environment_25d.md` | 楼高／庭院／净空、GeoJSON导入和静态功率图协议；历史夹具缺失已注明 |
| `map_encoding.md` | 随机场景、多行为时序数据、任务B标签、小编解码器、云端命令与未实现的LLM／RL接口 |
| `map_dataset_run_20261104.md` | 正式首版数据的实际路径、分组／覆盖、审核修正及真实训练记录 |
| `map_codec_v2.md` | 数据集正式命名／迁移、v1分通道诊断、v2局部解码与边界目标、真实对照结果 |
| `map_temporal_codec.md` | 固定v2的同区域时序保真、支持增失／功率变化误报、历史窗口导出与实际核查结果 |
| `map_temporal_training.md` | 在v2上保留空间重建并增加差值／年龄／支持增失监督，validation选模与基线保留 |
| `map_embedding_bridge.md` | 128个连续地图嵌入、实际时间投影、文字嵌入后拼接与尚未训练的LLM接口边界 |
| `map_alignment_data.md` | 冻结原v2的两类区域问答、完整逐题审核、本地Qwen3-1.7B无梯度前向及未训练边界 |
| `map_sensor_alignment.md` | 独立因果点测／测向证据、冻结LLM的投影训练、完整答案预算、OOM及云端恢复协议 |
| `../configs/README.md` | 主配置与所有字段单位／限制；可选配置位于configs/optional |
| `../temp/debug_full_pipeline.md` | 12阶段逐步调试、IDE／pdb断点和调用记录 |
| `../outputs/README.md` | 当前现存数值验收、Git候选文件和已缺失历史证据 |
| `../datasets/README.md` | 用户RadioMapSeer、缺失夹具、原始数据与默认主流程的关系 |

## 可选工具与背景资料

| 文档 | 状态 |
|---|---|
| `../tools/README.md` | 可选离线脚本新路径、用途和约束；不参与在线决策 |
| `../configs/optional/README.md` | 4份外部／离线配置；不是主实验前提 |
| `sionna_rt_environment.md` | 可选离线独立引擎；保留历史实算说明，但旧输出当前缺失 |
| `uav_sensing_reading.md` | 感知论文及其与理想化传感器的适用边界，不是本项目参数标定 |
| `electromagnetic_research_plan.md` | 环境量、信息边界和后续研究背景；不维护另一套执行路线 |
| `实践项目申请书.md` | 长期正式申请书，不是已实现清单，不因本轮目录整理改写研究承诺 |
| `submissions/README.md` | 用户原始提交文件保管规则；不自动改写或上传原件 |

## 维护规则

状态在交接、阶段在路线、协议在实现、公式在仿真基础、命令在根README／云端说明、证据在outputs索引。不要重复复制simulate、保存另一份待办或把测试全绿／模型成功写在未核查的历史数字上。

当前只声称仿真。none、mock、local和回退分开记账；当前数值回放未调用LLM。原始输出和运行时哈希不因整理而改写，旧通信质量日志不作语义兼容。
