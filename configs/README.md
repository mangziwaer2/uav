# 当前配置说明

默认`demo.json`，当前为干扰功率语义，不兼容旧`radio`通信质量配置。省略分组使用默认值；未知字段、重复键及非法值拒绝。字段在`src/uav_em/config.py`含中文注释。

2026-11-03主流程回到本地普通建筑＋时变干扰源。`demo.json`与`demo_25d.json`显式采用`single_reflection`，无需Sionna、外部数据或GPU；`sionna_rt_*.json`只是可选离线核查。省略传播模式仍用旧`log_distance`，防止旧配置／日志被悄悄重新解释。

## 地图与飞行器

可选配置统一放在`optional/`：Sionna RT使用`optional/sionna_rt_demo.json`模板与`optional/sionna_rt_sources.json`隐藏源设置，入口为`../tools/generate_interference_rt.py`，字段和完整命令见`../docs/sionna_rt_environment.md`。不要把模板直接当成已生成干扰场的任务；飞行入口读取导出目录的`simulation.json`，不读取源清单。

定高2.5D入口`demo_25d.json`，`demo.json`为平面矩形主配置，两者均使用新反射公式。外部静态图示例为`optional/demo_power_grid.json`，其数据夹具当前缺失，不能直接运行；导入模板为`optional/import_25d_template.json`。完整格式见`../docs/environment_25d.md`，不把示例称为真实设备／城市数据。

| 字段 | 含义／单位 |
|---|---|
| `seed` | 环境噪声与规划随机种子，非负整数 |
| `map.columns`、`rows` | 网格节点列／行数，至少2 |
| `map.resolution_m` | 间距m；边界为`(数量−1)×间距` |
| `map.start_m`、`goal_m` | 二维起终点，m |
| `map.obstacles_m` | 已知矩形`[xmin,ymin,xmax,ymax]`，m |
| `map.geometry_mode` | `planar`默认：所有建筑轮廓禁入；`fixed_altitude`：按楼高筛选障碍，矩形仍为显式平面禁入区 |
| `map.buildings` | 多边形楼体，含唯一id、米制footprint_m、正height_m；可选holes_m、非负base_m／loss_db及height_source |
| `map.provenance` | 地图来源、许可、坐标和高度假设等记录；非电磁观测 |
| `aircraft.speed_mps` | 恒定沿航路速度，m/s；不模拟转弯加速度 |
| `aircraft.altitude_m` | 定高m，有限正数，默认30；用于传播距离，不开放高度优化 |
| `aircraft.vertical_clearance_m` | 上下净空m，非负、默认5；楼体竖直区间扩净空后与机高相交就禁入，边界相等也禁止 |
| `flight_power_w`、`hover_power_w` | 飞行／决策悬停功率，W |
| `initial_energy_wh`、`reserve_energy_wh` | 初始／预留能量，Wh；普通任务不消耗预留 |
| `clearance_m` | 几何障碍膨胀裕量，m；不是干扰安全距离 |

## 仿真与预算

`electromagnetic.power_grid`默认null；设为静态总干扰dBm栅格字典或`{"path":"相对配置路径"}`时，必须`fields=[]`与`sensor.mode=power_only`。接受JSON／无pickle的NPZ，x_m/y_m严格递增且覆盖全图、power_dbm按[y,x]排列，机高与载频匹配；禁止nodata、外推或把路径损耗直接当总干扰。载入后嵌入评价配置，完整栅格不进入模型；详细协议见环境说明。

| `simulation`字段 | 含义 |
|---|---|
| `step_seconds` | 最大飞行推进间隔，s；每轮同步接收当前位置总功率及方向波束 |
| `review_seconds` | 常规复核间隔，s |
| `minimum_review_seconds` | 观测偏差触发咨询的最小间隔，s；不是学习门控 |
| `innovation_trigger_db` | 新测量相对更新前地图估计或旧源预测的绝对差达到此值时可提前咨询，dB；不是地图更新开关或远处变化预知 |
| `warning_window_seconds` | 趋势窗口s，有限正数，默认5；累计增强还须有至少3次点测 |
| `warning_rise_db` | 点测／测向相对上次复核的增强门槛dB，有限正数，默认6；仍受最小复核间隔约束 |
| `max_seconds` | 最大任务仿真时间，s |
| `decision_delay_seconds` | 固定虚拟决策悬停，s；实际推理时间另记，不推进时钟 |
| `max_advisor_calls` | 整个任务的模型调用上限，非负整数，默认24次；包含源假设细化轮，耗尽后数值回退并记录 |
| `source_refinement_rounds` | 同一决策点源假设最多咨询轮数，整数至少1，默认3；每轮计入总调用上限，观测集和仿真时刻不变 |
| `astar_expansion_budget` | 可选A星搜索（A* Search）的节点扩展上限 |
| `rrt_attempt_budget` | 主规划器渐近最优快速探索随机树（Rapidly-exploring Random Tree Star，RRT*）的样本提议上限，包含无效提议 |
| `max_probes` | 任务允许主动补测次数 |
| `probe_budget_seconds` | 每次补测航行上限，s；另计搜索和决策悬停 |

## 电磁干扰场

| `electromagnetic`字段 | 含义 |
|---|---|
| `background_dbm` | 均匀背景功率，dBm；也是未知处的计算先验，不代表已测低干扰 |
| `measurement_noise_db` | 当前点测量的独立高斯噪声标准差，dB |
| `interference_reference_dbm` | 干扰软参考值，dBm；默认−80，须高于背景，超过时增加代价；回放同时用它画等值边界和估计区域，不是禁飞或平台失效阈值 |
| `fields` | 隐藏合成源脚本，可为空；不进入模型上下文 |
| `fields[].center_m` | 源在活动开始时的位置，m |
| `carrier_frequency_hz` | 载频Hz，有限正数，默认2.4GHz；所有源在同一观测频带 |
| `reference_distance_m` | 参考距离m，正数且至少一个波长，默认1；夹紧不是近场求解 |
| `propagation_mode` | `single_reflection`：自由空间＋等效穿楼＋有效一次墙／地面反射；`log_distance`：旧经验模式；省略时为后者 |
| `path_loss_exponent` | 经验指数[2,6]，省略为2.2；反射模式必须显式为2，避免重复距离衰减 |
| `relative_permittivity` | 墙／地面统一相对介电常数，无量纲、有限且>=1，默认4；未标定 |
| `conductivity_s_per_m` | 墙／地面统一电导率S/m，有限非负、默认0.005；与介电常数共同计算Fresnel功率系数 |
| `obstacle_height_m` | 所有矩形统一棱柱高度m，非负，默认20 |
| `obstacle_loss_db` | 每穿过一个棱柱扣除的功率dB，非负，默认6，不是每米损耗 |
| `fields[].eirp_dbm` | 默认传播源的频带内EIRP，dBm，有限且在[−150,80]；不是中心接收功率 |
| `fields[].height_m` | 默认传播源相对地面高度m，有限非负 |
| `sigma_m` | 高斯空间功率尺度，m；不是干扰区域的硬边界 |
| `peak_dbm` | 单个源在中心贡献的功率，dBm；总值还包括背景和其他源 |
| `start_seconds`、`end_seconds` | 活动区间`[start,end)`，s |
| `velocity_mps` | 从活动开始计的源中心恒定速度，m/s |

所有功率在同一抽象观测频带解释，先转毫瓦叠加再取对数；无服务站、期望信号或链路质量模型。dBm越大干扰越强，例如−60强于−90。固定源设速度为零且活动区间覆盖任务；需要动态时设置移动、生灭脚本。无人机不能直接获得脚本，不能把动态源等价为天气平移。

### 默认传播场与历史高斯场

2026-10-30起，800×600m默认场使用−2／−5dBm传播源、高度1.5m，机高30m。第一源固定，第二源在24s出现并向南移动；默认值与`demo.json`一致。旧35／30m及85／65m高斯场属于历史实验，不直接用于新环境收益对比。显式高斯源仍接受原六字段格式，不能与EIRP／高度字段混写；它不应用棱柱传播损耗。详细公式、依据和未实现物理项见`../docs/simulation_foundation.md`。

`tests/test_demo_scenario.py`离线检查起终点、强干扰覆盖比例及一条低干扰绕行折线：每3s检查一次、沿折线约每5m取样。这只是场景可绕性的离散检查，不是连续时间安全证明；测试折线和源真值绝不传给模型或规划器。缩小场不等于实现避让：当前干扰仍是软代价，未知区域和局部外推也可能导致穿越。不得把参考等值线称为已标定的通信中断边界。

## 机载范围感知

默认假设360°阵列同步形成波束，不是空间圆盘扫描或直接测距；0°向东、90°向北。所有源先在每束叠加，再加噪声和检测，源数不作为观测输出。当前是直达路径主导近似，建筑只衰减、不产生反射来向，不是具体设备标定模型。

| `sensor`字段 | 含义／单位及限制 |
|---|---|
| `mode` | `power_bearing`为本地点功率＋测向阵列，默认；`power_only`仅测本地点功率、无方向快照，两种模式不改变点测随机数 |
| `azimuth_bins` | 方位波束数，整数[4,360]，默认24，即15°间隔 |
| `beamwidth_deg` | 半功率全波束宽度°，(0,180]，默认45；不是定位误差置信区间 |
| `front_back_db` | 方向抑制上限dB，有限非负，默认25；归一化主瓣峰值为0dB |
| `detection_threshold_dbm` | 接收灵敏度dBm，有限值，默认−94；还须超过背景与角对比门槛，不是−80dBm风险参考 |
| `minimum_excess_db` | 超背景最小增量dB，有限正数，默认3 |
| `minimum_contrast_db` | 角峰超过扫描功率中位数的最小增量dB，有限正数，默认3 |
| `noise_db` | 每束接收功率噪声标准差dB，有限非负，默认1 |
| `bearing_error_deg` | 扫描共同方位误差稳态标准差°，[0,45]，默认3，不是置信区间 |
| `calibration_correlation_seconds` | 方位误差相关时间s，有限非负，默认5；一阶Gauss–Markov过程，0为独立快照误差；不是设备已标定规格 |

没有固定探测半径：同距离是否可检测取决于EIRP、传播、遮挡和噪声。方位峰可能混叠、漏检或虚警；环形角距离小于`beamwidth_deg`的邻近峰按功率从强到弱去重，不改变原始波束读数。参数是研究近似，不是已标定设备规格；不能通过重复相关扫描声称消除了全部误差。论文及适用边界见`../docs/uav_sensing_reading.md`。

## 局部估计

| `belief`字段 | 含义 |
|---|---|
| `support_radius_m` | 点测量外推最大距离m，默认40；不是直接感知半径，测向不扩大地图支持 |
| `max_age_seconds` | 当前估计支持的有效测量年龄，s；时变场旧信息会失效 |
| `measured_error_db`、`unknown_error_db` | 测点误差下限／未知位置的启发式误差尺度，dB |
| `error_per_meter_db` | 外推每米增加的误差，dB/m |
| `error_per_second_db` | 每老化一秒增加的误差，dB/s |
| `source_max_fit_rmse_db` | 源假设有效观测窗口的年龄加权RMSE上限，dB，有限正数，默认8；不是定位误差或安全阈值 |
| `source_count_penalty_db2` | 拟合目标中每增加一个源的复杂度惩罚，dB²，有限非负数，默认1；不保证正确识别源数 |
| `source_bearing_weight_db2` | 归一化方向Huber损失的联合目标权重，dB²，有限正数，默认9；不是概率或设备规格 |
| `source_max_bearing_rmse_deg` | 候选源方向RMSE接受／停用与新观测复核容限，°，(0,180]，默认45；不是定位精度或置信区间 |

无有效源假设时维持局部常值外推；有效时使用源预测加局部观测残差。预测覆盖未测区域也不增加支持或降低未知误差。源拟合窗口和接受后的有效期均复用`max_age_seconds`，原始带时序点测最多128条，模型输入最多32条；源参数协议见`../docs/first_version.md`第2节。源参数来自LLM假设，不读取上方`electromagnetic.fields`真值脚本。

默认源假设使用中心、EIRP、源高和速度，不再要求用高斯尺度解释传播场。数值层仅前向检验功率和合成角峰，方向相关观测按时间块平均；不自动求源坐标。传播参数作为公开研究假设提供给模型，源高度和EIRP真值仍隐藏。方向容限、权重和时间块均未由实测校准，`power_only`模式不使用方向项。

默认规划为RRT*，用命令行`--planner astar`选择网格对照；`balanced`、`avoid_interference`、`explore`是预设软代价，不允许模型放宽几何和能量。当前地图是定高二维，含简化棱柱损耗，未模拟实际频率响应、多径／绕射和平台易损性。不同实验使用独立输出目录。
# 独立地图学习配置

`map_encoding.json`为小规模起步配置；`map_encoding_full.json`为正式首版120场景／720轨迹、30轮空间编解码训练配方。两者供tools下的generate_map_dataset／train_map_autoencoder读取，不是ProjectConfig，不传给run_experiment。正式数据先使用audit_map_dataset的`--require-coverage`审核，训练加`--require-quality`。字段中文含义、单位、限制、默认范围和独立云端命令见`../docs/map_encoding.md`；不含LLM微调或RL配置。

当前改进配置为`map_encoding_v2.json`，复用`datasets/interference_onboard_maps_v1/`，不重新生成数据。architecture显式为spatial_patch_decode_v2，仍为32×96瓶颈；boundary_v2为边界BCE＋Dice目标，augment=true仅对train同步翻转，epochs=40为轮数上限，patience=8为验证早停等待轮数。含义、兼容与结果见`../docs/map_codec_v2.md`；v1字段省略的历史配置语义不变。

`map_encoding_temporal.json`仅供train_map_temporal_autoencoder，不是生成数据的完整配置。它从已有boundary_v2权重微调：epochs=12、batch_size=16、learning_rate=0.0001、seed=7、patience=4；change_db=3为训练变化分层及功率差分组阈值dB，不是安全阈值。power_weight=0.5、age_weight=0.1、transition_weight=0.1为辅助损失权重，有限正数；augment只在train同步翻转两帧及ROI。字段限制、固定空间退化容限与validation选模见`../docs/map_temporal_training.md`。不改变单帧瓶颈，不运行test或LLM训练。

`map_alignment.json`仅供tools/train_map_alignment.py：冻结LLM和地图编码器，只训练空间投影／时间MLP，独立加入已有点测／测向。epochs=1；microbatch固定1、gradient_accumulation=8；learning_rate=0.0002；max_positions=8192包含完整问题、128地图位置、完整JSON答案和EOS，不截断；history_seconds=60为因果传感器历史窗s，不是探测范围；checkpoint_every=25按已完成优化器步保存；validation_examples=16只检查teacher-forced loss；max_steps=null按完整epoch，正整数仅作短训练；seed=7为非负随机种子。所有计数字段须为正整数，学习率及时间窗须有限正数。完整含义、显存策略与恢复限制见`../docs/map_sensor_alignment.md`，不用于ProjectConfig、地图编码器重训或RL。
