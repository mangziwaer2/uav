# 当前配置说明

默认`demo.json`，当前为干扰功率语义，不兼容旧`radio`通信质量配置。省略分组使用默认值；未知字段、重复键及非法值拒绝。字段在`src/uav_em/config.py`含中文注释。

## 地图与飞行器

| 字段 | 含义／单位 |
|---|---|
| `seed` | 环境噪声与规划随机种子，非负整数 |
| `map.columns`、`rows` | 网格节点列／行数，至少2 |
| `map.resolution_m` | 间距m；边界为`(数量−1)×间距` |
| `map.start_m`、`goal_m` | 二维起终点，m |
| `map.obstacles_m` | 已知矩形`[xmin,ymin,xmax,ymax]`，m |
| `aircraft.speed_mps` | 恒定沿航路速度，m/s；不模拟转弯加速度 |
| `aircraft.altitude_m` | 定高m，有限正数，默认30；用于传播距离，不开放高度优化 |
| `flight_power_w`、`hover_power_w` | 飞行／决策悬停功率，W |
| `initial_energy_wh`、`reserve_energy_wh` | 初始／预留能量，Wh；普通任务不消耗预留 |
| `clearance_m` | 几何障碍膨胀裕量，m；不是干扰安全距离 |

## 仿真与预算

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
| `path_loss_exponent` | 经验距离损耗指数[2,6]，默认2.2，未由实测标定 |
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
