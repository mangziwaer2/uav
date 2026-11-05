# Sionna RT离线干扰真值

**状态更新（2026-11-03）：** 按用户最新要求，本接口降为可选离线核查，不是默认主环境或后续LLM实验前提。主流程使用普通建筑中的本地时变干扰源及有限路径公式，见`simulation_foundation.md`；无需运行下文下载／安装步骤。适配代码、本机安装和缓存保留；此前生成输出当前不在工作区，不当作现存证据，也不从历史恢复。本文其余部分记录可选后端能力与历史验收，不代表当前默认任务。离线脚本现位于tools，配置位于configs/optional。

## 1. 目的与信息边界

优先补足可复现的电磁干扰真值，不把城市地图收集、Unity可视化或全波求解另设为研究主线。使用独立传播引擎，避免环境生成与LLM源拟合永远使用同一个简化公式。这里只声称射线追踪仿真，不声称实测、设备故障预测或任意平台泛化。

```text
Sionna RT内置场景＋独立源配置
  → 离线射线追踪的每源接收功率W
  → 非相干叠加＋一次背景 → 总干扰dBm栅格
  → 现有ElectromagneticWorld在实际位置取样并加测量噪声
  → belief → LLM咨询／补测／RRT* → 新观测
```

完整场、源位置和功率仅供环境生成及离线评价；不进入LLM上下文。建筑几何是公开先验。导出配置不包含发射机清单，清单仅保存在离线`manifest.json`。本轮不训练LLM、不执行生成代码，也不替换现有simulate。

## 2. 独立环境与资源

可选依赖固定在`requirements-rt.txt`，不加入基础依赖、不更改已有LLM环境。**引擎环境使用Python 3.11以上，推荐3.12**：2.2.0安装元数据虽写>=3.10，实际源码直接导入`typing.Self`，3.10导入失败，不能以pip安装成功代替运行验收。项目原有基础／LLM环境仍可用3.10，不补丁修改第三方源码。Sionna RT安装包自带本轮使用的`simple_street_canyon`场景，不需要下载整座城市或模型权重。保留上游许可证与通知，生成清单记录版本、场景和配置哈希。

Windows项目根目录：

```powershell
E:\Miniforge\python.exe -m venv temp\sionna_rt_py312
temp\sionna_rt_py312\Scripts\python.exe -m pip install -r requirements-rt.txt
temp\sionna_rt_py312\Scripts\python.exe -X utf8 tools\generate_interference_rt.py
```

Linux／云端建议另建环境，不搬运Windows的venv：

```bash
python3.12 -m venv temp/sionna_rt_venv
temp/sionna_rt_venv/bin/python -m pip install -r requirements-rt.txt
temp/sionna_rt_venv/bin/python tools/generate_interference_rt.py
```

默认显式选择CUDA后端；CPU后端需LLVM支持，不能把CPU运行计为GPU性能。失败直接报错，不静默退回当前合成场。实际驱动和运行时兼容性以本机运行记录为准。

## 3. 两份配置不能混用

`configs/optional/sionna_rt_demo.json`是导出模板，不是已经生成真值的正式实验配置。`configs/optional/sionna_rt_sources.json`仅供离线生成器读取，不能传给simulate或LLM。

| 字段 | 中文含义、单位与限制 |
|---|---|
| `scene_name` | 当前只接收经过检查的官方`simple_street_canyon`，不隐式导入任意网格 |
| `origin_xy_m` | 项目坐标原点在引擎场景中的二维位置，m；源和建筑只平移一次 |
| `bandwidth_hz` | 观测频带宽度Hz，正数；全部源功率均在此频带内，不自动按带宽换算 |
| `samples_per_tx` | 每个源发射的射线数，整数1至1亿；默认100万，小值只用于烟雾检查 |
| `max_depth` | 最大传播交互深度，整数0至10；默认3，仅启用声明的传播机制 |
| `seed` | 射线采样随机种子，非负整数；与飞行／测量随机种子分别记录 |
| `transmitters` | 1至8个独立同频带干扰发射机，固定位置和功率 |
| `position_m` | 项目局部`[x,y,z]`，m，z相对场景地面；源不得放在不透明楼体内 |
| `power_dbm` | 该观测频带内总发射功率dBm，支持−100至60；不是功率谱密度 |

飞行高度和载频取自模板，不硬改公开数据高度。本轮天线为单阵元、全向、垂直极化；公开几何从引擎网格提取，只接受验证为轴对齐矩形棱柱的楼体，遇到其他网格报错，不偷偷用包围盒代替。地面仅作为传播面，不作为禁飞楼体。

官方街谷地面并非恰好z=0；生成器读取平坦地面网格作为统一高度基准，对建筑、源和接收切片一致换算，并在清单保存`scene_ground_z_m`。不能只平移建筑而漏改接收高度。

默认任务为180m×120m、30m定高、两源5m高，带内发射功率−8／−11dBm，软参考−80dBm。源参数是显式研究设定，不是官方场景附带的实测干扰设备参数；功率针对所声明的1MHz观测频带，不代表宽带发射机的整机总功率。不按LLM表现调参，也不把单个场景到达作为收益证据。

## 4. 输出与接回完整闭环

生成器默认创建新的`outputs/interference_rt_时间戳/`，也可显式提供不存在的目录。不会覆盖已有证据。

| 文件 | 含义 |
|---|---|
| `simulation.json` | 真正供现有调试／实验入口读取的配置，引用同目录功率图 |
| `power_grid.npz` | 五键总干扰栅格；不含源参数，含背景，按`[y,x]`排列 |
| `source_received_watts.npz` | 每源接收功率，仅用于离线核查，单位W，不进决策 |
| `manifest.json` | 离线源清单、版本、传播设置、哈希、功率范围、零信号格统计与近似 |

生成后，回到原有基础环境运行：

```powershell
E:\Miniforge\envs\uav\python.exe temp\debug_full_pipeline.py --config outputs\interference_rt_实际时间戳\simulation.json --advisor none --plot --gif
```

先确认无模型完整流程；之后才使用现有真实LLM入口。数值到达、漂亮GIF和合法JSON均不是LLM收益证据。

## 5. 功率、空间与研究近似

引擎RSS输出为W。每源先转为mW，再非相干相加，最后加一次`10^(background_dbm/10)`并取`10log10`。不直接相加dBm，不把RSS误当路径损耗，也不把接收机热噪声自动当成干扰。源之间独立同频带；源内传播交互按引擎无线电地图定义处理。

每次生成先运行空场直达核查：0dBm各向同性源、2.4GHz默认载频、单元平均RSS与独立Friis公式数值积分比较，容限1dB，结果写入清单。核查失败不生成替代场；这是单位／引擎接线检查，不是实测标定或复杂城市传播精度保证。

本轮使用无线电地图的**单元平均功率**，不是亚波长尺度的相干点场。导出前逐格检查实际单元中心，拒绝转置、纵轴翻转与半格偏移；将单元平均值作为对应节点的局部场近似，飞行中在毫瓦域插值。栅格覆盖完整飞行边界，边缘单元覆盖额外半格；不做边界外推。

明确未实现：

- 全波求解、精确设备天线／接收带宽标定、频谱积分；当前是单载频的平坦频带近似。
- 本轮只启用直达与镜面反射，不启用漫反射、折射／穿透和绕射；遮蔽区域不等于现实绝对无信号。
- 有限射线数可能漏掉传播路径，零信号格加背景不意味着已证明安全；正式场景需要检查射线数及网格精度敏感性。
- 只导出静态单高度切片；源运动、功率开关和时间序列尚未接入。不能平移一张功率图伪造移动源。
- `sensor.mode=power_only`：不从真实源坐标伪造测向扇区，不提供圆盘范围的真值扫描。空间覆盖来自飞行观测与后续推断。
- LLM源假设的数值检验仍使用项目简化前向模型，与射线追踪有模型失配。拒绝拟合不自动说明LLM位置判断错误。

## 6. 本轮运行验收（2026-11-03）

- 实际环境：Python 3.12.12、Sionna RT 2.2.0、Mitsuba 3.9.1、Dr.Jit 1.5.0、NumPy 2.5.4、SciPy 1.18.1、Matplotlib 3.11.2；RTX 3060 Laptop，驱动610.62，CUDA后端。独立环境`pip check`通过，原基础环境未安装Sionna。
- 真值输出：`outputs/interference_rt_20261103_124744_448016/`；自由空间单元平均功率误差+0.0004126dB；每源100万射线、最多3次交互，总功率−110至−75.1507dBm。源1／源2分别有278／461格零信号；包含楼体和遮蔽以及有限采样影响，不把零值当现实安全证据。
- 完整闭环：`outputs/interference_debug_none_20261103_124826_884769/`，无LLM／无回退到达，160.394m、11.693s、超参考暴露7.716s，12次点测、2次决策、0测向。PNG目视检查通过，GIF生成，15条真实飞行子段几何复核通过，源参数未进入决策上下文；源代码与输出哈希匹配。
- 适配测试19项通过；全量308通过、2失败，原因是本轮开始前已缺失的`datasets/example_power_grid.json`、`datasets/example_buildings_local.geojson`。本轮不擅自恢复或改掉旧测试，不声称全量全绿。
- 本机LLVM初始化探测有警告，但所选CUDA后端实际通过上述核查并产生数据；CPU后端未验证。Matplotlib缓存固定在项目`temp/sionna_mpl_cache`，不要求写用户目录。
- 这里只验收资源、功率语义、坐标、信息隔离和闭环；未运行真实LLM、未证明路径优化收益，也未完成RT采样收敛性或实测验证。

## 7. 一手资料

- Sionna RT官方代码与安装说明：<https://github.com/NVlabs/sionna-rt>
- 官方入门教程、场景与多源无线电地图：<https://nvlabs.github.io/sionna/rt/tutorials/Introduction.html>
- 官方无线电地图API及RSS单位：<https://nvlabs.github.io/sionna/rt/api/radio_maps.html>
- NASA无人机沿航路接收功率及地面干扰源仿真先例：<https://ntrs.nasa.gov/citations/20210020364>

传播后端与更精细地图是环境支撑，不要求先完成Unity工程、训练无线电地图模型或设备标定，才能继续LLM辅助决策研究。
