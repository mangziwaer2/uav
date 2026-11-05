# 随机场景、时序机载地图与小编解码器

更新：2026-11-04。按用户新授权，先实现离线数据与地图编解码，不启动LLM微调或RL；研究终点仍是新信息驱动的动态重规划。这里只声称合成电磁环境，不声称实测标定、源可唯一识别或任意设备迁移。

## 1. 已实现的边界

- `src/uav_em/map_learning/scenarios.py`：随机普通建筑、时变多源、几何与传播健全性检查。
- `data.py`：多行为飞行、当时可见地图、短原始观测、按场景分组、因果窗口、任务B训练题。
- `model.py`：小Transformer空间编码器、固定数量潜在tokens、坐标查询解码器和重建损失。
- `training.py`：训练、验证选checkpoint、一次最终test检查、checkpoint重载、时序token与PNG导出。
- `quality.py`：逐帧从公开测量重建belief，校验航迹、能量、时序标签、分组和基本覆盖。
- `partition.py`：离线按源数量分层划分整场景，复制到新目录并记录原始manifest哈希，不改观测或地图。
- `tools/generate_map_dataset.py`、`split_map_dataset.py`、`audit_map_dataset.py`、`train_map_autoencoder.py`、`inspect_map_encoding.py`：生成、分组、审核、训练和检查入口。

没有修改正式simulate、LLM提示词、在线行为或RRT*语义。采集复用正式传感器、belief、RRT*和折线推进，不冒称完整LLM任务实验。时序任务已生成监督数据，但尚未训练LLM理解这些任务；导出的向量也尚未投影到LLM嵌入空间。

## 2. 随机场景为什么可用，以及不能保证什么

生成器不是任意随机涂色，而是随机化同一个公开物理近似的参数：

- 已知矩形楼体棱柱，楼高10—55m、每楼损耗5—16dB；楼间预留间隔，不重叠，不覆盖起终点净空。
- UAV保持30m定高及原有净空，低楼可飞越，高楼避碰。
- 载频从0.9／2.4／5.8GHz取样，介电常数3—7、电导率0.001—0.02S/m。
- 每场景0—3个源，包括无源负样本、固定／恒速移动、延迟出现／提前消失；EIRP为−12—10dBm，移动速度0.2—1.5m/s，源高1.5m。
- 直达、每栋一次等效穿透与有限一次墙／地面Fresnel反射、多源mW叠加；点测和波束沿用同一传播实现。

验收不通过则重新采样，达到`max_attempts`后明确报错，不静默退回固定demo：

1. 精确检查起终点及净空后的连续栅格边，不只看格点。
2. 默认可飞格比例至少0.55，起点连通分量至少覆盖90%的可飞格，终点在该分量。
3. 给出一条公开几何BFS见证路线，检查其能耗不侵占余量，默认还要求恒速时间在时长内。
4. 移动源从活动开始到结束的整段轨迹在地图内且不穿地面楼体，不允许源钻墙。
5. 用少量离线功率查询检查有限数；不是全图真值质量认证，也不用于选采集航路。

这些保证几何／数值自洽和基本任务可行性，不保证现实城市统计、传播真实度、低干扰通道存在、任意规划预算成功或源可辨识。默认建筑形状仅矩形，尚未随机生成庭院、复杂街区或屋顶源。

`require_goal_reachable_in_time=false`仅适合明确的短片段数据／软件冒烟；报告保留是否满足时限，不可把短片段当完整任务可行性证据。普通训练配置保持true。

## 3. 不只飞一遍，也不要求先有聪明策略

同一场景独立重置belief，默认采集三类行为，每类可重复`flights_per_policy`次：

| 行为 | 数据覆盖目的 |
|---|---|
| goal | RRT*使用已有估计，周期重规划，偏好avoid_interference |
| coverage | 只按已知几何与已访问目标选新路点，再用RRT*求路；不是隐藏源引导 |
| revisit | 选择公开几何内的锚点，起点／锚点往返并悬停，提供同一区域重复观测与年龄变化 |

RRT*采样预算耗尽时可用公开几何BFS，明确记录`geometry_fallbacks`，不计为RRT*成功，更不是LLM成功。采集使用能量余量和时长限制，并检查实际经过的折线段。

源和建筑在同一场景各轨迹间保持一致，噪声与目标种子不同。轨迹可能经过强干扰，这是软代价下的有效训练状态，不是错误标签。编码器需要学会表达强干扰、弱干扰和未知，而非只学习最优路线看见的地图。

每条轨迹报告支持比例、支持的超参考栅格数、航程和回退次数；这些是覆盖诊断，不是暴露／安全指标。随机数不保证小数据集有足够的每类样本，正式训练前检查这些分布及evaluation_only中的源活动报告。24场景默认仅用于起步，不是充分数据量或泛化验收承诺。

2026-11-04新增正式首版配方`configs/map_encoding_full.json`：120场景，84／18／18按场景分组，每场景3种行为各2次，共720条轨迹；33×25格、20m间距、120s上限、2s步长，建筑数在3—10之间取样。时间窗4帧、目标间隔6s、ROI边长8格。不是把同一张demo重复720遍，也不承诺全图观测覆盖；全图未知状态是输入的重要部分。

## 4. 数据格式与信息边界

新目录结构：

```text
manifest.json                  版本、归一化、配置、分组与轨迹索引
episodes/<id>.npz              当时可见地图、时间、位置、能量、分辨率
episodes/<id>.json             公开几何／参数、原始点测与测向、采集行为日志
temporal_tasks.jsonl           同一区域时序变化监督
evaluation_only/<scene>.json   完整生成配置，含隐藏源，仅复现／离线审核
evaluation_only/*_report.json  几何验收及真实源数量／运动诊断，不送actor
```

模型Dataset只加载episodes的地图；不读取evaluation_only。train／validation／test按**场景**分组，同场景的全部行为和时间窗都属于同一个split。不按相邻帧随机切分。测试集不参与epoch选择，仅在验证选择完成后检查一次。

`maps`为float32的`[时间,6,行,列]`，按统一世界坐标对齐：

| 通道 | 归一化与语义 |
|---|---|
| power | 有支持估计：裁剪至[−120,−30]dBm后映射至[0,1]；未知置0，必须结合mask |
| supported | 有效观测支持且可飞为1；不是每格直接实测 |
| uncertainty | 启发式误差尺度／30dB，不是概率 |
| age | 有支持的信息年龄／35s；未知置0，由mask区分 |
| blocked | 当前飞行高度／净空下的公开几何禁入mask |
| building_height | 公开楼体总顶高／80m，包含可飞越的低楼 |

归一化常量是格式协议，写入manifest和checkpoint并在加载时核对，不针对测试集拟合。更换功率量程须升级协议，不静默复用旧权重。

当前采集不生成源假设，地图为局部观测支持外推，`source_prediction_encoded=false`。未来接入有模型源解释的RL数据前，应保留独立预测通道／假设版本，避免把模型改口当作实测场变化。不能用事后新假设重算并覆盖旧帧。

原始点测／方向在独立JSON保留，避免单张栅格丢失源推断证据。导出某个窗口时只给锚点之前的近期测量，绝不把完整轨迹的未来观测塞进输入。

## 5. 任务B的数据怎样得到

不需要人写标签，也不需要隐藏源坐标：

1. 从同一轨迹取过去4个目标时刻，按实际时间找不晚于目标的帧。
2. 缺失历史或重复命中标无效，零向量不是有效观测。
3. 选择同一世界坐标区域，混合当前UAV附近、早期位置附近和随机区域；区域边界为栅格`[left,top,right,bottom]`，右／下不含。
4. 比较最早有效帧与当前帧，输出新增支持、失去支持、当前年龄、共同有支持格的功率估计差。
5. 任一端未知的功率差为null，不把背景填充值算成干扰变化。

每题保存episode、anchor、各实际帧索引、有效mask、相对时间、ROI和target。采样间隔不一定等于实际时间间隔，必须读取offset_seconds。

`interpretation=belief_change_not_source_motion`：任务B教模型比较证据／估计，不能直接监督“源正在移动”。不同位置接收功率变化、信息过期和新增支持都会改变地图。源运动仍需多位置时序功率／测向和后续检验。

任务A可直接从同一npz裁剪指定帧／ROI构造，不必再采集一份数据。任务B作为后续LLM对齐重点；自主选ROI与动态行动由之后的任务训练／RL学习，不能把裁剪监督冒称自主决策。

## 6. 编解码与训练目标

当前改进版为`spatial_patch_decode_v2`，配置`configs/map_encoding_v2.json`。新增局部卷积编码和patch网格／PixelShuffle解码，仍为32×96瓶颈、无原图旁路；使用边界加权BCE＋Dice、训练专用同步翻转和验证早停。详细诊断、版本兼容、损失含义及实际结果见`map_codec_v2.md`。下段描述保留的v1基线，省略architecture仍为v1，不改变历史权重语义。

空间编码：分块卷积＋二维位置编码＋小Transformer，将特征自适应池化到固定空间锚点，再以锚点加可学习查询做交叉注意力，压缩为`[batch,K,D]`；默认K=32、D=96。空间锚点避免初始查询近似全图平均而丢失局部位置。解码：将固定锚点tokens双线性展开为局部特征，再结合世界栅格归一化坐标和交叉注意力重建6通道；只读瓶颈tokens，无原图旁路。每个数据集使用统一尺寸；模型支持不同尺寸调用，但暂未实现混合尺寸batch。checkpoint核对architecture标记，不静默重解释不同结构的权重。

训练功率／年龄只监督有支持格，对支持的强干扰格额外加权；support与blocked分别做类别平衡二元损失，uncertainty和height单独回归。不是隐藏真值补全，也不是连续向量转文本。

验证／test报告：支持格功率MAE、超参考支持格MAE、支持mask IoU、信息年龄MAE。无对应格时MAE为null，不伪造0误差。checkpoint按验证总损失选择。重建好不等于LLM能理解，也不等于规划更安全。

每帧32个tokens，4帧得到128个潜在位置；仍需投影到所选LLM嵌入维数并加入时间／位置语义。当前只导出tokens、valid、相对时间、位置、公开参数和近期原始观测，不是已经完成LLM适配。

## 7. 云端命令

配置补充约束：seed为非负整数；scene_count至少3，按场景约70%／15%／15%分组且每组至少1场景。columns／rows至少8、building_count非负、max_sources为0—3；resolution_m、horizon_seconds为有限正数；max_attempts为正整数，两个比例在(0,1]。

collection的step_seconds、dwell_seconds、window_spacing_seconds为正秒数；flights_per_policy、budget、task_stride、roi_size为正整数，window_frames至少2。task_stride按帧而非秒，roi_size按格而非米。每次实际步长会在到点／时限／能量边界缩短，时间窗必须使用真实时间。

encoder除architecture外各项为正整数，hidden_dim必须能被heads整除；architecture为显式结构版本字符串，patch_size为格、latent_tokens为每帧向量数。training的epochs为训练遍数上限、batch_size为批内地图帧数、frame_stride为每轨迹取帧间隔，均为正整数；learning_rate为有限正数，seed固定初始化与洗牌。v2的augment为布尔值，只对train作六通道同步翻转；patience为正整数轮数或null，连续该轮数验证loss未改善则停止。objective只接受balanced_v1／boundary_v2，定义不同不直接比较总loss；默认按结构选择。不支持恢复优化器继续训练，best.pt仅保存选择后的模型及协议元数据。

`scenario.min_building_count`省略时保持原先固定`building_count`；显式设置时为整数下界，单位栋，范围0至`building_count`。`--workers`为独立场景进程数1—32，只影响速度，不改变种子、数据和划分。每轨迹保留实际折线`transitions`及数据／元数据SHA256，全局时序题也有SHA256；分场景时序文件是生成中间产物，最终训练题入口是全局`temporal_tasks.jsonl`。

从根目录执行；先在云端独立环境安装适配该GPU的PyTorch，再安装依赖。不会自动下载LLM，也不要求Tesla具体型号相同。

```bash
python -m pip install -r requirements.txt -r requirements-map.txt
python tools/generate_map_dataset.py --config configs/map_encoding.json --scenes 24 --output outputs/interference_map_data_run01
python tools/train_map_autoencoder.py --config configs/map_encoding.json --dataset outputs/interference_map_data_run01 --device cuda --output outputs/interference_map_encoder_run01
python tools/inspect_map_encoding.py --checkpoint outputs/interference_map_encoder_run01/best.pt --episode outputs/interference_map_data_run01/episodes/scene_00000_revisit_000.npz --output outputs/interference_map_inspect_run01
```

正式首版改用以下流程，先审核成功再训练；输出目录同样必须为新目录：

```bash
python tools/generate_map_dataset.py --config configs/map_encoding_full.json --workers 4 --output outputs/interference_map_data_raw01
python tools/split_map_dataset.py --dataset outputs/interference_map_data_raw01 --seed 7 --output datasets/interference_onboard_maps_example
python tools/audit_map_dataset.py --dataset datasets/interference_onboard_maps_example --require-coverage
python tools/train_map_autoencoder.py --config configs/map_encoding_v2.json --dataset datasets/interference_onboard_maps_example --device cuda --require-quality --output outputs/interference_map_encoder_full01
```

审核默认新建`quality_report.json`，不覆盖；失败返回非零。它逐帧用当时点测重放belief，检查未知mask、公开几何、连续航迹碰撞／速度／能耗、时间窗和ROI标签、文件校验和及场景分组。`--require-coverage`额外要求每组有0／1／2／3源、移动源、三种行为、强弱／未知格、支持新增／失效／功率变化题，并拒绝跨组相同建筑布局及重复轨迹数组。源数只供离线覆盖审核，训练器不读取隐藏源配置。覆盖通过不是现实标定、源可辨识、全地图测遍或样本量足够的证明。

随机分组并不保证类别齐全：本轮120场景原始划分实际出现test无单源案例，严格审核拒绝，未放宽条件。正式流程加入`split_map_dataset.py`，按离线源数作比例分层，确保每个类别在train／validation／test各至少1场景，保持总数84／18／18；不足时明确报错。真值只用于离线分组，不作为模型输入或采集路径依据。分层仅改变split字段并重算相应哈希，轨迹npz完全不变，原始生成目录保留且不改写；新目录不复制旧质量报告，必须重新逐帧审核。仍不保证移动源或强弱场覆盖，交给严格审核检查。

`--require-quality`要求完整覆盖审核通过且manifest哈希未变，并将报告哈希写入训练配置；训练加载轨迹时再次核对其数据哈希。正式训练缓存已加载的轨迹地图以避免反复解压，内存量随数据规模增长；不是显存缓存。云端须额外传输数据目录，Git忽略outputs，不能假设拉取源码会得到数据和权重。

所有output必须不存在。run01只是示例，重跑换名或省略output自动时间戳，不能覆盖现有证据。生成数据不需要torch；训练与检查编码才需要。`--help`延迟导入训练后端，支持从其他工作目录调用绝对脚本路径。

先看数据manifest与temporal_tasks，再看训练history／metrics，最后看检查目录中的reconstruction.png、window.json和window_tokens.npz。PNG左侧是机载估计，不是全图真值；未知保留空白。

## 8. 后续RL是否联合更新编码器

路线以current_roadmap为准，建议先冻结已对齐编码器，再训练LLM决策，避免输入表示与策略同时漂移。新的RL策略确实会看到新分布，可以保留原始快照、混合旧／新策略数据，离线更新编码器并重新对齐；只有初期稳定后再考虑低学习率联合训练。

联合训练时保留旧数据重建／时序辅助目标、版本化编码器和投影层，并按对应版本重算缓存tokens，不能混用不同语义的旧向量。更聪明的飞行策略和更广的数据覆盖并不是同义词，仍需保留探索／往返数据。

当前已有连续地图投影／文字嵌入拼接、独立点测／测向及冻结主体的投影训练入口，见map_sensor_alignment.md。本地真实训练显存不足、优化器步数0，云端按CLOUD_ALIGNMENT_HANDOFF继续。没有LoRA、RL接口、PPO或源推断训练，也没有把地图编码接入在线simulate。保留编码保真与任务B限制；最终用实际到达、暴露、航程、能量和新信息引起的有效重规划评价，不奖励单纯多猜源或多重规划。

## 9. 不加载模型也能查看数据

当前唯一正式数据集命名为`../datasets/interference_onboard_maps_v1/`，先读其中DATASET_CARD.md；previews/goal和previews/revisit已保存两条原始样本的PNG／GIF／CSV。不要为了换v2编码器重新生成或重新划分数据；上面的example流程仅在确需新数据时另开新目录。原始npz是二进制数组，不适合文本编辑器；同名JSON、manifest和JSONL可以直接查看。实际采集记录见map_dataset_run_20261104，编解码改进见map_codec_v2。

其他轨迹可从manifest选id后运行，不需要torch或checkpoint，也不读取evaluation_only：

```bash
python tools/preview_map_dataset.py --dataset datasets/interference_onboard_maps_v1 --episode scene_00022_goal_000
```

工具默认新建时间戳目录；6通道PNG还原物理单位，4帧窗口只取锚点之前，同一区域时序题可对照JSON。`frame_grid.csv`为选定帧每格的坐标与6通道，未知功率／年龄留空；`observations.csv`是整条轨迹原始点测。GIF默认1倍仿真速度，`--stride`为正整数抽帧间隔（默认2，保留末帧），`--speed`为有限正倍速（默认1），末帧停2秒；`--no-gif`可只输出静态文件。选帧依据可见ROI变化，不是模型自主选ROI或代表性评价采样。预览只新增输出，不修改数据和分组。

## 10. 同一区域时序保真与窗口重建

`tools/audit_map_temporal_codec.py --dataset datasets/interference_onboard_maps_v1 --checkpoint <best.pt>`固定权重检查原始任务B，默认只用validation；核对哈希／窗口／原始标签，不读取隐藏源。报告支持新增／失效、功率变化事件及方向、共支持功率差／年龄差和分行为结果；无分母返回null，重叠任务不当独立样本。协议、实际结果及图例见map_temporal_codec.md。本轮发现功率变化误报，不能以单帧IoU高或总体差值均值小代替动态保真。

inspect_map_encoding新导出window_schema_version=2，增加reconstructed_window，形状[窗口位,6,行,列]，只来自各有效帧tokens解码；原anchor重建和其他字段保留。无效历史位为零但不是已测零干扰，必须使用valid。temporal_encoding标明逐帧空间结构，不能单凭此字段判断是否接受过时序监督。后续LLM只消费tokens和公开时间／位置等，不消费onboard_target等诊断原图作为压缩旁路。不同权重缓存不能混用。

2026-11-05在v2上增加配对辅助监督的候选已训练12轮、选第8轮，仍是逐帧空间结构而非时序Transformer；方法／混合结果见map_temporal_training.md。它与原v2权重不同，不得混用tokens。连续投影器只准备接口、尚未训练，见map_embedding_bridge.md。

## 11. 区域问答与真实模型接线

为推进对齐而非无限重训，显式固定原v2第39轮作为接口基线，保留其变化误报限制与时序候选。派生目录datasets/interference_map_alignment_v1含train 16,144题、validation 3,462题及冻结tokens，不导出test；输入／答案严格分离，完整审核已从原图逐题重算答案。两类任务为指定ROI六通道还原和最早有效帧／当前帧比较，不监督源位置或模型自主选ROI。

工具为build_map_alignment_dataset、audit_map_alignment_dataset与check_map_llm_bridge。实际本地Qwen3-1.7B一次无梯度forward已通过，输入444位置＝文字316＋地图128，维度2048；投影器随机初始化，没有LLM训练或完整回答结论。数据结构、边界、实际输出及命令见map_alignment_data.md。
