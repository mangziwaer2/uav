# 独立测向证据与地图投影训练

更新：2026-11-05。用户授权保留现有六通道编码器，独立加入点测／测向，并继续推进投影层训练；遇实际显存不足时停止并交接云端。本轮不更新LLM主体、不启动RL、不改在线simulate或传播公式。

## 1. 为什么不重训地图编码器

六通道机载地图继续由原空间v2第39轮编码，保持原权重、32×96瓶颈和现存缓存不变。输入分成两路：4帧机载地图经空间投影／实际时间层成为128个连续地图位置；历史点测／测向成为独立结构化文本，与文字问题一起输入冻结LLM。

测向是收到的角峰，不是已知源坐标。源假设仍由后续LLM决策提出，数值层用新到证据检查。多角峰不等于多源，反射可能指向墙；无检测不等于安全，没有测距。扫描共同方向误差具有时间相关性，波束宽度不是置信区间。不能用真源位置或三角定位结果替模型完成研究任务。

## 2. 正式独立证据包

`datasets/interference_sensor_evidence_v1/`配合`datasets/interference_map_alignment_v1/`使用，不复制或改写原始数据／地图tokens。612条train／validation轨迹，共30,843次点测及扫描，16,878次扫描有角峰。没有导出test轨迹、源位置、源标识、反射点或evaluation_only。所有16,144条train和3,462条validation问题的因果窗口已检查，见quality_report.json。

- manifest绑定原机载数据、地图对齐数据和编码器哈希；每份公开观测文件单独校验哈希。
- observations每episode存完整公开观测，读取时先按当前anchor截断，再取最近60秒全部记录；不采未来，不按角峰或答案选择历史，不省略窗口内的无检测记录。
- 报告时间窗、当前时间及被窗口排除的旧观测条数；60秒是公开输入配方，不表示全部历史，也不重置地图年龄。
- 每条含时间s、位置m、本地点测功率dBm，以及角峰方向／接收功率、波束宽度、误差标准差、误差相关时间、检测阈值。
- 原采集日志保存的是角峰摘要，不是全波束功率向量或阵元IQ；本轮不补造这些未保存的信息。
- directional=null表示没有扫描记录；存在扫描但peaks=[]表示扫描没有角峰。两者都不能被改成安全或零干扰。
- 提示词使用列式记录和去重的scan_settings，保留原始数值精度及全部角峰，不把浮点向量写成文字地图。方向以东为0度、逆时针、世界坐标。

先看samples/train.json或samples/validation.json：sensor_evidence是展开可读版，question是实际紧凑问题，地图插入点仍是占位符。样本不含答案或源位置。SensorAlignmentExamples只返回地图tokens／时间／有效位／问题，answer仍独立取出。

```bash
python tools/build_map_sensor_evidence.py --dataset datasets/interference_map_alignment_v1 --source datasets/interference_onboard_maps_v1 --output datasets/interference_sensor_evidence_v1
```

示例对应现存目录；重建须换不存在的新目录。只读原数据的train／validation公开元数据及既有token时间，不打开原图、test轨迹或隐藏真值。

## 3. 当前训练学什么

仍是两类基础对齐题：指定ROI六通道还原、同ROI历史变化比较。加入测向不等于已有源定位／运动监督，也不等于学会自主选ROI。区域监督仍来自当时原始机载估计，不是解码重建或完整环境真值。

实际训练：冻结LLM所有权重，地图编码器不加载、直接用已冻结缓存；只优化MapEmbeddingProjector的空间投影及时间MLP。使用因果teacher-forcing：

1. 用真实聊天模板关闭thinking，地图占位符替换为连续嵌入，独立观测以文本进入user问题。
2. 紧凑JSON答案加EOS接在assistant前缀之后；这是训练目标，不是把标准答案放进user问题。因果掩码只允许预测当前位置时读取此前答案token。
3. 所有问题、地图和无效位置的labels=-100；只对答案及EOS计算标准下一token交叉熵，保留模型内部的单次shift。
4. backward必须穿过冻结LLM回到投影／时间层，不能对整段LLM用no_grad。非重入梯度检查点、SDPA、use_cache=false用于控制显存。
5. validation只算固定等间隔样本的teacher-forced loss，记录索引；没有自由生成验收，低loss不能当作地图理解、源定位或规划收益。

first_training_example.json把实际问题与supervision_only_answer分开供人检查；不是在线推理的标准答案输入。LLM保持冻结也不免除激活／反向传播和长词表logits占用。

## 4. 配方与预算

`configs/map_alignment.json`逐字段：

| 字段 | 含义、单位与限制 |
|---|---|
| epochs | 正整数，完整训练集遍历次数，当前1 |
| gradient_accumulation | 正整数，microbatch固定1，累积8个样本后更新；末组按实际数量归一化，不降低单样本显存 |
| learning_rate | 有限正数，AdamW学习率，当前0.0002 |
| max_positions | 正整数，含完整问题、地图padding、完整答案及EOS，当前8192；不超过具体模型上下文，不截断 |
| history_seconds | 有限正数，测向／点测因果历史窗长度s，当前60；不是探测半径或地图有效期 |
| checkpoint_every | 正整数，每25个已完成优化器步保存last.pt；初始化及正常结束也保存 |
| validation_examples | 正整数，最多16条validation题等间隔检查；不是全量生成评价 |
| max_steps | null表示按epochs完成；正整数可做短训练，但step_limit_reached不等于完成全数据训练 |
| seed | 非负整数，当前7，按seed+epoch生成确定性shuffle次序 |

完整长度审核已通过：train最长7,885位置（问题＋地图6,295、答案1,590），validation最长7,808；两组中位数约3,738／3,725.5，95分位6,539／6,684。所有19,606题均在8,192以内，没有截断。无测向时的444位置forward不能据此估计本轮长答案训练显存。

CUDA自动选支持时BF16、否则FP16；投影参数保持FP32，FP16使用GradScaler，CPU显式选择时为FP32。不量化、不自动CPU卸载、不减少历史／答案，不为挤进本机而更换题目。不同GPU／精度可以恢复相同数据与权重身份，但不保证逐位相同；BF16转FP16时旧checkpoint无scaler状态则初始化新scaler。

```bash
python tools/train_map_alignment.py --dataset datasets/interference_map_alignment_v1 --evidence-dataset datasets/interference_sensor_evidence_v1 --model-path models/Qwen3-1.7B --config configs/map_alignment.json --device cuda --budget-only
python tools/train_map_alignment.py --dataset datasets/interference_map_alignment_v1 --evidence-dataset datasets/interference_sensor_evidence_v1 --model-path models/Qwen3-1.7B --config configs/map_alignment.json --device cuda
```

默认分别创建新outputs/interference_map_alignment_training_时间戳目录，不能覆盖。长度审核先完成，再加载真实LLM。依赖使用requirements-map.txt与requirements-llm.txt；源码不会下载权重或启用remote code。

## 5. OOM与云端恢复协议

CUDA OOM专门记录为cuda_oom并以非零退出码2结束，不反复重试、不截断或退回CPU。oom.json包含完整错误栈、具体样本、forward／backward阶段及位置数；report记录峰值显存、实际完成步数与已保存步数。其他错误不能冒充OOM。

last.pt只含投影／时间层、优化器、scaler、采样游标、随机状态、设置及数据／模型身份，不含完整LLM或地图编码器。若在第一次更新前失败，它是step=0初始化恢复点，不能称为已训练权重。OOM中间未保存的更新或累积梯度不算可恢复进度；以saved_optimizer_steps和checkpoint为准。

只继续训练时需完整地图对齐数据目录、独立证据目录、同一具体LLM的完整本地权重／tokenizer，以及选定训练输出与last.pt。原始机载数据及原v2权重仅在重建／完整重审核时额外需要。不要复制Windows虚拟环境。新agent先读CLOUD_HANDOFF.md，使用新输出目录：

```bash
python tools/train_map_alignment.py --dataset datasets/interference_map_alignment_v1 --evidence-dataset datasets/interference_sensor_evidence_v1 --model-path /path/to/Qwen3-1.7B --config configs/map_alignment.json --device cuda --resume /path/to/previous_run/last.pt
```

恢复绑定地图／证据manifest、具体LLM文件哈希、投影结构及训练设置；允许改变max_steps、保存间隔和validation样本数，不允许静默改变历史窗、学习率、epoch或数据。模型位置可迁移，内容不能换。后续先完成投影对齐，再验证自由生成／证据利用，最后按路线接源假设及动态重规划；当前无RL实现。

## 6. 本次实际结果

完整预算预检在outputs/interference_map_alignment_training_20261105_134844_225657通过。真实训练在outputs/interference_map_alignment_training_20261105_135229_972149遇6GB CUDA OOM，优化器步数0、sample_cursor=0，last.pt仅初始化；已停止本机真实重试。该次安装版本以AcceleratorError报告OOM，旧捕获器未覆盖，因此没有自动oom.json／report.json；事后说明及终端摘录保留在该目录，不能冒充完整运行时诊断。最新异常处理已修正并用小模型验证，详细云端接手见CLOUD_ALIGNMENT_HANDOFF.md。

最终全量测试448通过、2失败（122.11s）；新增13项测向／训练测试通过，包括真实小Qwen3反向、LLM冻结、确定性保存恢复、两类OOM异常和空scaler跨精度恢复。失败仍是datasets/example_power_grid.json与datasets/example_buildings_local.geojson两个既有可选夹具缺失，不修补、不跳过。基础命令另加--basetemp temp/interference_sensor_alignment_full_20261105_02。
