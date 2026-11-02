# 当前输出说明

## GitHub迁移范围

本次仅精选`interference_source_local_debug_20261102/`、`interference_debug_numeric_20261102/`、`interference_source_direction_mock_20261031_193820/`的JSON／JSONL／文本／PNG入库，供云端复核当前传播协议、调试流程及源拒绝／修订夹具。它们不是在线输入数据库；`evaluation_only_config.json`和episode内评价配置含真值，只用于仿真／回放，绝不能直接送给LLM。

GIF及下方其余历史目录仅本地保留，不随克隆传输；可从已入库episode离线重建当前回放。新的`outputs/interference_.../`仍默认忽略，提交前需人工选择证据，不上传模型、缓存或全部新日志。迁移不移动、删除或覆盖已有结果。

## interference_source_local_debug_20261102

- 当前传播／含误差测向协议的真实Qwen3-1.7B短诊断，RRT*、4次调用上限、生成不限时。前2次完整输出，分别补测和重规划；第3次7532输入词元时CUDA显存不足，第4次8736词元超8000上限、生成前拒绝。
- 前2次尚无检测角峰，有方向证据的两轮未取得响应；源提议／数值评价／接受均0，不能将其直接解读为模型源推断能力不足或故意忽略方向。0次真实源细化验证。
- 9次决策、2次后端失败、7次回退（含5次预算耗尽）、1次补测、10次求解；780.319m到达、超参考飞行暴露7.737s。是混合回退轨迹，不是全程LLM成功或同预算收益实验。
- `toolkit/python.exe`启动异常，诊断显式用uav Python加载现有toolkit推理包，RTX 3060 Laptop 6GB／float16；没有修复或安装环境。真实命令、版本、包路径和哈希在`run_metadata.json`。
- `calls.jsonl`、逐调用文件、正式日志、`audit.json`及`review.md`保留全过程。离线数字精度压缩仅计词元，未修改真实输入，也未验证显存或决策收益。
- PNG和首次方向失败检查帧已目视检查；离线派生`replay.gif`63帧85.52s、1倍仿真时间＋每次决策3s停顿，未重新推理／规划。图中没有模型源H，因为真实模型没有提源。

## interference_debug_numeric_20261102

- 当前传播／含误差测向配置的无模型完整调试流程，默认RRT*、种子7；54.338s到达，航程755.073m、能耗5.227Wh。
- 53次观测／扫描、47次检测到角峰；8次决策／求解，4次感知预警复核，0模型调用／回退／补测／源提议。
- 超参考飞行暴露3.746s、超参考积分3.210dB·s；到达不等于安全避让，不与旧感知或旧高斯场结果作收益对比。
- 保存逐阶段`debug_stages.jsonl`、正式episode／metrics／中文日志、运行元数据和已目视检查的`summary.png`；调试入口见`../temp/debug_full_pipeline.md`。

## interference_source_direction_mock_20261031_193820

- 2秒确定性模拟响应接口检查，不是LLM或通用定位算法；A*仅为短测试对照。
- 同点3轮：功率一致但方向错90°被拒绝，修订后接受，最后省略源字段；3次观测、2个源提议、1次接受、0解析失败。未到达，以时间上限结束，不证明收益。
- 夹具使用恰好匹配该合成场景的距离／功率假设，并关闭噪声以隔离接口；默认场景噪声未调整。完整限制见该目录`review.md`。
- 保存`episode.json`、`metrics.json`、`decision_log.txt`和已目视检查的`summary.png`；旧实验未覆盖。

## interference_sensing_numeric_seed7_20261030

- 新传播／360°测向基础的无模型RRT*接口检查，不是真实LLM实验，也不是收益对照。
- 53次观测／扫描，44次存在可检测角峰；9.5s在−93.94dBm时提前复核，仍低于−80dBm软参考。
- 全程7次决策、4次感知预警复核；航程751.424m，超参考飞行暴露5.05s，仍不能宣称安全绕行。
- `first_detection.png`展示首次方向发现，`summary.png`为末帧；`replay.gif`53帧69.59s，橙色扇区只是接收方向，不是已测圆盘或探测距离。
- 完整`episode.json`、`metrics.json`与`decision_log.txt`均保存，旧输出未覆盖。传播模型与旧高斯场不同，不以数值差异证明算法改进。

## interference_source_local_seed7_20261029

- 新源接口真实Qwen3-1.7B完整试跑；默认24次调用／1024输出词元／生成不限时，实际6次，0失败／覆盖／规划失败／回退。
- 原始JSON全部省略`source_hypothesis`，源提议／接受均0；不是数值拒绝或预测源数为零。27.167s已接收−65.85dBm及强干扰历史，仍没有源假设；真实多轮细化和新点预测检验均未触发。
- 输入3376—6621词元，输出97—108，未触及预算；航程761.939m、超参考飞行暴露18.382s，与前一无源接口试跑的轨迹完全相同，不证明源推断或路径收益。
- `review.md`、`run_metadata.json`、`audit.json`记录逐轮证据和审计；`calls.jsonl`及`decision_log.txt`保留实际输入输出。`prompt_audit.json`仅离线量化3位小数压缩潜力，未改变真实输入或验证压缩后的决策。
- `summary.png`、60帧73.80s的`replay.gif`和`strong_interference_decision.png`为本次新结果；右侧没有H，因为模型没有提出源，不伪造推断展示。旧输出未覆盖。

## interference_local_unlimited_seed7_20261029

- 2026-10-29真实Qwen3-1.7B闭环：不设生成墙钟时限，默认8次调用预算，实际6次；输入2602—3461词元，每次完整输出112或102词元，0失败／覆盖／回退。
- 首次补测后5次均为相同的`replan + avoid_interference`响应；27.167s模型已收到−65.85dBm点测及7次超参考的区间反馈，仍未体现针对性计划修订。参与执行不等于策略有效。
- 航程761.94m，超参考飞行暴露18.38s；不是单因素或同预算收益对照，不与旧试跑差异直接比较来证明提升。
- `review.md`记录逐轮证据和局限，`run_metadata.json`记录命令／模型／环境／预算；原始输入输出见`decision_log.txt`和`episode.json`。
- `summary.png`及`replay.gif`为本次新结果，GIF共73.80s，1倍仿真时间加决策阅读停顿；`strong_interference_decision.png`为观测强干扰后的决策帧。

## interference_stage_plan_local_actions_seed7_20261027

- `episode.json`、`metrics.json`及`decision_log.txt`：已有Qwen3-1.7B非代码阶段计划试跑记录，未因本轮回放修改重新推理或规划。
- 3次真实模型调用的输入分别2602／3065／3316词元；前两次完成补测和反馈后重规划，第三次不完整JSON导致解析失败。总计4次回退（1次解析失败、3次调用预算耗尽），到达不等于模型优化有效。
- 仿真53.824s，航程762.37m，超参考飞行暴露19.03s；总干扰参考值不是平台失效阈值。
- `summary.png`、`replay.gif`：原有图像，保留不覆盖。
- `replay_slow.gif`：2026-10-27从同一episode离线派生，1倍仿真时间，每次决策额外停顿3s，60帧共73.83s；去掉底部技术文字，顶部解释实际行动和回退，金色菱形标记补测目标。阅读停顿不是推理耗时。
- `replay_first_frame.png`和`replay_fallback_frame.png`：易读版首帧和模型解析失败时的检查帧。

完整配置、实际输入输出和执行层覆盖以原始日志为准；压缩建议不是已实现的新实验。
