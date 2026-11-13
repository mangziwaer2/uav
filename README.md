# 局部点测下的LLM辅助电磁干扰动态重规划

研究目标是在动态、部分可观测的电磁环境中，让LLM利用观测证据和数值反馈辅助航路优化。当前流程是：合成已知建筑与隐藏动态发射源 → UAV当前位置带噪总干扰功率点测 → 局部时空GP → 路线修复过程 → 数值规划和几何/能量复核 → 短段飞行与新点测。无预飞行稀疏测量、全图补全、地图编码或训练。

**[全部结果入口](outputs/index.html)** · **[最新离线结构生成结果](outputs/interference_structure_generation_20261111/index.html)** · **[重点GIF对照](outputs/interference_structure_generation_20261111/replay/focus.html)** · [全部16个新GIF](outputs/interference_structure_generation_20261111/replay/index.html) · [生成程序与执行例子](outputs/interference_structure_generation_20261111/reports/generated_examples.md)

最新[离线LLM结构表达式生成v2](docs/offline_structure_generation_v2.md)已完成4个独立真实设计会话（2初稿+2修订）、4份合法程序、16同状态求解及16次新闭环，全部到达，运行时0模型调用/策略错误，564测试通过。生成8个有限标量表达式，控制入口、前向接回点、优先级、样本/边额度、停止与直连/C-RRT*；不是任意Python或新GP/RRT*底层实现。38个原模块未改，四数值参考的16飞行/16配对明确复用并核验，未重新运行。draft_02飞行T+Q347.146→338.697（−2.43%）、总超参考94.664→91.101dB·s（−3.76%），但dev613的T+Q约+9.34%、超参考0→6.394，违反每城≤105%的预声明条件；修订未保留总体收益。**四产物均未通过，无胜者，本轮4份额度用尽，不追加第5份。** 16GIF均为新实际闭环；未见验证/测试继续封存。下一步先另冻dev613首次分岔与入口/接回/调度的因果归因协议，必要时研究对所有方法一致的预测/接受改进，不盲目追加生成或私改GP/C。

以下为既有阶段摘要，其中“下一步”是当时记录；当前待办仅以current_roadmap第15项为准。

前一[多接回点与分批竞争](docs/reconnection_search_v1.md)已完成16同状态求解/16真实闭环，全部到达，0模型/新生成，544测试通过。新结构支持固定剩余路线上的多个接回点、最多6批独立C-RRT*和共享台账硬边切片；37原有模块不变。四组为单次/同点分批/多点均分/多点反馈，初始池、GP/C、首批随机流及700追加采样/2100总样本/10000边上限一致。本轮新配对流不拿上轮数值当分母。主反馈组真实飞行T+Q347.146→348.977（+0.528%）、超参考94.664→106.158dB·s（+12.142%），未获任务收益；均分T+Q−1.432%但超参考+6.772%，不是稳定避强或LLM收益。113决策GP重建、完整候选/批额度/几何能量核验，16GIF为实际闭环。本阶段已由最新结构生成研究接续，后续顺序以current_roadmap为准。

前一[C一致搜索原语](docs/aligned_search_v1.md)已完成16同状态求解、8实际闭环，全部到达，0模型/新生成提交，536测试通过。新原语由研究者编写，让池后RRT*选父/重连/终点连接使用共同C；原GP/传播/接受器及36旧模块不变，共同池仍为旧J。两组同初始池、空搜索缓存、配对随机流和2100样本/10000边总账。8组状态比较7组当前C下降，但其中6组真实联合分变差；闭环合计飞行T+Q354.936→354.459（-0.134%），含悬停377.292→377.442，不改善。dev619超参考18.482→15.585dB·s，dev617为92.035→93.690而退步；总边+3.58%，无稳定避强/效率或LLM收益。8GIF为实际闭环。下一步进入多接回点、分批竞争搜索的结构设计，再冻结新离线生成合同/预算；不重复旧profile选择或追加旧生成名额，验证测试继续封存。

前一轮联合评价减少了起点绕路：平均航程−19.15%，但dev617暴露明显增加；主偏好下合计实际联合分改善7.31%，不是全面避强或LLM收益，见[共同目标v3](docs/balanced_objective_v3.md)。

此前完成4个独立离线设计会话、2初稿+2修订、28新闭环；四份生成过程均未通过完整收益门槛，无胜者，不追加原预算。这些产物只组合现有有限算子，不执行任意Python。当前基础强LLM可绕行，尚未证明同信息/目标/预算下的在线LLM推理优化优势。到达仅是必要条件，还须报告航程、暴露和计算成本。

| 已完成阶段 | 证据入口 |
|---|---|
| 点功率下在线强LLM基础绕行 | [报告及4个GIF](outputs/interference_point_agent_20261109/index.html) |
| 固定目标/总预算区段修复、采样分配与反馈 | [第一版](docs/route_repair_v1.md) |
| 直接数值与有限AST算子程序比较 | [结果](docs/program_comparison.md) |
| 快照瓶颈诊断、有限过程数值基线 | [诊断](docs/repair_diagnostics.md)、[过程](docs/repair_process_v1.md) |
| 开发任务、共同目标及证据/额度检查 | [任务](docs/task_readiness.md)、[目标](docs/decision_objective_v2.md)、[额度](docs/evidence_budget_process.md) |
| 离线接口与第一轮真实生成 | [合同](docs/offline_process_contract.md)、[生成结果](docs/offline_generation_v1.md) |
| 生成过程阶段截断与同状态归因 | [诊断](docs/process_attribution_v1.md) |
| 航程—连续功率评价与独立搜索流 | [结果](docs/balanced_objective_v3.md) |
| 风险证据状态候选覆盖与代理偏差 | [诊断](docs/risk_repair_diagnostics_v3.md) |
| 因果可靠性反馈与真实在线profile选择 | [前一阶段结果](docs/causal_feedback_v1.md) |

当前待办只以[研究路线](docs/current_roadmap.md)为准；接手看[AGENT_HANDOFF.md](docs/AGENT_HANDOFF.md)，已实现能力见[first_version.md](docs/first_version.md)。历史阶段的计划不作为当前待办，冻结报告不覆盖。

## 运行

无需GPU或外部电磁仿真软件。依赖只维护`requirements.txt`，本机解释器`temp/uav_cloud_env/bin/python`可替代下面的`python`。

```bash
python -m pip install -r requirements.txt
# 基础数值闭环，新目录拒绝覆盖
python tools/run_local_power_demo.py --random-city --seed 231 --output outputs/interference_demo_new
# 当前全部测试
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider
# 前一固定快照诊断的CPU复现，无新飞行/模型
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/run_risk_diagnostics.py all --out outputs/interference_risk_diagnostics_new
# 前一联合评价的完整CPU复现，不调用LLM
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/run_balanced_objective.py all --out outputs/interference_balanced_objective_new
# 只从已有episode生成集中回放，不重新规划
python tools/render_navigation_replays.py --root outputs/interference_balanced_objective_20261111
```

其他研究的单入口和阶段参数见[tools/README.md](tools/README.md)，每研究GIF集中`replay/`。在线实验需要另行启动新的独立决策会话，只读对应public合同/请求；文件桥接脚本不会自动召唤或伪造LLM。指定调试入口为`temp/debug_full_pipeline.py`。

现有本地模型与原始数据保留，不参与当前CPU研究；可选本地后端另需`torch>=2.6,<3`、`transformers>=4.51,<6`及`safetensors>=0.5`。未开展训练/RL，不依赖LoRA。

## 研究边界

已知建筑几何、自定位与机载计算是显式条件。功率dBm越大干扰越强；−80dBm为软参考，dB·s不是物理剂量、通信质量或失效概率。GP仅从历史点测作有支持范围和时效的相关性外推，不是周围扫描，未知保持null；没有源坐标、测向或未来脚本。隐藏真值只生成当前位置读数及事后评价，不送决策模型。

当前无低层动力学、抗干扰定位、通信中断或实机验证。固定0.5s仿真悬停不等于实际推理时间；墙钟单列，不据此声称部署实时性。环境依据见[环境假设](docs/local_power_environment.md)与[信息合同](docs/llm_information_contract.md)，候选论文贡献见[方法设计](docs/research_method_20261109.md)，目录/资产边界见[工作区](docs/workspace_layout.md)。

[C一致搜索原语与冻结协议](docs/aligned_search_v1.md)；运行`tools/run_aligned_search.py`，新建输出，拒绝覆盖。
