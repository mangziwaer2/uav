# 局部点测下的 LLM 辅助电磁干扰动态重规划

研究目标：在已知城市建筑、隐藏动态发射源、UAV 仅能测量自身位置总干扰功率的条件下，让 LLM 利用历史观测、局部估计及数值反馈，在有限规划预算内改善航程与干扰暴露的权衡。

| 要做什么 | 从这里进入 |
|---|---|
| 看新方向的实施计划和第一步 | [阶段计划](docs/event_inference_plan.md)、[事件时间线](outputs/interference_event_inference_20261118/index.html)（数值接口验证，无新飞行/LLM） |
| 看当前等效表示能力检查 | [无公式说明](docs/equivalent_capacity_guide.md)、[指标与限制](outputs/interference_equivalent_capacity_20261118/index.html)、[6城市GIF](outputs/interference_equivalent_capacity_20261118/replay/index.html) |
| 看预测模型训练结果 | [无公式说明](docs/prediction_model_guide.md)、[四方法指标](outputs/interference_prediction_model_20261118/index.html)、[预测GIF](outputs/interference_prediction_model_20261118/replay/index.html) |
| 了解新数据怎样用于预测 | [数据采集说明](docs/prediction_data_guide.md)、[24航迹试采结果](outputs/interference_prediction_data_20261118/index.html) |
| 不看公式理解当前飞行流程 | [简单闭环说明](docs/short_loop_guide.md) |
| 看当前结果 | [结果总入口](outputs/index.html) |
| 直接看 GIF | [集中回放](outputs/replay/index.html)，按实际闭环、旧轨迹与反事实标明类型 |
| 理解等效源的计算 | [输入输出、公式与手算例子](docs/equivalent_source_walkthrough.md) |
| 确认下一步 | [当前研究路线](docs/current_roadmap.md) |
| 查环境和感知依据 | [仿真环境与文献](docs/local_power_environment.md)、[信息合同](docs/llm_information_contract.md) |
| 找代码、工具和文档 | [目录说明](docs/workspace_layout.md)、[工具入口](tools/README.md)、[文档导航](docs/README.md) |
| 写实践材料 | [记录表](记录表/README.md)、[实践项目申请书](docs/实践项目申请书.md) |

## 当前进展

当前采用[事件式LLM组织推断与重规划计划](docs/event_inference_plan.md)。阶段A已实现持续假设维护、出生后计分、事件触发、结构化推断计划及有界数值拟合。[事件时间线](outputs/interference_event_inference_20261118/index.html)：两条旧航迹×三数值方法、144一步预测，工程核验通过。固定等效混合尚未胜功率延续，两种维护均8/20s到期；没有真实LLM或新飞行，不称自适应/规划收益。

809测试/独立核验通过，58旧核心不改，新增第59模块event_inference。下一步另冻真实LLM组织机制/区域/容量分配的独立合同，再接多场景短程规划。[当前研究路线](docs/current_roadmap.md)。

前置[等效表示能力检查](outputs/interference_equivalent_capacity_20261118/index.html)：4基元oracle留出RMSE2.529dB、归一化Q误差10.26%，但部分状态排序不佳、另两状态无可区分路线对，完整门槛仍False；不是点测反推或新飞行。

前置[简单LLM闭环](outputs/interference_short_loop_20261118/index.html)和[混合数据](outputs/interference_prediction_data_20261118/index.html)保留；[小模型CPU试验](outputs/interference_prediction_model_20261118/index.html)未胜功率延续，权重保留且未部署。当前实际飞行接口仍是局部GP＋固定RRT*＋数值或LLM短程决策。旧等效压缩/SLSQP修复是暂缓支线，非必经前置。

## 运行

CPU 即可；依赖只维护 `requirements.txt`，现有解释器为 `temp/uav_cloud_env/bin/python`。

```bash
# 新建独立输出；基础数值闭环，不调用 LLM
temp/uav_cloud_env/bin/python tools/run_local_power_demo.py --random-city --seed 231 --output outputs/interference_demo_new

# 需要改代码时的全量检查
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 temp/uav_cloud_env/bin/python -m pytest -q -p no:cacheprovider
```

最近一次全量记录为 809 测试通过、1 项第三方警告。已完成研究的输出、预算和冻结协议保留原位；各研究入口见结果导航，勿把旧驱动当作追加实验的命令。

当前是仿真研究。−80dBm 为归一化软参考，GP 支持和未校准方差不构成安全保证；具体假设与边界见环境文档。[本次整理记录](docs/maintenance/workspace_cleanup_20261118.json)。
