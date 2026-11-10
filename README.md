# 局部点测下的LLM辅助电磁干扰动态重规划

当前研究使用合成城市建筑与隐藏动态发射源。UAV只测量当前位置总干扰功率，局部时空GP根据已发生的点测提供估计与未知标记；决策LLM选择继续、补测或带中间点的重规划，数值层求解并检查几何和能量。

**结果入口：[outputs/index.html](outputs/index.html)**；GIF统一位于对应研究的`replay/`。当前强LLM点测验证见[本轮研究](outputs/interference_point_agent_20261109/index.html)。此处验证基础可行性，尚不证明提出的推理优化方法优于同预算传统算法。

环境能力和文献依据见[环境假设](docs/local_power_environment.md)与[LLM信息合同](docs/llm_information_contract.md)。后续方向、候选贡献和具体方法只在[研究路线](docs/current_roadmap.md)与[方法设计](docs/research_method_20261109.md)维护；已实现能力见[实现说明](docs/first_version.md)。

## 运行

无需GPU或外部电磁仿真软件。基础依赖统一在`requirements.txt`：

```bash
python -m pip install -r requirements.txt
# 数值闭环：新随机城市、无LLM、无预飞行稀疏测量
python tools/run_local_power_demo.py --random-city --seed 231 --output outputs/interference_demo_new
# 测试
python -m pytest -q -p no:cacheprovider
```

本机已有解释器`temp/uav_cloud_env/bin/python`，可替代上述`python`。已有输出目录拒绝覆盖。

在线实验分为准备和运行两步；运行前必须启动仅允许读取新研究`public/`目录的决策子agent，完整格式见生成的`public/contract.md`。文件桥接器本身不会自动调用外部LLM，也不是固定规则的模拟输出：

```bash
python tools/run_point_agent_study.py --output outputs/interference_point_agent_new --prepare
python tools/run_point_agent_study.py --output outputs/interference_point_agent_new
python tools/summarize_point_agent_study.py --output outputs/interference_point_agent_new
```

只生成已有episode的回放：

```bash
python tools/render_navigation_replays.py --root outputs/interference_point_agent_20261109
```

逐阶段调试入口为`temp/debug_full_pipeline.py`。保留的本地模型后端需要另装`torch>=2.6,<3`、`transformers>=4.51,<6`及`safetensors>=0.5`；当前实验没有加载本地权重，未开展训练/RL，不需要LoRA依赖。原始模型与数据原位保留，但不参与当前点测验证。

## 研究边界

总功率dBm越大干扰越强；−80dBm是软评价参考，不是设备失效阈值。真值只用于产生当前读数和事后评价，不送入决策LLM。局部GP不是全图扫描或已验证精度的无线电地图；不提供源位置、测向或未来动态脚本。到达、原始模型输出、解析接受、实际执行和数值回退分别记账。

当前没有低层动力学、抗干扰定位、通信失效/中断预算或实机验证。模型和求解的真实计算时间独立记录，不据此声称实时部署。目录与原始材料保留边界见[workspace_layout.md](docs/workspace_layout.md)。
