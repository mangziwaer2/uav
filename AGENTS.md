# 当前项目维护约定

- 先读`docs/AGENT_HANDOFF.md`、`docs/current_roadmap.md`和`docs/first_version.md`。本文件按2026-11-09最新授权收敛，旧方案不构成隐含待办。
- 主代码`src/uav_em/`，入口`src/run_experiment.py`、`src/replay_experiment.py`，工具`tools/`；主配置`configs/local_power.json`。指定调试入口`temp/debug_full_pipeline.py`保留。em指Electromagnetic。
- 当前主线：合成2.5D已知建筑+隐藏时变发射源；有限一次反射/穿楼/毫瓦叠加；UAV仅当前位置带噪总干扰功率点测；局部时空GP；LLM辅助重规划/补测。无预飞行稀疏测量、全图补全、地图编码训练、外部功率栅格或Sionna依赖。
- 环境依据与研究参数边界见`docs/local_power_environment.md`、`docs/llm_information_contract.md`。论文支撑建模路线，不冒充具体参数实测标定。只声称仿真研究，不以未实测阻止研究，也不宣称任意设备迁移。
- 总外部干扰功率dBm越大越强，不是通信质量或服务站覆盖；参考值是软代价，不是禁飞/失效保证。已知几何、自定位和机载计算是显式条件。缺失测向不能从标量功率伪造，未知不等于低干扰。
- 子agent可作为在线决策LLM；必须新建独立会话，仅给本轮public请求、合同和同任务记忆，禁止真源/未来脚本/真值图/其他实验泄漏。共享文件系统仅为流程allowlist，不是OS隔离。实际请求、原始响应、解析接受、执行、错误与回退分别留证。
- 默认RRT*，A*仅可选对照。当前基础验证仅continue/replan/probe、预设profile及一个中间点；内部保留的其他接口不构成本轮方法授权或结果。后续候选创新是证据条件化的区段修复操作选择+固定目标约束总预算下的数值反馈优化，尚需实现与同条件对照。
- 用户允许离线LLM设计算法、在线选择/修订算法和算子组合。不得把未来设计写成已实现。不得擅自启动主体微调/RL、门控、通信中断模型或在线代码执行。后续顺序以current_roadmap为准。
- 用户已明确授权删除旧实验、temp冗余、训练编码代码和失效文档，不创建历史项目归档。保留用户models、原始datasets、提交文件、来源不明外部缓存及现有venv；不要自动删除这些资产。旧实验结果不再作为当前证据引用。
- 新输出必须新建`outputs/interference_...`，拒绝覆盖；一研究一入口，GIF集中`replay/`。当前证据见outputs/index.html。冻结运行源码与清理后源码可能不同，必须说明版本和重放核验，不伪造哈希一致。
- 依赖只维护requirements.txt；当前无需GPU。基本测试`python -m pytest -q -p no:cacheprovider`；本机解释器`temp/uav_cloud_env/bin/python`。不通过补造/跳过缺失夹具冒充测试成功；随明确删除功能移除的旧测试需说明范围。
- 中文解释配置含义、单位和限制。文档无旧方向歧义；任何结果必须区分基础可行性与方法优化收益。不要自动git提交/推送。
