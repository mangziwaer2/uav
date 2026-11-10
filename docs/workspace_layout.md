# 工作区与结果目录

更新：2026-11-09。当前研究为仅点功率感知下的LLM辅助绕行与动态重规划；一切运行结果从[outputs/index.html](../outputs/index.html)进入。本轮记录为[interference_point_agent_20261109](../outputs/interference_point_agent_20261109/index.html)，验证已完成。

```text
src/uav_em/                   仿真、传播、点测、局部估计、数值规划和决策接口
src/run_experiment.py          通用实验入口
src/replay_experiment.py       已保存episode的回放入口
tools/                        当前演示、在线agent实验和集中回放工具
configs/local_power.json      点功率/局部时空GP主配置
tests/                        当前保留能力的测试
docs/                         接手、当前方法、环境依据与信息合同
outputs/index.html            唯一结果总入口
outputs/interference_.../     每项新研究的独立完整记录
models/                       用户模型与既有最终权重，保留但不作为当前依赖
datasets/                     原始/派生数据，保留但不作为当前依赖
docs/submissions/             用户提交原件
temp/debug_full_pipeline.py   指定调试入口
temp/uav_cloud_env/            当前Python环境，保留原位避免路径失效
```

场景生成器的正式位置为`src/uav_em/scenarios.py`，不再挂靠地图编码训练包。当前不使用旧地图训练、初始稀疏输入、外部功率图或Sionna流程。测向/源假设内部兼容分支仍存在；主配置和本轮公共协议不启用，不据此宣称已清除源码每一个历史分支。

用户已授权删除旧实验、临时文件及废弃代码/文档，不创建归档。保留模型、原始数据、提交材料和来源不明的外部缓存；实际清理清单在本轮reports/cleanup_manifest.json；旧报告已删除。原始提交材料中的历史路径不代表当前存在。

每项研究使用一个新的`outputs/interference_<主题>_<日期>/`，已有记录不覆盖：

```text
index.html       人可读结果入口
RESULTS.md       结论、比较条件与限制
protocol.json    场景/方法/信息协议及运行前冻结信息
status.json      当前执行状态
scenarios/       仿真配置；隐藏源不可提供给决策agent
public/          决策agent允许读取的协议、控制、请求和自己的响应
runs/            按方法/场景/种子分组的episode、metrics、日志和静态图
reports/         汇总、审计及必要的冻结源码
replay/          集中GIF、index.html与来源清单
logs/            运行和验证日志
```

共享目录不是信息隔离沙箱。决策agent只能读公共协议明确允许的文件，不可读隐藏场景或完整episode；模型输入、原始响应、接受/拒绝、实际执行和回退必须保留。

GIF集中在`replay/`，不在每个run散放。每次收尾更新研究入口和总入口，使用户无需搜索日志目录就能看表现。失败/追加尝试在同一研究下建清楚标注的新run，不新增含糊的resume根目录，不篡改旧响应或指标。
