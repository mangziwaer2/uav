# 当前工具

- `run_local_power_demo.py`：随机/指定城市的点测→局部GP→数值规划闭环，不调用LLM。`--output`必须是新目录。
- `run_point_agent_study.py`：准备冻结的基础点测场景与公共合同，然后通过文件桥接器接受真实在线决策agent响应。程序不会自动召唤LLM；无外部决策进程时会等待并最终记为失败。
- `summarize_point_agent_study.py`：从已有episode/请求/响应生成对照表、调用审计、静态图及集中GIF，不重新做策略决策。
- `reproduce_point_agent_study.py`：用已保存的真实动作在当前源码上重放闭环，核验轨迹和指标；这属于记录动作重放，不是新的模型调用或独立重复实验。
- `render_navigation_replays.py --root <研究目录>`：只从已保存episode重建`replay/`画廊，绝不重新规划。

每项的准确参数用`--help`查看。旧训练、地图编码、LoRA、菜单研究和外部射线工具已删除；模型/原始数据保留不意味着当前还依赖它们。
