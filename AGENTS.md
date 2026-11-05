# 当前项目维护约定

- 先读`docs/AGENT_HANDOFF.md`、`docs/current_roadmap.md`和`docs/first_version.md`。
- 当前代码为`src/uav_em/`及src下实验／回放入口；可选离线工具在`tools/`，配置在`configs/optional/`，指定调试入口仍为`temp/debug_full_pipeline.py`。em指Electromagnetic（电磁），不是旧项目。输出使用`interference_`前缀。旧归档不是依赖，不恢复历史接口。
- 仿真基础支持定高2.5D建筑及静态外部总干扰功率栅格，见`docs/environment_25d.md`。只声称仿真数据实验，不以无实测标定为由阻止研究，也不声称已验证任意设备迁移。功率栅格不能伪造测向，数据适配不能泄漏全图真值。
- 主配置为`configs/demo.json`和`demo_25d.json`：普通建筑＋本地时变多源＋有限一次Fresnel反射，不依赖外部数据、GPU或Sionna。Sionna只保留可选核查；省略传播模式的历史配置仍为log_distance，不擅自重算历史语义。
- 范围：飞行局部测量、地图更新、直接模型咨询、路径优化及补测。2026-11-04新增随机场景、离线时序地图与小编解码器；2026-11-05用户进一步授权独立测向证据、冻结LLM主体的地图投影／时间层训练，遇本地显存不足后交接云端，见docs/map_sensor_alignment.md及docs/CLOUD_ALIGNMENT_HANDOFF.md。不擅自启动LLM主体微调／RL、专门模型评价、中断时间、门控或在线代码执行。
- 研究顺序与待办以`docs/current_roadmap.md`为准：先完善闭环和模型策略，再按前提探索中断穿越、局部代码及可选门控；模型评价与调优不以代码生成或门控成功为条件。未来方向不等于本轮实现授权。
- 当前环境量为总电磁干扰功率dBm，越大干扰越强；不是服务站覆盖或通信质量。默认RRT*，A*仅为可选对照；日志版本2拒绝旧通信质量日志，不做错误的语义兼容。
- 配置字段中文解释含义、单位与限制。明确研究近似、信息来源及未实现能力。
- 决策不能访问未观测真值或未来脚本。模拟响应、真实模型与回退分开记账，回退到达不等于模型优化有效。
- 云端迁移先读`docs/CLOUD_HANDOFF.md`。基础测试命令：`python -m pytest -q -p no:cacheprovider`；Windows历史解释器为`E:\Miniforge\envs\uav\python.exe`。云端使用独立且CUDA可用的环境，不复制Windows的toolkit路径混用方案。
- Git保留源码、测试、接手文档、指定调试入口及明确选定的现存复现记录；旧三份精选记录当前已缺失，不恢复或冒充现存证据。当前数值验收清单见`outputs/README.md`。模型权重、其他输出、缓存、原始提交文件和根目录压缩包不上传、不自动删除。新实验必须写入新的`outputs/interference_...`目录，不覆盖入库证据。
- 用户已要求直接删除旧项目，不再创建或保留归档；此前自动删除被策略拦截，2026-10-22复查旧归档和旧通信输出已不在工作区，不再列为待删，也不声称由本次操作删除。保留用户模型、提交文件和来源不明的外部缓存。
