# 当前配置

仅local_power.json作为参考配置。它指定已知2.5D建筑、仿真隐藏源、点功率传感和GP参数。fields只供仿真器使用，不可直接发给LLM。

ProjectConfig保留部分旧字段以读取冻结快照；当前链不启用测向、A*、主动补测或旧源定位控制。参考飞行仍用scenario_loop的固定合同和事件时刻，不能靠改simulation旧触发字段声称实现了新异步调度。新合同另行明确，见[路线](../docs/current_roadmap.md)。
