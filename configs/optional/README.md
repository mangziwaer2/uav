# 可选环境配置

主实验优先使用上一级`demo.json`或`demo_25d.json`，均有本地时变源和含误差测向。本目录不作为启动LLM实验的前提。

| 文件 | 用途与限制 |
|---|---|
| `demo_power_grid.json` | 外部静态总干扰功率适配示例，仅power_only；引用的`datasets/example_power_grid.json`当前缺失，不能直接运行 |
| `import_25d_template.json` | 离线GeoJSON导入模板，建筑和矩形障碍为空，配合`tools/import_buildings.py` |
| `sionna_rt_demo.json` | Sionna导出模板，不是已经生成干扰场的飞行任务 |
| `sionna_rt_sources.json` | 隐藏离线源设置，只供生成器；禁止送给LLM或作为在线定位结果 |

文件路径相对JSON所在目录解析，已随移动调整功率图引用。外部适配不会从纯功率图伪造来波方向。完整字段见`../README.md`，工具说明见`../../tools/README.md`。
