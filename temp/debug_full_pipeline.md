# 点测闭环调试

入口`temp/debug_full_pipeline.py`复用正式`simulate`，不复制另一套算法。默认配置`configs/local_power.json`，默认`none`是数值基线，`mock`为工程夹具，`local`才加载本地LLM；当前主实验使用`FileDecisionAdvisor`连接在线子agent。

```bash
temp/uav_cloud_env/bin/python temp/debug_full_pipeline.py --max-seconds 8 --quiet --output-dir outputs/interference_debug_new
```

在`PipelineDebugger.__call__`设置断点，或加`--pause-at observation_received`。可见测量、估计、上下文、模型响应、规划和执行均有阶段日志。调试器取得深拷贝，修改它不改变仿真。完整进程变量仍可能含研究者真值，不能把调试目录授权给决策agent。

输出拒绝覆盖。`evaluation_only_config.json`含隐藏源，仅用于复核；`call_*_context.json`和`calls.jsonl`保存模型真实可见输入/响应。`none`没有模型调用，`mock`不能用于证明LLM能力。模型与断点墙钟时间不推进仿真。
