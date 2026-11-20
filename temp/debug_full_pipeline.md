# 调试当前入口

保留用户指定文件名，debug_full_pipeline.py仅转发当前唯一入口。可在src/uav_em/scenario_loop.py设置断点查看数值参考执行；这仍是固定时刻参考，不是异步LLM方案。

```bash
temp/uav_cloud_env/bin/python temp/debug_full_pipeline.py describe
```
