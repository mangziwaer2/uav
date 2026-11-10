# 当前配置

只维护`local_power.json`：已知2.5D城市建筑、隐藏时变源、有限一次反射、机载当前位置功率点测、局部时空GP。单位为米、秒、Hz、dBm、dB和Wh，参数含义见`docs/first_version.md`和`docs/local_power_environment.md`。

静态建筑可以手工给定；`tools/run_local_power_demo.py --random-city`使用`uav_em/scenarios.py`随机生成建筑和源。原始配置中的发射源脚本只供仿真器，不进入LLM。实际任务配置保存在新研究的`scenarios/`，只能供研究者查看。

旧编码训练、测向演示、稀疏初始测量和外部引擎配置已退出当前项目，不作为默认备用路线。
