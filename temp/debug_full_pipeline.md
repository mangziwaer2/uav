# 完整流程逐步调试

日期：2026-11-02。入口为同目录`debug_full_pipeline.py`。这是用户要求的完整流程调试代码，不是算法演示副本：它直接调用`src/uav_em/simulation.py`中的正式`simulate`，因此数值基线、模拟响应和真实LLM经过同一条接收—估计—咨询—验证—规划—飞行闭环。

## 1. 先运行不加载模型的版本

在项目根目录运行：

```powershell
& 'E:\Miniforge\envs\uav\python.exe' temp/debug_full_pipeline.py --advisor none --plot
```

默认读取`configs/demo.json`，默认RRT*与配置预算，运行完整任务；默认不绘图，可用`--plot`输出PNG、`--gif`额外生成GIF。每次自动新建`outputs/interference_debug_后端_时间戳/`，不覆盖旧结果。`--output-dir`指定的目录必须不存在。

只想先检查接线，可以加`--max-seconds 2`；这仅是仿真时长上限，不是模型生成时限，也不是完整任务验证。`--quiet`关闭逐阶段刷屏，但完整阶段日志仍保存。

## 2. IDE调试：推荐入口

- 脚本路径选择`temp/debug_full_pipeline.py`，工作目录选择项目根目录，解释器先用`uav`。
- 无模型参数填`--advisor none`，要看工程模拟响应填`--advisor mock`。mock不是LLM，默认也不会提交源假设。
- 在`main`中的`episode = simulate(...)`放断点，按“步入/F7”进入正式循环；不要只“单步跳过/F8”而把整个仿真一次执行完。
- 更便于定位的断点是在`PipelineDebugger.__call__`中的`self.sequence += 1`。停住后展开`stage`和`data`；可加条件`stage == "source_evaluated"`或`stage == "observation_received" and data["time_seconds"] >= 9`。
- IDE调用栈往上可以查看`emit`和`simulate`的变量；只有研究者调试器能看到整套真值对象，决策上下文并没有这些信息。

`data`是深拷贝，改它不会修改真实仿真。回调不返回控制指令、不把调试内容塞给LLM。不要在调试器中改实际`world`、`belief`或局部决策后，把结果作为正常实验。

## 3. 命令行断点

```powershell
& 'E:\Miniforge\envs\uav\python.exe' temp/debug_full_pipeline.py --advisor mock `
  --pause-at observation_received --pause-at context_built --pause-at decision_executed
```

暂停后是Python自带pdb，不需要安装调试器。常用命令：`p stage`查看阶段、`pp data`展开数据、`n`下一行、`s`进入函数、`u`上一级调用栈、`c`继续到下个断点、`q`退出。`--pause-at all`会在全部阶段暂停，较繁琐，初次建议只选2至3个阶段。

## 4. 一轮到底经过哪些阶段

| 阶段名 | 应观察的变量 | 对应正式代码 |
|---|---|---|
| `initialized` | 起终点、能量、后端、规划器 | `simulate`初始化 |
| `observation_received` | `measurement`、旧估计、`innovation_db`；此时地图尚未更新 | `ElectromagneticWorld.observe`，内部进入传播与测向 |
| `belief_updated` | `source_check`（旧源新点预测误差）、已测比例、源状态与预警 | `ElectromagneticBelief.update/snapshot`、`SourceTracker.observe` |
| `review_checked` | `due`、功率变化、方向失配、周期、补测结束 | `simulate`复核条件 |
| `context_built` | `context`；没有真实源坐标或活动脚本 | `build_context`、阶段与执行反馈 |
| `advisor_started` | 本轮编号和实际传给后端的上下文；多轮细化会重复 | `RecordedAdvisor.decide` → `LocalModelAdvisor.decide`或`MockAdvisor.decide` |
| `advisor_returned` | 原始JSON、严格解析后的决策，或错误 | `parse_decision`；此阶段已完成解析和可选源检验 |
| `source_evaluated` | `candidate_fit`、功率／方向RMSE、接受／拒绝理由、旧源保留状态 | `SourceTracker.consider/score/bearing_score` |
| `plan_solved` | 路径、成功／失败、搜索预算、目的；可有补测尾程和回退求解 | `plan` → RRT*／A* |
| `decision_executed` | 原提议与真正接受的行动、覆盖／回退、阶段计划、当前航路 | 几何／能量复核及`update_stage_plan` |
| `flight_executed` | 本次经过的折线、剩余航路、新位置、能量 | `advance_route`与短段执行 |
| `finished` | 到达或终止原因、调用／回退／暴露等指标 | 仿真终止 |

重要的时序：

1. 每个有效测量都会更新地图，innovation不是是否更新地图的开关。
2. 新点功率／方向检查先使用旧源版本，再把这次测量加入窗口；窗口拟合是另一件事。
3. 不一定每步都咨询；`due=false`会直接沿已有路径推进。
4. 同点源细化可能有多次`advisor_started/returned/source_evaluated`，期间没有飞行和新观测。
5. 只执行最后一轮行动；接受源假设不等于接受飞行动作，更不等于证明真实定位。
6. 决策悬停单独计固定仿真时间；实际模型运行多久、断点停多久都不让场按墙钟时间移动。
7. 暴露积分只作事后评价，不是根据隐藏真值提前避让的安全盾。

断点／调试写盘不会推进仿真时钟，但会影响外层墙钟统计，例如咨询阶段的`advisor_seconds`可能包含人工停顿。每轮`model_runtime`另记实际后端生成情况；调试运行不用于比较推理速度或规划速度。

## 5. 调试真实LLM

正常解释器可用时：

```powershell
& 'E:\Miniforge\envs\toolkit\python.exe' temp/debug_full_pipeline.py --advisor local `
  --model-device cuda --pause-at advisor_started --pause-at source_evaluated
```

默认本地模型`models/Qwen3-1.7B`，输入上限8000、输出上限1024，生成不限墙钟时长；不自动下载、不自动换模型、不修补JSON。`--max-calls`只覆盖真实调用预算，耗尽后显式数值回退，不是以后每次都假装由模型决策。源假设不一定出现，因此`source_evaluated`可能一次都不停；看`advisor_returned`可以确认是省略源、解析失败还是被数值拒绝。

2026-11-02环境检查：`toolkit/python.exe -V`就退出，返回码`-1073741790`，不是LLM能力或词元问题；原因尚未诊断到具体DLL。未修改用户环境。当前可以使用`uav`的Python 3.10加载已有`toolkit`推理包，以下为显式临时进程内导入路径，不安装或复制包，也不是无提示自动回退：

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'E:\Miniforge\envs\uav\python.exe' -u -c "import sys,runpy; sys.path.insert(0,r'E:\Miniforge\envs\toolkit\Lib\site-packages'); sys.argv=['temp/debug_full_pipeline.py','--advisor','local','--model-device','cuda','--max-calls','4']; runpy.run_path('temp/debug_full_pipeline.py',run_name='__main__')"
```

此混合导入方式只在本机当前包组合检查过；不是通用环境修复方案。不需要模型时仍直接使用`uav`环境。真实运行元数据记录解释器、torch/transformers实际包位置、版本、GPU、模型dtype和文件哈希，便于复核环境差异。4次预算仅适合短接口诊断，不能代表全程LLM优化效果。

## 6. 输出文件怎样看

- `debug_stages.jsonl`：逐阶段JSON行，包含顺序号、阶段名及可见状态；启动后持续追加，可在模型尚未生成时看进度。
- `call_0001_context.json`：第1次真实传给后端的结构化上下文。
- `call_0001_prompt.txt`：进入聊天模板前的提示词；不冒充tokenizer实际输入。
- `call_0001_model_input.txt`：仅真实后端取得实际模板输入后生成；包含聊天模板的分词前文本。
- `call_0001_response.txt`、`calls.jsonl`：原始响应、词元数、耗时及错误；响应不作修复。
- `episode.json`、`metrics.json`、`decision_log.txt`：正式闭环格式结果与中文日志，和标准入口一致。
- `evaluation_only_config.json`：回放／评价配置，含隐藏场脚本，**不是LLM输入**。
- `run_metadata.json`：启动、模型加载、仿真、保存、完成或失败状态；中断时保留已完成的调用和阶段记录。
- 可选`summary.png`、`replay.gif`：左右图显示规则与正式回放一致，真值只在研究者侧。

旧`temp/run_source_interface_check.py`和审计脚本固定于2026-10-29高斯接口，不用于当前传播／方向协议；不要覆盖或将其旧结论套在新日志上。研究优先级仍只维护在`docs/current_roadmap.md`。

## 7. 本次已跑通的记录

- `outputs/interference_debug_numeric_20261102/`：默认配置无模型完整任务，53次观测、8次决策，755.073m到达；用于先理解全部阶段。
- `outputs/interference_source_local_debug_20261102/`：当前传播／测向协议4次真实调用的完整记录，只有前2次成功生成；首次方向发现时OOM，下一次输入超限。不是4次有效决策，也不是全程模型收益；细节见其中`review.md`。
- 想不等待本地模型，先打开已有`call_0003_context.json`看首次方向证据，再对照`calls.jsonl`的显存错误和`first_direction_fallback.png`。源评价未发生，所以真实日志没有`source_evaluated`阶段；不要把缺失误认为断点无效。
- `mock`默认也不提源；功率／方向拒绝与修订的确定性接口日志另见`outputs/interference_source_direction_mock_20261031_193820/`。这是夹具，不是真实模型推断。
