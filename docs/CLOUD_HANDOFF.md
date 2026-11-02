# 云端迁移与新agent接手

日期：2026-11-02。用户因本地显存不足转到云端，并将在云端开启新的agent。本次只整理迁移、文档和Git入库范围，没有改传感器、仿真、模型提示词或规划算法。研究顺序仍唯一维护在`current_roadmap.md`，这里不另立研究路线。

## 0. 发布状态与本机最后一步

迁移文件和忽略规则已在工作区准备好，**当前执行环境未能提交或上传**：`.git`为只读，`git add`创建`index.lock`时被拒绝；查询既有GitHub远程也因无法连接443端口失败。这不是已经推送成功，也没有改写Git历史、凭据或远程地址。当前分支为main，已有origin，无需重新初始化仓库。

请用户在可写且能联网的本机终端进入项目根目录，复核清单后执行：

```powershell
git status --short
git add .
git diff --cached --stat
git diff --cached --check
git commit -m "Prepare cloud handoff and reproducible project layout"
git push origin main
git rev-parse HEAD
```

本次规则已排除权重、ZIP、凭据文件、缓存、原始docx及非精选输出。推送失败时保留错误并处理连接／权限或远程分支分歧，不使用force覆盖远程。上传后把最终提交号交给云端agent，并确认克隆中存在本文件、完整tests及调试入口；下文“Git包含／入库”描述的是准备好的发布范围，需以上步骤成功后才出现在远端。

## 1. 接手先读什么

发布状态记录的是本次整理时的限制；用户手动发布后以实际提交号为准。云端若已经克隆到本文件且文件清单完整，不要因为第0节的历史状态再次阻塞接手或重复要求提交。

1. 根目录`AGENTS.md`：维护约定和禁止越界事项。
2. `AGENT_HANDOFF.md`：已实现能力、最近实际运行、已知问题。
3. `current_roadmap.md`第6节：下一步与阶段验收，不能跳到训练或代码生成。
4. `first_version.md`：真实流程、信息隔离、源协议、日志和指标。
5. `../temp/debug_full_pipeline.md`：直接进入正式闭环的IDE／pdb操作。

当前是部分可观测时变电磁干扰下的非代码LLM辅助决策。观测为机载总功率与可选含误差测向，不是全域空间扫描，也不直接给源位置。模型提出行动和可选传播源假设，数值层检验后用于预测地图和传统路径规划。默认RRT*；A*仅对照。强干扰是软代价，没有硬禁入或真实飞行安全保证。

## 2. 迁移带走哪些文件

Git包含正式源码、配置、所有测试、维护／研究文档、基础依赖文件，以及`temp/debug_full_pipeline.py`和同名说明。保留原目录与接口，不建立第二套仿真循环。

精选三份复现记录已经配置为入库：

| 目录，均位于`outputs/` | 云端用途 |
|---|---|
| `interference_source_local_debug_20261102/` | 当前传播协议真实模型失败诊断；含逐调用上下文、原文、实际模板输入、episode、指标、审计与PNG |
| `interference_debug_numeric_20261102/` | 无模型完整闭环；含逐阶段日志、episode、指标和PNG |
| `interference_source_direction_mock_20261031_193820/` | 确定性夹具展示功率拟合但方向错误被拒绝、修订后接受；不是LLM能力证明 |

不入库且本地原样保留：`models/`权重、其他历史输出、GIF、IDE文件、第三方／推理缓存、`docs/submissions/`原始docx、根目录`uav.zip`。旧压缩包可能缺少最后的诊断记录，不作为迁移入口。云端无需这些缓存或旧调试脚本；完整代码不依赖旧项目。

新实验输出仍默认被Git忽略，使用新的`outputs/interference_...`目录，不覆盖三份入库证据。正式入口重复指定旧目录会覆盖结果，优先使用调试入口的自动时间戳新目录。

## 3. Linux环境与权重

从GitHub克隆本项目，进入仓库根目录并确认实际版本：

```bash
git clone https://github.com/mangziwaer2/uav.git
cd uav
git rev-parse HEAD
python --version
nvidia-smi
```

先确认用户已经推送本次迁移提交，克隆中应存在本文件、`tests/`和指定调试入口。若仓库只有最初源码，请先同步迁移提交，不用重新实现缺失文件。

建议用云端已配置CUDA的Python 3.10环境；激活正确环境后安装基础依赖：

```bash
python -m pip install -r requirements.txt
```

基础测试与无模型仿真不需要torch、transformers或模型权重。需要真实LLM时，先检查云端的PyTorch及GPU是否可用，再处理`requirements-llm.txt`，不要直接覆盖服务商预装的CUDA/PyTorch。该文件是版本范围约束，不是严格锁文件；实际版本必须另行记录。若CUDA不可用或依赖不满足，先解决环境，不把CPU回退当成GPU测速。

```bash
python -c "import sys,torch; print(sys.executable); print(torch.__version__,torch.version.cuda); print('cuda_available=',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
# 确认环境满足条件后，才按需安装可选推理依赖
python -m pip install -r requirements-llm.txt
python -c "import torch,transformers; print(torch.__version__,transformers.__version__); print(torch.__file__); print(transformers.__file__)"
```

本地历史运行使用uav Python 3.10.22加载toolkit的torch 2.11.0+cu126／transformers 4.54.1，RTX 3060 Laptop 6GB、float16。原因是本地toolkit解释器本身启动异常。这只是历史事实，**云端不要复制Windows绝对路径或混合site-packages的临时方案**；本次没有在Linux/GPU云主机实际验证安装。

模型权重由用户另行传输，先保持现有`Qwen3-1.7B`以区分资源问题与模型变更。默认位置为`models/Qwen3-1.7B/`；也可以使用`--model-path /实际模型路径`，这个参数是本地权重目录，不是HTTP接口。

原目录包含`config.json`、`generation_config.json`、tokenizer相关文件、`model.safetensors.index.json`及两个safetensors分片；不能只复制一个分片或配置文件。传输后校验文件完整性，条件允许时比较源端／目标端SHA256。当前程序不自动下载、不自动换模型，也没有远程推理客户端。服务商与模型下载凭据放环境或外部安全存储，不写入仓库。

## 4. 先复现，再改动

在仓库根目录运行：

```bash
python -m pytest -q -p no:cacheprovider
python temp/debug_full_pipeline.py --advisor none --quiet --plot
python temp/debug_full_pipeline.py --advisor mock --max-seconds 2 --max-calls 1 --quiet
```

最近本地全量回归为251项通过，92.23s；这是迁移前Windows记录，不是云端已经通过。测试由`tests/conftest.py`加入src路径，不需要安装本项目为Python包。`test_debug_pipeline.py`会使用项目`temp/`下自清理临时目录，仓库根目录应可写。2秒mock运行通常以time_limit结束，只检验接线，不是完整成功任务，更不是模型收益。

调试时在`PipelineDebugger.__call__`看`stage/data`，或从`main`步入`simulate`。`source_evaluated`只有模型真正提交源假设才出现，默认none/mock不会触发。断点不推进仿真时间，但会影响外层墙钟耗时，不用于GPU测速。

权重和CUDA就绪后，真实接口可用下列命令短诊断；输出自动写入新目录，生成不限时：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python temp/debug_full_pipeline.py --advisor local --model-device cuda --max-calls 4
```

4次预算只作接口诊断，不是默认24次的全任务评价；不必反复重跑低信息前缀。可先受控复用已保存的第3轮可见上下文检查云端生成，但必须标为离线上下文诊断，不伪装成新飞行或新观测。输出解析、源一致性与闭环效果是不同验收环节，不能仅看JSON合法。

**更多显存不会解除8000输入词元上限。** `LocalModelConfig.max_input_tokens`当前默认8000；上述两个CLI均未提供修改它的参数，不要使用不存在的`--model-max-input-tokens`。第4轮8736词元在GPU生成之前就会被拒绝，换4090仍会如此。既有路线要求受控验证数值精度压缩、保留证据语义；若需要显式可配置输入上限，也必须记录改动与实际设置，不能偷偷截断或假定问题自动消失。

确认输入策略及资源后，再按路线第6节覆盖足够调用预算的完整飞行阶段；默认任务预算24、每点最多3轮，每轮均计入预算。扩大调用数只增加可咨询次数，不提高单轮输入上限，也不保证有效源推断。

## 5. 必须带着走的失败结论

迁移补充验证：按Git实际候选清单建立不含`.git`、权重和其他忽略文件的临时副本，251项测试通过（82.69s），副本已清理。候选共91个文件、约10.65MiB，检查了必要文件、大文件及常见凭据模式。这证明待上传文件在本地自足，不等于云端CUDA已验证或GitHub已上传。

最新真实诊断详见`../outputs/interference_source_local_debug_20261102/review.md`：

- 第1／2轮输入4895／7107词元，完整输出109／97词元，分别补测和重规划。当时尚无可检测角峰；两次均省略源字段，不是显式零源。
- 第3轮首次检测方向时输入7532词元，CUDA OOM，无回答；第4轮8736词元，超8000上限、生成前拒绝。不能把这两轮解释为LLM忽略方向证据或定位能力不足。
- 9次决策、4次后端调用、2次后端失败、7次回退（2次失败＋5次预算耗尽）；源提议／源评价／接受均0。到达不是全程LLM成功，真实拒绝反馈后的源修订仍未验证。
- 离线将浮点保留3位小数，同tokenizer下4轮输入变为4479／5918／6056／6703。只做了计数，没有改变真实输入，也没有证明压缩后决策质量或GPU内存收益。
- 旧metadata和audit中的Windows源码路径／哈希描述当时环境，不应用于定位云端文件；云端使用当前相对路径，重新记录实际命令、提交、包版本、GPU、dtype及输入输出。

`call_0003_context.json`和`call_0004_context.json`才是需要受控检查的有方向证据输入。`evaluation_only_config.json`及episode顶层的评价配置含隐藏真值，只可由仿真器／回放读取，不能整体送给LLM。数值前向检验只用公开传播参数，不替LLM自动反演源位置；不把真源坐标加噪声当成传感器输出。

GIF没有入库，可离线重建，不重新推理或规划：

```bash
python src/replay_experiment.py outputs/interference_source_local_debug_20261102/episode.json --output outputs/interference_source_cloud_replay/replay.gif --speed 1 --decision-pause 3
```

## 6. 给云端新agent的起始消息

可直接复制以下内容，不要让新agent从历史对话猜状态：

> 请先读根目录AGENTS.md，以及docs/AGENT_HANDOFF.md、docs/current_roadmap.md、docs/first_version.md、docs/CLOUD_HANDOFF.md。继续当前非代码LLM研究主线，不复制仿真、不扩大传感器噪声、不引入训练／门控／在线代码执行。先检查云端CUDA环境和权重，运行251项基线测试与无模型完整调试流程。再根据已入库的当前传播协议第3／4轮上下文，受控验证数值精度压缩与真实模型生成，保留原始观测、未知／时效和测向语义；8000输入上限不会因迁移GPU自动解除。随后按路线第6节检查真实源提议、数值拒绝反馈修订和补测规划。每次输出写新目录，完整保存输入、原文、实际模板、词元数、GPU／版本、失败及回退。源推断尚未真实验证，不以夹具成功或回退到达宣称LLM收益；局部代码生成仍在后续计划，不在这一轮开启。
