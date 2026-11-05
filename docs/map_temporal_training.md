# 保留空间结构的时序约束训练

更新：2026-11-05。当前阶段仍是地图编解码，不运行LLM微调、RL或在线代码。主数据继续使用datasets/interference_onboard_maps_v1，不重新采集／划分，不修改历史帧或原始标签。

## 1. 本次改变了什么

在原v2最佳权重上微调同一个空间编解码器，保留每帧32×96瓶颈；不增加时序Transformer，不把4帧压成总共32个向量。每张地图仍只根据自身内容编码，未来帧不能影响历史帧编码。这里的“时序训练”是用前后两帧的监督关系约束表示，而不是已经具备源移动推理。

训练入口tools/train_map_temporal_autoencoder.py，实现src/uav_em/map_learning/temporal_training.py。继承boundary_v2单帧功率、支持边界、不确定性、年龄、不可飞与楼高目标，并加入：

1. 同一任务ROI中前后共支持格的功率差监督。用实际较早／当前原图差值作目标，不要求所有变化为零；未知填充值不作差值标签。
2. 共支持格的年龄差监督，允许刷新后年龄下降，不能把信息变新强行惩罚为不平滑。
3. 支持新增／失效的概率监督，分别由前后支持概率计算转移概率；不能用原图mask修正模型输出。

辅助功率项以10dB为归一化尺度使用SmoothL1；年龄输入本身以35s归一化。功率差至少3dB的变化格和平稳格分别求平均后等权组合，缺少某类时只用非空类。总目标为原空间目标＋0.5×功率差项＋0.1×年龄差项＋0.1×支持转移项。尺度／权重是研究配方，不是设备或安全阈值。

## 2. 样本与信息边界

读取原temporal_tasks，每题使用最早有效帧和当前帧，固定同世界坐标ROI；不是重新标注源运动。模型输入为两张机载地图，ROI只选择辅助损失位置，原图目标不作为额外模型输入旁路。

只在train按“是否至少3dB功率变化×是否支持增失”分层，对非空层分配相同采样总权重，带放回采样，每轮样本数仍等于train题数。目标是避免平稳题淹没变化题；不按隐藏源数、移动脚本或test结果采样。单帧空间损失覆盖整图；水平／垂直增强必须两帧六通道及ROI同步，不能用于validation。

validation保持全部题的自然分布，不加权采样。核对任务文件、轨迹哈希、场景划分、因果窗口、原始任务标签；正式训练加--require-quality检查已有完整审核。训练器不会打开evaluation_only或test轨迹文件，不会再执行test评价。

## 3. 选模与回退

使用自然validation配对总目标选模，同时要求相对原v2基线：支持IoU下降不超过0.01，支持功率／强干扰功率MAE各增加不超过0.25dB。空间指标固定frame_stride=2，不把大量重复任务帧当单帧指标样本。

原权重保存为epoch=0候选。没有合格改善时保留原权重，而不是强行宣布新模型有效；连续4轮无合格改善早停，上限12轮。初始权重、基线指标、配方、采样分布、源码哈希与逐轮指标全部另存；该选模损失改善仍不等于功率变化误报改善，必须再用固定核查协议检查。

新checkpoint保持空间结构标记与boundary_v2兼容字段，另存training_stage、temporal_settings、parent_epoch和parent_checkpoint_sha256。这里epoch为本轮微调轮数，不是加上父模型的累计轮数；不保存优化器恢复状态。改变权重后不能混用旧tokens，后续投影器也必须绑定对应编码器哈希。

## 4. 复现

```bash
python tools/train_map_temporal_autoencoder.py --dataset datasets/interference_onboard_maps_v1 --checkpoint outputs/interference_map_encoder_20261104_174105_494050/best.pt --config configs/map_encoding_temporal.json --device cuda --require-quality
python tools/audit_map_temporal_codec.py --dataset datasets/interference_onboard_maps_v1 --checkpoint <新目录>/best.pt --device cuda
```

所有输出为新目录，省略output自动时间戳。配方字段epochs／batch_size／patience为正整数，seed为非负整数，learning_rate及三个辅助权重、change_db为有限正数，augment为布尔且仅用于train；CLI的--epochs可覆盖上限。

## 5. 实际记录

本次候选目录为`../outputs/interference_map_encoder_temporal_20261105_114745_542222/`，完成12轮CUDA训练，按既定配对目标与空间限制选择第8轮。train为8,072道题，validation为1,731道；分层数量及完整设置见training_config.json，原始数据／划分未改。

选模后的完整时序核查在`../outputs/interference_map_temporal_codec_20261105_115226_042619/`，仍为同一108条validation轨迹、5,433帧、1,731道题；未运行test，没有以test继续调参。

| 指标 | 原v2 | 时序约束候选 |
|---|---:|---:|
| 支持mask IoU，单帧validation | 0.97950 | 0.98418 |
| 支持功率MAE，dB，单帧validation | 2.28066 | 2.16490 |
| 强干扰功率MAE，dB，单帧validation | 2.23683 | 1.91071 |
| 共支持功率差MAE，dB，任务ROI | 1.32855 | 1.35408 |
| 至少3dB变化格功率差MAE，dB | 4.51302 | 3.92244 |
| 功率变化精度 | 0.35707 | 0.34405 |
| 功率变化召回 | 0.63846 | 0.70427 |
| 变化方向召回 | 0.62821 | 0.68974 |
| 共支持年龄差MAE，s | 3.71627 | 3.44498 |
| 新增支持召回 | 0.98479 | 0.99108 |
| 失效支持召回 | 0.96094 | 0.96441 |

这是混合结果：空间指标、变化格差值、年龄差与变化召回改善，但总体功率差MAE略升，假变化由1,345增至1,571，精度下降。不能宣称已解决时序误报，也不能只报告召回提升。该候选保留供对照，不自动取代原v2作为后续LLM输入；没有继续搜索第二套配方或查看test。

本轮23项新增时序训练／嵌入接口测试通过；全量423通过、2个既有缺失夹具失败（138.24s）。测试覆盖未知值不计功率差、真实变化不被抹平、同步翻转、分组采样、基线保留、训练不读test／隐藏源，以及连续嵌入、因果时间、padding及labels屏蔽。未补造或跳过夹具。

后续适配器张量接口已另行准备，见map_embedding_bridge.md；它没有训练对齐。下一步仍需在保留此次结果的前提下，明确如何控制功率变化误报与召回的权衡，不能把新接口可运行当作LLM理解地图或动态重规划有效。
