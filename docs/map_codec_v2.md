# 地图编解码器v2：支持边界与局部结构改进

更新：2026-11-04。当前数据为datasets/interference_onboard_maps_v1，编解码器v2不改变数据、分组或标签。研究仍是后续LLM辅助动态重规划；本轮不训练LLM或RL，不扩大传播环境。

## 1. 诊断依据

对旧v1最佳checkpoint只重新检查validation，报告在`../outputs/interference_map_codec_diagnostic_20261104_172932_498897/validation_v1.json`：

- 支持召回率0.98357，但精度0.55232；未知格误报率0.06618。不是没有学到测量位置，而是把过多未知边界恢复成支持。
- 障碍IoU约0.25752；总验证loss中blocked项约0.34682、support项约0.11947，已超过其他回归项。
- 旧解码器将4×8空间锚点双线性铺成整张地图，再逐点输出；结构与类别平衡损失共同可能造成粗边界与泛化不足。此处是根据结构和误差提出的工程解释，不是单因素消融证明。

没有用test选择修改方案。v1历史test已检查过，v2最终test仅作这一候选的描述性检查，不把同一test已被观察过的结果冒充全新盲测；进一步论文验收须新增未用过的独立场景。

## 2. 修改内容

1. 保持瓶颈32×96、输入6通道、patch=4不变，不通过增加LLM地图token数掩盖问题。
2. v2用局部3×3卷积提取空间特征，再进入原空间Transformer与32锚点瓶颈。解码先恢复patch网格，再经卷积与PixelShuffle还原格点，最后裁去padding；输出只依赖tokens与公开尺寸，没有原图、support或建筑真值旁路。
3. `boundary_v2`对support／blocked使用边界加权的自然比例BCE与逐图Dice，抑制把未知扩大为支持；功率和年龄仍只监督有支持格，未知功率不作标签。
4. 训练地图随机水平／垂直翻转，6通道同步，不能用于validation／test，也不改写原始坐标、观测和缓存。地图自编码训练不使用方向日志，翻转不是给后续LLM伪造测向。
5. validation连续8轮总loss未改善则早停，上限40轮；按validation选best.pt，选定后才检查一次test。
6. 日志增加分通道loss、支持精度／召回率、未知误报率、障碍IoU、不确定性及楼高MAE，避免只看混合平均loss。

v1结构和旧权重继续明确支持；省略architecture仍表示v1，不静默换成v2。新checkpoint保存结构标记、目标函数和设置，二者不一致时拒绝。历史v1与v2总loss定义不同，不能直接拿loss的数值下降算相对提升；比较同一划分上的物理单位误差及mask指标。

## 3. 配置与复现

```bash
python tools/train_map_autoencoder.py --config configs/map_encoding_v2.json --dataset datasets/interference_onboard_maps_v1 --require-quality --device cuda --output outputs/interference_map_encoder_v2_reproduce01
```

输出必须为新目录。训练seed=7，batch=32，学习率0.0003，frame_stride=2；augment=true只启用上述翻转。epochs为上限，patience为正整数轮数，null表示关闭；objective只接受balanced_v1或boundary_v2。encoder.architecture只接受spatial_anchor_local_decode_v1或spatial_patch_decode_v2。

正式数据迁移只变存放位置，manifest SHA256保持740492a8f3d851a04462bef7e7306a41fbed558b3ee2bfefc78fdbb605c5e91a。既有训练／预览元数据保留原运行时路径，不能改写成看似当时就在datasets下。

## 4. 实际运行

已完成40轮CUDA训练，validation选择第39轮，未触发早停。权重与最终指标在`../outputs/interference_map_encoder_20261104_174105_494050/`；best.pt为当前v2候选。迁移后严格数据审核通过，720条轨迹零错误。

### Validation：改进依据与选模结果

两者均用frame_stride=2，共2,741帧；比较各自validation选出的checkpoint。

| 指标 | v1 | v2 |
|---|---:|---:|
| 支持mask IoU，越大越好 | 0.54727 | 0.97950 |
| 支持精度／召回率 | 0.55232／0.98357 | 0.99237／0.98693 |
| 未知格误报率，越小越好 | 0.06618 | 0.00063 |
| 障碍mask IoU，越大越好 | 0.25752 | 0.87033 |
| 支持功率MAE，dB | 2.26714 | 2.28066 |
| 强干扰功率MAE，dB | 2.36017 | 2.23683 |
| 支持年龄MAE，s | 2.98105 | 2.11392 |

边界和局部结构改善明显，整体功率误差并未降低。瓶颈仍是每帧32×96，但参数量由302,982增至683,142；这是结构、损失及增强的组合改进，不能宣称等算力对照或单因素效果。

### Test：选模后的描述性检查

v2选模后检查一次，共2,767帧。v1数值来自历史结果，未重新用test调参；本表不是新的盲测。

| 指标 | 历史v1 | v2 |
|---|---:|---:|
| 支持mask IoU | 0.53572 | 0.97625 |
| 支持功率MAE，dB | 2.47709 | 2.47988 |
| 强干扰功率MAE，dB | 2.91424 | 2.37037 |
| 支持年龄MAE，s | 2.89766 | 2.07699 |

v2未知格误报率0.00069、障碍IoU 0.88218；不确定性MAE 1.33669dB、楼高MAE 2.75742m。完整指标以metrics.json为准，不用混合loss比较v1／v2。

### 看图与边界

打开`../outputs/interference_map_codec_comparison_20261104_175339_533173/README.md`，其中goal／revisit各有一张comparison.png：三列依次是机载原图、v1、v2，两行分别比较有支持功率与支持mask。均为validation样本，不读取隐藏源或全图真值。原始六通道、历史窗口和时序题另见正式数据的previews目录。

全量回归386通过、2个既有缺失夹具失败；地图相关定向51项通过。未补造或跳过夹具。

当前只验证逐帧重建；历史窗口导出不等于已学会时序推理。后续先核查同一区域的估计变化、支持过期、强干扰与时间保真，再适配LLM。近似解码不能代替公开几何的硬约束，也不能据此声称安全、源定位或动态重规划收益；尚未训练LLM或RL。

导出的tokens元数据记录结构、设置、目标函数、checkpoint及训练manifest哈希。不同权重或结构的缓存不能混用；32个连续潜在向量也不等于已经接入LLM的32个文本词元。
