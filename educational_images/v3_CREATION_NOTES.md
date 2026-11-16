# V3 educational image creation notes

七张 V3 教学图均由
[`scripts/generate_v3_educational_images.py`](../scripts/generate_v3_educational_images.py)
读取 [`reports/v3_evidence_summary.json`](../reports/v3_evidence_summary.json)
后确定性生成。输出合同为 3:4 portrait、1800×2400 PNG 与可编辑 SVG；脚本使用固定
NumPy seed，并在本机检测到 `Microsoft YaHei` 后才允许渲染，避免中文静默回退到
DejaVu Sans。

本轮真实打开并检查了两张 V2 风格参考图：

- `v2_rule_flow_comparison.png`
- `v2_pipeline.png`

继承的是纵向教材海报的信息层级、圆角章节、信号插图和有限强调色，不复制旧图的文字、
布局或科学错误。Image generation skill 已检查；由于精确文本、阈值、数字和 SVG 可编辑性
必须由代码控制，本轮没有使用生成式位图模型。

| Figure | Scientific purpose | Synthetic / visual content | Primary fact source |
|---|---|---|---|
| `v3_01_events_to_stages_pass1_pass2` | 解释 main Legacy V1 整夜 Pass 1 与仅处理 U 的 Pass 2 | PSG、事件、证据与阶段示意 | remote main `src/hmc_sbtb/rules.py` at `9dd964c…` |
| `v3_02_spindle_yasa_wonambi` | DREAMS spindle Recall/F1 与 FP trade-off | 13.2-Hz tapered spindle burst | sealed spindle event metrics |
| `v3_03_kcomplex_mtkcd` | K-complex FP reduction 与 HMC 负结果 | 双相 K-complex | sealed K event metrics + HMC paired counts |
| `v3_04_arousal_unet` | Frozen U-Net proposal path 与事件级结果 | EEG/EOG/Chin EMG arousal-like traces | sealed PSG-IPA event metrics |
| `v3_05_rem_antiphase` | Bilateral EOG AntiPhase 验证 | opposite-polarity EOG movements | sealed REM event metrics |
| `v3_06_event_to_stage_ablation` | 固定 R1 下 A/K/S 八臂消融 | 8-arm matrix + zero-based metric chart | sealed HMC integration metrics |
| `v3_07_remaining_bottlenecks` | Wake/N3 全 epoch 可观察失败机制 | aggregate bars + 30-s duration timelines | sealed Wake/N3 audit |

## Visual review record

每张图均实际检查了三种视图：1800×2400 原图、800 px GitHub README 预览、关键波形或
图表局部放大。首轮发现并修复了候选时间轴与说明文字距离过近、Figure 06 图例与指标卡
遮挡、Figure 07 Wake 长标签裁切和 N3 时间条文字拥挤。第二轮未见中文缺字、图文重叠、
裁切、缺失箭头或误导性坐标截断。随后在两个独立输出目录完整复渲染，14/14 PNG/SVG
文件的 SHA256 逐文件一致。详细记录见
[`v3_visual_quality_audit.md`](v3_visual_quality_audit.md)。

含波形的 Figure 02–05 均在图内标明“合成教学波形 · 非临床 PSG”。这些图是教学辅助，
machine-readable JSON 仍是数字 source of truth；stage label 也不被当作 event truth。
