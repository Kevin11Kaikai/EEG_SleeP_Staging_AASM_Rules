# AASM Rule Discovery v5 — Executive Summary

## 结论

- 固定输入：`M_6408 = E04 + E11 + SUMOv2 Spindle + U-Net Arousal`。
- 规则候选：384个；只改decision logic，不改Detector、Event Bank、Evidence、QC或6秒/15秒生理阈值。
- Discovery选择：`C196`，规则补丁：`R.BACKWARD_BRIDGE, N2_POST_AROUSAL_REENTRY`。
- Confirmation科学状态：`HMC_CONFIRMATION_GAIN_SUPPORTED`。

## Confirmation split (5 nights)

| Metric | Temporal R1 | New rules | Delta |
|---|---:|---:|---:|
| Accuracy | 0.593306 | 0.606602 | +0.013297 |
| Macro-F1 | 0.482999 | 0.494475 | +0.011476 |
| Coverage | 0.992435 | 0.992664 | +0.000229 |
| W F1 | 0.208955 | 0.208955 | +0.000000 |
| N1 F1 | 0.423550 | 0.425926 | +0.002376 |
| N2 F1 | 0.698261 | 0.703469 | +0.005208 |
| N3 F1 | 0.335755 | 0.335755 | +0.000000 |
| R F1 | 0.748476 | 0.798271 | +0.049795 |

Changed=90，Corrected=72，Introduced=14，Net=58。

Recording-level bootstrap 95% CI：Accuracy [0.003324625979577389, 0.023126734505087843]；Macro-F1 [0.0024007690953149474, 0.018443042311455105]。

## 关键机制

Temporal R1缺少AASM公开更新中的definite-R backward bridge。旧R.2只能从已有R向后续Epoch延续；新规则允许未来definite R为其之前连续、LAMF占多数、低肌张力、无K/spindle/arousal/SEM的Epoch提供R上下文。C196还测试了arousal后合格K/spindle结构证据优先恢复N2；AASM支持segment-level重入，但whole-epoch优先于majority是本项目工程操作化。Wake/N1/fallback候选也被测试，但未进入冻结winner。

实际累计执行时间：2805.73秒。结果仍属于反复使用HMC开发集上的post-hoc confirmation，不是外部临床验证；Full-development结果只用于描述，不能替代Confirmation。
