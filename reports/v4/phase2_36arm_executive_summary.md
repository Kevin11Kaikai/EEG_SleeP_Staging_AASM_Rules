# AASM V4 Phase 2 Executive Summary

## 结论

本轮真实完成 36/36 臂与 360/360 个夜晚-臂重放，15 个神经来源均在读取 HMC Stage Reference 前完成 label-blind 封存；B000 的 Event Bank、Evidence 与 Temporal R1 Prediction 在十晚全部精确回归。工程状态：PHASE2_36_ARM_ENGINEERING_PASS；有效替换状态：FULL_15_HANDLER_REPLACEMENT_ESTABLISHED。

F_ALL15 没有整体胜出：相对 B000，Accuracy -0.016738，Macro-F1 +0.034211，N3 F1 -0.116602，Coverage -0.069086。它提高了 Wake、N1 和 REM F1，却以 Accuracy、N2、N3 和 Coverage 回退为代价。

最可靠的单源是 Arousal U-Net：Delta Accuracy +0.018760、Delta Macro-F1 +0.030609，recording-bootstrap 95% CI 分别为 [+0.003904, +0.032809] 与 [+0.018582, +0.039870]。SUMOv2 Spindle 提高 Accuracy/N2，但单换 Macro-F1 略降；它在完整系统中的 leave-one-out 贡献为正。K08 提高部分宏观类别表现和 REM，却使 N2 F1 下降 -0.039213；SSSM Slow-wave 使 N3 F1 下降 -0.083252。E13 的 Wake/Macro-F1 表面收益伴随 Coverage 下降 -0.022804，且仍是 movement proxy，不能采纳为正式 MBM Detector。

科学结论：DETECTOR_TO_STAGE_HYPOTHESIS_PARTIALLY_SUPPORTED；DETECTOR_GAIN_NOT_SUFFICIENT；NEGATIVE_CROSS_EVENT_INTERACTIONS；EVIDENCE_QUALITY_LIMITED；INCONCLUSIVE_DUE_TO_INPUT_OR_VALIDATION_LIMITS。HMC 没有这些事件的独立事件级真值，全部 montage transfer 仍未独立验证。

实际总 Wall Time：2068.8 秒。

## 核心对照

| Arm | Accuracy | Macro-F1 | Wake F1 | N1 F1 | N2 F1 | N3 F1 | REM F1 |
|---|---|---|---|---|---|---|---|
| B000 | 0.519996 | 0.400815 | 0.109149 | 0.227042 | 0.636655 | 0.359604 | 0.671624 |
| F_ALL15 | 0.503258 | 0.435026 | 0.255014 | 0.316825 | 0.622289 | 0.243001 | 0.738003 |
