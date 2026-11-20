# AASM V4 Phase 2B Executive Summary

**工程结论：36/36 臂、360/360 Night-Arm 的固定结果已从逐 Epoch Prediction/Evidence/Trace 独立复算；未训练、未调参、未修改规则。**

## 三个最重要的真实发现

1. **Full-15 的 Macro-F1 上升不是整体更准。** 相对 B000，Macro-F1 `+0.034211`，但 Accuracy `-0.016738`、N3 F1 `-0.116602`、Coverage `-0.069086`。共同接受的 Epoch 净改善 `+183`，Coverage 转换净损失 `-332`，闭合为 `-149` 个净正确 Epoch。
2. **下游效用取决于 rule-qualified evidence，不取决于 Event count 单调增加。** E01 的慢波数量增加，但总持续时间从 `17337.4s` 降至 `5957.9s`，削弱 6 秒 N3.1；E02 的 K 候选从 `13719` 降到 `2060`，N2 降但 REM 升；AR/SP 的新增证据则能被 reset/N2 structural predicates 稳定消费。
3. **更多 Evidence 可能降低 Coverage。** E13 从 `36` 个 frozen MBM 候选增至 `13916` 个 proxy 候选，单换 Macro-F1 `+0.029326`，但 Coverage `-0.022804`；其 movement preemption 与 pending/unknown 路径直接制造拒判，E13 仍只能称 `MOVEMENT_PROXY`。

## 两个关键机制

- **Duration/overlap semantic alignment**：Detector 输出必须跨过当前规则真正消费的持续时间、重叠与关联条件；原始候选数或独立 Event F1 不是充分统计量。
- **Temporal propagation and preemption**：当前 Epoch 的 Evidence 会改变后续 Temporal State；Full-15 的多源变化多数不可唯一归因。Movement 还会在普通规则前抢占控制流。

## 首选研究问题与最小实验

**研究问题：Rule-relevant evidence sufficiency（合格持续时间、overlap、association）是否比 Event F1 更能预测 Detector 替换的 Stage utility？**

最小可证伪实验：在一组未用于开发、同时具有专家事件与睡眠分期标注的 PSG 上，预注册 SP/AR/E01/E02 的冻结 Detector、Eventization、Evidence 映射和规则；同时测量 Event F1、rule-qualified evidence coverage 与固定规则 Stage Δ。若 rule-qualified 指标不能比 Event F1 更稳定地解释跨夜 Stage Δ，则该假设被否定。

## 科学边界

这是反复使用的十晚 HMC 开发集机制研究。Recording-level Full-15 ΔAccuracy 95% CI 为 `[-0.088972, +0.026442]`，ΔMacro-F1 为 `[-0.000113, +0.076059]`；没有使用 withheld final set，也没有外部泛化或多数 Handler 的 HMC 事件级真值。观察、Trace-supported mechanism 与待验证假设在完整报告中分开标记。
