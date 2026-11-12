# 实际执行方法

PSG → 原生质量检查和μV单位 → 有效块128Hz预处理 → 生理候选事件 → 区间并集/窗口证据 → 分期规则 → StageDecision + trace。原始评分只在独立评价器打开。

`aasm_v2`是已冻结的新工程实现：data负责EDF结构和物理量；preprocessing处理质量、重采样、滤波与局部频谱；events产生有来源的事件及失败状态；evidence汇总30秒覆盖、N2窗口、事件关联；rules包含独立重建original-style与enhanced引擎；provenance保存哈希与checkpoint。代码不import旧hmc_sbtb工程。

V0/V1采用全夜明确规则pass及顺序continuity/inheritance；V2/V3/V4采用冻结增强状态机。增强主实验明确使用candidate_proxy，不把临床UNKNOWN伪装成已验证阴性。严格证据与条件继承是单独敏感性输出，未替代主V3。

R1接口为`aasm_temporal_r1.stage(event_bank)`，输入仍是同一增强事件银行。实际伪代码：

```text
rows = frozen_aggregate(events)
context = empty physiological EndContext
for epoch in chronological order:
    recover valid prior-last-15s N2 structure anchor, if eligible
    split at actual arousal / LAMF / K / spindle boundaries
    preserve N2 before arousal
    invalidate N2 anchor at arousal onset
    count arousal interval as UNKNOWN
    after offset: count only actual LAMF as N1
    at new qualifying structure: reenter N2
    preserve frozen N3/W, R.3 and movement priority
    test definite R.1 before N2-arousal fragment assignment
    if N2-arousal context engaged:
        assign N2 or N1 only when respective time >15s / fixed30s
    otherwise continue unchanged remaining rule predicates
    if unresolved: call frozen INHERIT_PREVIOUS(previous output label)
    store epoch label separately from physiological end context
    never let fallback resurrect an invalidated N2 anchor
resolve MBM with unchanged offline algorithm-prediction lookahead
```

这不是一般REM/N1/N3/MBM重写。Arousal后片段模块限定在N2中断上下文；无Arousal的普通N2continuation不扩展。R.3.N2保留当前标签优先级，但不抹掉其后的Arousal末态。

每个决策保留rule_id、输入状态、事件ID、来源、预测anchor/chain、片段账本、生理anchor有效性与Arousal起源。完整病例级trace只存在本地运行产物，公开仓库不包含患者时间轴。
