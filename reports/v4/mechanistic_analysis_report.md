# Final Phase 2B Mechanistic Analysis Report

## Scope and status

- Frozen source: the aggregate artifacts of the sealed Phase 2 36-arm development run.
- Primary: 8902 epochs; full-valid: 9211 epochs.
- Arms/night replays: 36 / 360.
- No training, detector search, HMC tuning, rule modification, withheld final-set access, or out-of-scope record enumeration.

## Q1. 为什么 Macro-F1 上升，而 Accuracy、N3 F1、Coverage 下降？

Full-15 将 W/N1/REM F1 分别改变 `+0.145865` / `+0.089783` / `+0.066378`，同时 N2/N3 F1 改变 `-0.014367` / `-0.116602`。Macro-F1 对五类等权，因此低召回阶段的改善可超过 N2/N3 的 macro 贡献损失；Accuracy 则受 N2/N3 大支持度和拒判影响。Coverage 从 0.993934 降至 0.924848：Accepted→UNDETERMINED 634，反向 19。共同接受 strata 为 +183，Coverage strata 为 −332，总和 −149。

## Q2. 哪些 Handler 发生 context reversal？

- Accuracy `Single+ / Full−`: E07, E13；`Single− / Full+`: E12。
- Macro-F1 `Single+ / Full−`: E04, E06；`Single− / Full+`: E08, SP。

主要机制不是单一“协同”：N2 组共享 structural/reset/continuation 路径；ocular 组共享 Wake/N1/REM 分支；EMG heads 彼此相关且 E13 会 preempt；Temporal state 又把早期差异传播到后续 Epoch。因此 LOO 是当前 Full context 的条件边际，不是普适生理因果效应。

## Q3. K08、SSSM Slow-wave、Movement Proxy 在哪里破坏链条？

| Source | Broken layer | Frozen evidence |
|---|---|---|
| E01 SSSM | Event duration → evidence threshold | Event 总时长 `17337.4→5957.9s`; 6-s down-crossing 396; ΔN3 F1 -0.083252 |
| E02 K08 | Sparse K / association → N2 structural evidence → temporal fallback/REM | Event `13719→2060`; associated `314→66`; ΔN2 -0.039213, ΔREM +0.097081 |
| E13 proxy | Event density → movement preemption/pending → abstention | Event `36→13916`; ΔCoverage -0.022804; proxy semantics only |

这些是 rule trace 支持的传播机制；它们不等价于在 HMC 上测得 Event FP/FN，因为对应独立 Event Truth 不存在。

## Q4. 为什么 Arousal 和 Spindle 相对有效？

AR 单换 ΔAccuracy `+0.018760`、ΔMacro-F1 `+0.030609`；它改变持续时间/重叠、K-association、N2 reset 和 post-arousal 路径，而不只是增加候选。SP 单换 ΔAccuracy `+0.007639`、ΔN2 F1 `+0.013758`；`n2_spindle` 可直接进入 N2.1。共同点是 Detector output 与现有 predicate 的 temporal semantics 对齐。二者仍只是内部开发证据。

## Q5. 最值得研究的问题

首选问题是：**Event-level accuracy 与 rule-relevant evidence sufficiency 哪一个真正决定 downstream utility，以及这种关系是否跨数据集成立？** 详见 `PAPER_RESEARCH_QUESTIONS.md`。最低限度需要一套新 PSG cohort 同时提供事件和 Stage Truth，并在完全冻结的 eventization/evidence/rule 方案下验证。

## Group nonadditivity and uncertainty

四组的完整 residual 位于 `grouped/group_interactions.csv`。Full-15 recording-equal ΔAccuracy mean `-0.019633`（positive nights 40.0%），ΔMacro-F1 mean `+0.036818`（positive nights 70.0%）。这些十晚区间没有多重比较校正，也不是外部统计认证。

## Evidence classification

- `OBSERVED_STAGE_EFFECT`: 所有固定 Mask 指标与 paired error movement。
- `TRACE_SUPPORTED_MECHANISM`: E01 duration→N3.1；E13 movement preemption→abstention；AR/SP rule-consumable paths。
- `CONTEXT_DEPENDENT_INTERACTION`: Single/LOO sign reversals and grouped residuals。
- `METRIC_COVERAGE_TRADEOFF`: Full-15 和 E13。
- `HYPOTHESIS_REQUIRING_VALIDATION`: rule-relevant sufficiency predicts Stage utility。
- `UNRESOLVED_ATTRIBUTION`: Full-15 individual-source causal allocation and epochs with multiple simultaneous evidence/temporal changes。
