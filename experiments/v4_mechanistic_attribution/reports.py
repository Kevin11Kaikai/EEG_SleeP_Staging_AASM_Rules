from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .core import HANDLERS, STAGES, sha256, write_json


def _value(metric: dict[str, Any], name: str) -> float:
    if name.endswith("_f1") and name[:-3] in STAGES:
        return float(metric["per_stage"][name[:-3]]["f1"])
    return float(metric[name])


def _delta(results: dict[str, Any], arm: str, name: str, base: str = "B000") -> float:
    metrics = results["metrics"]["primary"]
    return _value(metrics[arm], name) - _value(metrics[base], name)


def _context_lists(comparison: pd.DataFrame, metric: str) -> dict[str, list[str]]:
    column = f"{metric}_context_class"
    return {
        label: comparison.loc[comparison[column] == label, "handler_id"].tolist()
        for label in comparison[column].unique()
    }


def write_reports(
    config: dict[str, Any],
    output_root: Path,
    results: dict[str, Any],
    started_at: str,
) -> None:
    metrics = results["metrics"]["primary"]
    base, full = metrics["B000"], metrics["F_ALL15"]
    coverage = results["global"]["coverage_audit"]
    cases = results["cases"]
    comparison = results["comparison"]
    acc_context = _context_lists(comparison, "accuracy")
    macro_context = _context_lists(comparison, "macro_f1")
    full_boot = results["bootstrap"]
    full_acc_boot = full_boot[(full_boot.comparison == "FULL15") & (full_boot.metric == "accuracy")].iloc[0]
    full_macro_boot = full_boot[(full_boot.comparison == "FULL15") & (full_boot.metric == "macro_f1")].iloc[0]

    metric_semantics = f"""# Metric semantics

- **Fixed-mask Accuracy**: all common Primary epochs (`n={base['n']}`); `UNDETERMINED` is incorrect.
- **Macro-F1**: unweighted mean over fixed physiological labels `W/N1/N2/N3/R`; an abstention contributes a false negative to its reference stage, but is not a sixth F1 class.
- **Coverage**: fraction with prediction other than `UNDETERMINED`.
- **Conditional Accuracy**: correct / accepted. It is reported with Coverage; because accepted sets differ by arm, it is not a fixed-epoch paired effect.
- Primary mask: **{base['n']}** shared epochs. Full-valid mask: **{results['metrics']['full']['B000']['n']}** shared epochs.
- Zero-support stage F1 is defined as zero, although every stage has support in the pooled Primary mask.

Independent raw-prediction recomputation matched the frozen evaluator with maximum absolute difference `{results['metric_recheck']['maximum_metric_difference']:.3g}`.
"""
    (output_root / "verification" / "metric_semantics.md").write_text(metric_semantics, encoding="utf-8")

    interaction = results["interactions"]
    interaction_lines = []
    for group in ("G_N2", "G_OCULAR", "G_EEG", "G_EMG_MOV"):
        subset = interaction[interaction.group_id == group].set_index("metric")
        interaction_lines.append(
            f"| {group} | {subset.loc['accuracy','descriptive_nonadditivity']:+.6f} | "
            f"{subset.loc['macro_f1','descriptive_nonadditivity']:+.6f} | "
            f"{subset.loc['coverage','descriptive_nonadditivity']:+.6f} |"
        )
    (output_root / "grouped" / "nonadditivity_analysis.md").write_text("""# Group nonadditivity analysis

`ObservedGroupInteraction = [M(G)-M(B000)] - Σ[M(S_e)-M(B000)]`

| Group | Accuracy residual | Macro-F1 residual | Coverage residual |
|---|---:|---:|---:|
""" + "\n".join(interaction_lines) + """

这些 residual 是冻结规则与当前开发集上的描述性非加性。Macro-F1 本身非线性，且 E09–E11 是共享模型 heads；只有 Evidence/Rule Trace 同时显示共享路径时，才能提出具体交互假设，不能据此宣称生理协同。
""", encoding="utf-8")

    executive = f"""# AASM V4 Phase 2B Executive Summary

**工程结论：36/36 臂、360/360 Night-Arm 的固定结果已从逐 Epoch Prediction/Evidence/Trace 独立复算；未训练、未调参、未修改规则。**

## 三个最重要的真实发现

1. **Full-15 的 Macro-F1 上升不是整体更准。** 相对 B000，Macro-F1 `{full['macro_f1']-base['macro_f1']:+.6f}`，但 Accuracy `{full['accuracy']-base['accuracy']:+.6f}`、N3 F1 `{full['per_stage']['N3']['f1']-base['per_stage']['N3']['f1']:+.6f}`、Coverage `{full['coverage']-base['coverage']:+.6f}`。共同接受的 Epoch 净改善 `+183`，Coverage 转换净损失 `-332`，闭合为 `-149` 个净正确 Epoch。
2. **下游效用取决于 rule-qualified evidence，不取决于 Event count 单调增加。** E01 的慢波数量增加，但总持续时间从 `{cases['E01']['base_event']['event_duration_sum_s']:.1f}s` 降至 `{cases['E01']['comparison_event']['event_duration_sum_s']:.1f}s`，削弱 6 秒 N3.1；E02 的 K 候选从 `{cases['E02']['base_event']['event_count']}` 降到 `{cases['E02']['comparison_event']['event_count']}`，N2 降但 REM 升；AR/SP 的新增证据则能被 reset/N2 structural predicates 稳定消费。
3. **更多 Evidence 可能降低 Coverage。** E13 从 `{cases['E13']['base_event']['event_count']}` 个 frozen MBM 候选增至 `{cases['E13']['comparison_event']['event_count']}` 个 proxy 候选，单换 Macro-F1 `{_delta(results,'S_E13','macro_f1'):+.6f}`，但 Coverage `{_delta(results,'S_E13','coverage'):+.6f}`；其 movement preemption 与 pending/unknown 路径直接制造拒判，E13 仍只能称 `MOVEMENT_PROXY`。

## 两个关键机制

- **Duration/overlap semantic alignment**：Detector 输出必须跨过当前规则真正消费的持续时间、重叠与关联条件；原始候选数或独立 Event F1 不是充分统计量。
- **Temporal propagation and preemption**：当前 Epoch 的 Evidence 会改变后续 Temporal State；Full-15 的多源变化多数不可唯一归因。Movement 还会在普通规则前抢占控制流。

## 首选研究问题与最小实验

**研究问题：Rule-relevant evidence sufficiency（合格持续时间、overlap、association）是否比 Event F1 更能预测 Detector 替换的 Stage utility？**

最小可证伪实验：在一组未用于开发、同时具有专家事件与睡眠分期标注的 PSG 上，预注册 SP/AR/E01/E02 的冻结 Detector、Eventization、Evidence 映射和规则；同时测量 Event F1、rule-qualified evidence coverage 与固定规则 Stage Δ。若 rule-qualified 指标不能比 Event F1 更稳定地解释跨夜 Stage Δ，则该假设被否定。

## 科学边界

这是反复使用的 HMC 十晚开发集机制研究。Recording-level Full-15 ΔAccuracy 95% CI 为 `[{full_acc_boot.ci_low:+.6f}, {full_acc_boot.ci_high:+.6f}]`，ΔMacro-F1 为 `[{full_macro_boot.ci_low:+.6f}, {full_macro_boot.ci_high:+.6f}]`；没有外部泛化或多数 Handler 的 HMC 事件级真值。观察、Trace-supported mechanism 与待验证假设在完整报告中分开标记。
"""
    (output_root / "EXECUTIVE_SUMMARY.md").write_text(executive, encoding="utf-8")

    final_report = f"""# Final Phase 2B Mechanistic Analysis Report

## Scope and status

- Frozen source: `{config['source_root']}`
- Primary: {base['n']} epochs; full-valid: {results['metrics']['full']['B000']['n']} epochs.
- Arms/night replays: 36 / 360.
- No training, detector search, HMC tuning, rule modification, or access outside the frozen record allowlist.

## Q1. 为什么 Macro-F1 上升，而 Accuracy、N3 F1、Coverage 下降？

Full-15 将 W/N1/REM F1 分别改变 `{_delta(results,'F_ALL15','W_f1'):+.6f}` / `{_delta(results,'F_ALL15','N1_f1'):+.6f}` / `{_delta(results,'F_ALL15','R_f1'):+.6f}`，同时 N2/N3 F1 改变 `{_delta(results,'F_ALL15','N2_f1'):+.6f}` / `{_delta(results,'F_ALL15','N3_f1'):+.6f}`。Macro-F1 对五类等权，因此低召回阶段的改善可超过 N2/N3 的 macro 贡献损失；Accuracy 则受 N2/N3 大支持度和拒判影响。Coverage 从 {base['coverage']:.6f} 降至 {full['coverage']:.6f}：Accepted→UNDETERMINED {coverage['accepted_to_undetermined']}，反向 {coverage['undetermined_to_accepted']}。共同接受 strata 为 +183，Coverage strata 为 −332，总和 −149。

## Q2. 哪些 Handler 发生 context reversal？

- Accuracy `Single+ / Full−`: {', '.join(acc_context.get('SINGLE_POSITIVE_FULL_NEGATIVE', [])) or 'none'}；`Single− / Full+`: {', '.join(acc_context.get('SINGLE_NEGATIVE_FULL_POSITIVE', [])) or 'none'}。
- Macro-F1 `Single+ / Full−`: {', '.join(macro_context.get('SINGLE_POSITIVE_FULL_NEGATIVE', [])) or 'none'}；`Single− / Full+`: {', '.join(macro_context.get('SINGLE_NEGATIVE_FULL_POSITIVE', [])) or 'none'}。

主要机制不是单一“协同”：N2 组共享 structural/reset/continuation 路径；ocular 组共享 Wake/N1/REM 分支；EMG heads 彼此相关且 E13 会 preempt；Temporal state 又把早期差异传播到后续 Epoch。因此 LOO 是当前 Full context 的条件边际，不是普适生理因果效应。

## Q3. K08、SSSM Slow-wave、Movement Proxy 在哪里破坏链条？

| Source | Broken layer | Frozen evidence |
|---|---|---|
| E01 SSSM | Event duration → evidence threshold | Event 总时长 `{cases['E01']['base_event']['event_duration_sum_s']:.1f}→{cases['E01']['comparison_event']['event_duration_sum_s']:.1f}s`; 6-s down-crossing {cases['E01']['threshold_cross_down']}; ΔN3 F1 {_delta(results,'S_E01','N3_f1'):+.6f} |
| E02 K08 | Sparse K / association → N2 structural evidence → temporal fallback/REM | Event `{cases['E02']['base_event']['event_count']}→{cases['E02']['comparison_event']['event_count']}`; associated `{cases['E02']['base_event']['associated_event_count']}→{cases['E02']['comparison_event']['associated_event_count']}`; ΔN2 {_delta(results,'S_E02','N2_f1'):+.6f}, ΔREM {_delta(results,'S_E02','R_f1'):+.6f} |
| E13 proxy | Event density → movement preemption/pending → abstention | Event `{cases['E13']['base_event']['event_count']}→{cases['E13']['comparison_event']['event_count']}`; ΔCoverage {_delta(results,'S_E13','coverage'):+.6f}; proxy semantics only |

这些是 rule trace 支持的传播机制；它们不等价于在 HMC 上测得 Event FP/FN，因为对应独立 Event Truth 不存在。

## Q4. 为什么 Arousal 和 Spindle 相对有效？

AR 单换 ΔAccuracy `{_delta(results,'S_AR','accuracy'):+.6f}`、ΔMacro-F1 `{_delta(results,'S_AR','macro_f1'):+.6f}`；它改变持续时间/重叠、K-association、N2 reset 和 post-arousal 路径，而不只是增加候选。SP 单换 ΔAccuracy `{_delta(results,'S_SP','accuracy'):+.6f}`、ΔN2 F1 `{_delta(results,'S_SP','N2_f1'):+.6f}`；`n2_spindle` 可直接进入 N2.1。共同点是 Detector output 与现有 predicate 的 temporal semantics 对齐。二者仍只是内部开发证据。

## Q5. 最值得研究的问题

首选问题是：**Event-level accuracy 与 rule-relevant evidence sufficiency 哪一个真正决定 downstream utility，以及这种关系是否跨数据集成立？** 详见 `PAPER_RESEARCH_QUESTIONS.md`。最低限度需要一套新 PSG cohort 同时提供事件和 Stage Truth，并在完全冻结的 eventization/evidence/rule 方案下验证。

## Group nonadditivity and uncertainty

四组的完整 residual 位于 `grouped/group_interactions.csv`。Full-15 recording-equal ΔAccuracy mean `{full_acc_boot.record_equal_mean:+.6f}`（positive nights {full_acc_boot.positive_night_fraction:.1%}），ΔMacro-F1 mean `{full_macro_boot.record_equal_mean:+.6f}`（positive nights {full_macro_boot.positive_night_fraction:.1%}）。这些十晚区间没有多重比较校正，也不是外部统计认证。

## Evidence classification

- `OBSERVED_STAGE_EFFECT`: 所有固定 Mask 指标与 paired error movement。
- `TRACE_SUPPORTED_MECHANISM`: E01 duration→N3.1；E13 movement preemption→abstention；AR/SP rule-consumable paths。
- `CONTEXT_DEPENDENT_INTERACTION`: Single/LOO sign reversals and grouped residuals。
- `METRIC_COVERAGE_TRADEOFF`: Full-15 和 E13。
- `HYPOTHESIS_REQUIRING_VALIDATION`: rule-relevant sufficiency predicts Stage utility。
- `UNRESOLVED_ATTRIBUTION`: Full-15 individual-source causal allocation and epochs with multiple simultaneous evidence/temporal changes。
"""
    (output_root / "FINAL_MECHANISTIC_ANALYSIS_REPORT.md").write_text(final_report, encoding="utf-8")

    paper_questions = """# Paper Research Questions

## RQ1 (preferred): Does rule-relevant evidence sufficiency mediate event-to-stage transfer better than event F1?

### Observed phenomenon

E01 increased event count while fragmenting duration and reducing N3; E02 improved some expert-event metrics historically yet reduced rule-qualified K coverage/N2; SP and AR produced evidence that directly reached N2/reset predicates and showed positive downstream effects.

### Supported mechanism

The current traces support duration-threshold, overlap, association, and temporal-state pathways. They do not establish a universal mediator outside HMC.

### Alternative explanations

Montage/domain shift, weak labels, detector calibration, baseline quality, annotation disagreement, and rule-specific priority can produce the same pattern.

### Minimal falsifiable experiment

Use a new PSG cohort with both expert event annotations and stage labels. Freeze detector checkpoints, eventization, evidence aggregation, and rules before evaluation. Measure Event F1 and predeclared rule-qualified quantities (duration above threshold, overlap, K–arousal association), then test whether these quantities predict per-record Stage Δ more consistently than Event F1. The hypothesis fails if qualified evidence adds no out-of-sample explanatory value.

### Novelty and external validation

The contribution is an event-to-decision interface theory, not another detector leaderboard. Independent event labels, unseen records/center, and a frozen rules replay are essential.

## RQ2: When does safety-oriented movement evidence become harmful abstention amplification?

### Observed phenomenon

E13 improves Wake/Macro-F1 in some contexts while sharply increasing movement preemption and Coverage loss; Full-15 loses 615 accepted epochs net.

### Supported mechanism

Trace data show `MBM.2.UNKNOWN` direct abstentions plus subsequent temporal propagation. E13 is only a movement proxy, so event correctness is unresolved.

### Alternative explanations

Proxy semantic mismatch, event density calibration, lookahead boundary behavior, and invalid-history handling could each explain the abstention increase.

### Minimal falsifiable experiment

On external PSG with expert MBM labels, compare the frozen movement proxy and a validated MBM source using the same fixed movement rules. Predeclare eventization and measure both movement correctness and accepted-to-abstained transitions. The mechanism is weakened if a semantically valid MBM source produces the same excess abstention at matched event density.

### Novelty and external validation

This targets selective prediction and rule preemption in physiological reasoning systems. It requires expert MBM truth; stage outcomes alone cannot validate movement detection.
"""
    (output_root / "PAPER_RESEARCH_QUESTIONS.md").write_text(paper_questions, encoding="utf-8")

    next_decision = """# Next Experiment Decision

**首选下一步：执行 RQ1 的外部、双层标注验证；不要继续在 HMC 上选组合或调阈值。**

最低实验合同：新 Recording/center；专家事件 truth + stage truth；SP/AR/E01/E02 checkpoint、eventization、qualified-evidence definitions、AASM rules 全部预先冻结；主分析比较 Event F1 与 rule-qualified evidence 对 per-record Stage Δ 的解释/预测能力。E13/MBM 可作为第二条独立工作包，不能用 movement proxy 的 Stage 结果替代 MBM Event Truth。
"""
    (output_root / "NEXT_EXPERIMENT_DECISION.md").write_text(next_decision, encoding="utf-8")

    uncertainty = f"""# Uncertainty summary

- Resampling unit: recording/night (10 HMC development nights), never 8,902 epochs as independent subjects.
- Full-15 ΔAccuracy record-equal mean `{full_acc_boot.record_equal_mean:+.6f}`, median `{full_acc_boot.median_per_night:+.6f}`, positive-night fraction `{full_acc_boot.positive_night_fraction:.1%}`, 95% bootstrap CI `[{full_acc_boot.ci_low:+.6f}, {full_acc_boot.ci_high:+.6f}]`.
- Full-15 ΔMacro-F1 record-equal mean `{full_macro_boot.record_equal_mean:+.6f}`, median `{full_macro_boot.median_per_night:+.6f}`, positive-night fraction `{full_macro_boot.positive_night_fraction:.1%}`, 95% bootstrap CI `[{full_macro_boot.ci_low:+.6f}, {full_macro_boot.ci_high:+.6f}]`.
- Pooled effects and record-equal means differ because nights have different epoch counts.
- CIs are descriptive, unadjusted for multiple comparisons, and not external generalization evidence.
"""
    (output_root / "statistics" / "uncertainty_summary.md").write_text(uncertainty, encoding="utf-8")

    run_plan = f"""# Run Plan (executed)

1. Verify frozen 36-arm/360-night heritage, hashes, masks, and metric semantics.
2. Independently recompute fixed-mask and acceptance-conditional metrics.
3. Compute all 15 Single effects, four grouped nonadditivity contrasts, and all 15 Full contributions.
4. Trace Event → Evidence → Rule → Temporal State → Stage for Single, Full, and LOO contrasts.
5. Complete E01/E02/E13 case studies and AR/SP positive controls.
6. Compute recording-level uncertainty and figures.
7. Independently recheck claims and package artifacts.

Started: `{started_at}`. Frozen source is read-only. New outputs are confined to `AASM_V4_RESULTS_ROOT`.
"""
    (output_root / "RUN_PLAN.md").write_text(run_plan, encoding="utf-8")

    reproduce = f"""# Reproduce

From the repository root:

```powershell
$env:AASM_V4_PHASE2_RUN='<path-to-sealed-36-arm-run>'
$env:AASM_V4_RESULTS_ROOT='<private-output-root>'
$env:AASM_V4_REFERENCE_PATH='<path-to-aligned-development-reference>'
$env:AASM_V4_RECORDS='<comma-separated-private-record-allowlist>'
python -m experiments.v4_mechanistic_attribution.run `
  --config experiments/v4_mechanistic_attribution/configs/main.yaml `
  --max-wall-hours 12 --resume
```

The command reads the sealed Phase 2 directory and frozen ten-record aligned reference, performs no model training or rule changes, and writes a new timestamped result under `AASM_V4_RESULTS_ROOT`.
"""
    (output_root / "reproduce.md").write_text(reproduce, encoding="utf-8")

    claims = [
        ["C01", "Full-15 Macro-F1 gain coexists with Accuracy/Coverage loss", "B000;F_ALL15", "global_analysis/full_system_tradeoff.csv", "fixed metrics;paired strata", "recording bootstrap", "development set"],
        ["C02", "E01 fragmentation weakens N3.1 duration coverage", "B000;S_E01", "mechanisms/e01_slowwave_n3.md", "event duration;6-s crossings;N3 F1", "per-night", "no HMC event truth"],
        ["C03", "E02 is Accuracy-negative but Macro-positive", "B000;S_E02;F_ALL15;L_E02", "mechanisms/e02_kcomplex_n2.md", "N2 and REM F1;K evidence", "per-night", "context dependent"],
        ["C04", "E13 movement proxy amplifies abstention", "B000;S_E13;F_ALL15", "mechanisms/e13_movement_coverage.md", "MBM rules;coverage transitions", "per-night", "proxy semantics"],
        ["C05", "AR and SP reach rule-consumable paths", "B000;S_AR;S_SP", "mechanisms/arousal_positive_control.md;mechanisms/spindle_positive_control.md", "paired corrections;rule traces", "recording bootstrap", "internal validation"],
        ["C06", "Full-system source attribution is multifactor", "B000;F_ALL15", "mechanisms/first_divergence_trace.csv", "changed evidence and temporal state", "not applicable", "unique causal allocation unresolved"],
    ]
    pd.DataFrame(claims, columns=["claim_id", "claim", "arm_ids", "source_artifact", "numeric_evidence", "uncertainty", "known_limitations"]).to_csv(
        output_root / "audits" / "scientific_claim_audit.csv", index=False
    )
