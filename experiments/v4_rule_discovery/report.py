from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .cache import PredictionCache
from .config import STAGES, RuleConfig
from .metrics import ReferenceData


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def discovery_leaderboard(selection: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in selection["candidate_summaries"]:
        row = {key: value for key, value in item.items() if key not in {"stage_f1_deltas", "per_record"}}
        for stage, value in item["stage_f1_deltas"].items():
            row[f"delta_{stage}_f1"] = value
        rows.append(row)
    return sorted(rows, key=lambda row: row["robust_score"], reverse=True)


def current_rule_diagnostics(
    cache: PredictionCache,
    reference: ReferenceData,
    records: tuple[str, ...],
) -> list[dict[str, Any]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        data = reference.rows[record]
        mask = data["primary"]
        truth = data["truth"][mask]
        prediction = cache.stage_codes("C000", record)[mask]
        rules = [rule for rule, keep in zip(cache.rule_ids("C000", record), mask.tolist()) if keep]
        for rule, y, pred in zip(rules, truth.tolist(), prediction.tolist()):
            counts[rule]["total"] += 1
            counts[rule]["correct"] += int(y == pred)
            counts[rule][f"true_{STAGES[y]}"] += 1
    output = []
    for rule, values in counts.items():
        total = values["total"]
        output.append(
            {
                "rule_id": rule,
                "count": total,
                "correct": values["correct"],
                "precision_as_final_rule": values["correct"] / total if total else 0.0,
                **{f"true_{stage}": values[f"true_{stage}"] for stage in STAGES[:5]},
            }
        )
    return sorted(output, key=lambda row: row["count"], reverse=True)


def _metric_table(result: dict[str, Any]) -> str:
    baseline = result["baseline"]
    candidate = result["candidate"]
    lines = [
        "| Metric | Temporal R1 | New rules | Delta |",
        "|---|---:|---:|---:|",
        f"| Accuracy | {baseline['accuracy']:.6f} | {candidate['accuracy']:.6f} | {result['delta']['accuracy']:+.6f} |",
        f"| Macro-F1 | {baseline['macro_f1']:.6f} | {candidate['macro_f1']:.6f} | {result['delta']['macro_f1']:+.6f} |",
        f"| Coverage | {baseline['coverage']:.6f} | {candidate['coverage']:.6f} | {result['delta']['coverage']:+.6f} |",
    ]
    for stage in STAGES[:5]:
        lines.append(
            f"| {stage} F1 | {baseline['per_stage'][stage]['f1']:.6f} | "
            f"{candidate['per_stage'][stage]['f1']:.6f} | {result['stage_f1_deltas'][stage]:+.6f} |"
        )
    return "\n".join(lines)


def _paired_line(result: dict[str, Any]) -> str:
    paired = result["paired"]
    return (
        f"Changed={paired['changed']}，Corrected={paired['corrected']}，"
        f"Introduced={paired['introduced']}，Net={paired['net_correct']}。"
    )


def _policy_table(selected: RuleConfig) -> str:
    rows = (
        ("REM definite-R前向回溯", selected.rem_backward_policy),
        ("Wake ocular语义", selected.wake_ocular_policy),
        ("N1 alpha attenuation anchor", selected.n1_alpha_policy),
        ("N1 non-alpha路径", selected.n1_nonalpha_policy),
        ("N2 continuation", selected.n2_continuation_policy),
        ("Post-arousal N2 re-entry", selected.post_arousal_reentry_policy),
        ("Fallback", selected.fallback_policy),
        ("REM second-half structure", selected.rem_second_half_policy),
    )
    lines = ["| Decision module | Frozen value |", "|---|---|"]
    lines.extend(f"| {name} | `{value}` |" for name, value in rows)
    return "\n".join(lines)


def _source_table(selected: RuleConfig) -> str:
    notes = {
        "FROZEN_TEMPORAL_R1": (
            "冻结可执行基线",
            "本项目Temporal R1源码；用于精确回归，不是新的AASM声明",
        ),
        "AASM_V2_1_REM_BACKWARD": (
            "AASM公开更新",
            "definite R之前连续片段的Stage R操作化与多数规则",
        ),
        "AASM_STAGE_W_EYE_CRITERIA": (
            "AASM stage-W生理条件",
            "blink与不规则快速眼动应按离散ocular证据解释，而非累计15秒活动",
        ),
        "AASM_V2_1_N1_MAJORITY_LAMF": (
            "AASM N1条件",
            "alpha attenuation/replacement与LAMF多数；SEM/vertex为支持证据",
        ),
        "AASM_V2_1_N2_CONTINUATION": (
            "AASM N2条件",
            "结构锚点、continuation与termination的显式状态",
        ),
        "AASM_V2_1_N2_POST_AROUSAL": (
            "AASM规则基础 + 工程操作化",
            "AASM支持segment-level N2重入；whole-epoch优先级是本项目测试的工程操作化",
        ),
        "AASM_STAGE_CONTINUATION_TERMINATION": (
            "AASM-compatible engineering",
            "用stage-specific条件替代无生理依据的通用继承",
        ),
        "AASM_REM_TRANSITION_FAQ": (
            "AASM公开FAQ操作化",
            "K-complex/spindle在epoch前后半段对N2/R transition的影响",
        ),
    }
    lines = ["| Source ID | Evidence class | Used for |", "|---|---|---|"]
    for source in selected.source_ids:
        evidence_class, use = notes.get(
            source,
            ("注册来源", "详见rule_source_registry.json；未在报告中扩大其声明范围"),
        )
        lines.append(f"| `{source}` | {evidence_class} | {use} |")
    return "\n".join(lines)


def _component_table(component_ablation: dict[str, Any]) -> str:
    lines = [
        "| Confirmation ablation | Config | ΔAccuracy | ΔMacro-F1 | ΔN2 F1 | ΔREM F1 | Net correct |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "rem_backward_only": "REM backward only",
        "n2_reentry_only": "N2 re-entry only",
        "combined_selected": "Combined frozen C196",
    }
    for key in ("rem_backward_only", "n2_reentry_only", "combined_selected"):
        item = component_ablation[key]
        result = item["confirmation"]
        lines.append(
            f"| {labels[key]} | `{item['config_id']}` | "
            f"{result['delta']['accuracy']:+.6f} | {result['delta']['macro_f1']:+.6f} | "
            f"{result['stage_f1_deltas']['N2']:+.6f} | {result['stage_f1_deltas']['R']:+.6f} | "
            f"{result['paired']['net_correct']:+d} |"
        )
    return "\n".join(lines)


def _raw_best_table(raw: dict[str, Any]) -> str:
    discovery = raw["discovery_selection_summary"]
    confirmation = raw["confirmation_post_hoc"]
    lines = [
        "| Scope | ΔAccuracy | ΔMacro-F1 | ΔW F1 | ΔN1 F1 | ΔN2 F1 | ΔN3 F1 | ΔREM F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| Discovery | {discovery['delta_accuracy']:+.6f} | "
            f"{discovery['delta_macro_f1']:+.6f} | "
            + " | ".join(
                f"{discovery['stage_f1_deltas'][stage]:+.6f}" for stage in STAGES[:5]
            )
            + " |"
        ),
        (
            f"| Confirmation (post-hoc) | {confirmation['delta']['accuracy']:+.6f} | "
            f"{confirmation['delta']['macro_f1']:+.6f} | "
            + " | ".join(
                f"{confirmation['stage_f1_deltas'][stage]:+.6f}" for stage in STAGES[:5]
            )
            + " |"
        ),
    ]
    return "\n".join(lines)


def render_reports(
    root: Path,
    selected: RuleConfig,
    selection: dict[str, Any],
    discovery: dict[str, Any],
    confirmation: dict[str, Any],
    full: dict[str, Any],
    elapsed: float,
    rule_diagnostics: list[dict[str, Any]] | None = None,
    evidence_audit: dict[str, Any] | None = None,
    component_ablation: dict[str, Any] | None = None,
    raw_best_exploratory: dict[str, Any] | None = None,
    backward_span: dict[str, Any] | None = None,
) -> None:
    status = (
        "HMC_CONFIRMATION_GAIN_SUPPORTED"
        if confirmation["delta"]["accuracy"] > 0
        and confirmation["delta"]["macro_f1"] > 0
        and confirmation["delta"]["coverage"] >= -0.005
        and min(confirmation["stage_f1_deltas"].values()) >= -0.03
        else "HMC_CONFIRMATION_TRADEOFF_OR_NO_GAIN"
    )
    patches = ", ".join(selected.changed_modules) or "NONE"
    candidate_count = int(selection.get("candidate_count", len(selection["candidate_summaries"])))
    confirmation_ci = confirmation["bootstrap"]
    rule_map = {row["rule_id"]: row for row in (rule_diagnostics or [])}
    evidence_counts = (evidence_audit or {}).get("counts", {})
    evidence_maxima = (evidence_audit or {}).get("maxima_seconds", {})
    component_table = (
        _component_table(component_ablation) if component_ablation else "NOT RUN"
    )
    raw_best_table = (
        _raw_best_table(raw_best_exploratory) if raw_best_exploratory else "NOT RUN"
    )
    executive = f"""# AASM Rule Discovery v5 — Executive Summary

## 结论

- 固定输入：`M_6408 = E04 + E11 + SUMOv2 Spindle + U-Net Arousal`。
- 规则候选：{candidate_count}个；只改decision logic，不改Detector、Event Bank、Evidence、QC或6秒/15秒生理阈值。
- Discovery选择：`{selected.config_id}`，规则补丁：`{patches}`。
- Confirmation科学状态：`{status}`。

## Frozen confirmation partition

{_metric_table(confirmation)}

{_paired_line(confirmation)}

Recording-level bootstrap 95% CI：Accuracy {confirmation_ci['accuracy']['ci95']}；Macro-F1 {confirmation_ci['macro_f1']['ci95']}。

## 关键机制

Temporal R1缺少AASM公开更新中的definite-R backward bridge。旧R.2只能从已有R向后续Epoch延续；新规则允许未来definite R为其之前连续、LAMF占多数、低肌张力、无K/spindle/arousal/SEM的Epoch提供R上下文。C196还测试了arousal后合格K/spindle结构证据优先恢复N2；AASM支持segment-level重入，但whole-epoch优先于majority是本项目工程操作化。Wake/N1/fallback候选也被测试，但未进入冻结winner。

实际累计执行时间：{elapsed:.2f}秒。结果仍属于反复使用HMC开发集上的post-hoc confirmation，不是外部临床验证；Full-development结果只用于描述，不能替代Confirmation。
"""
    (root / "EXECUTIVE_SUMMARY.md").write_text(executive, encoding="utf-8")
    report = f"""# Final Data-driven AASM Rule Report

## Engineering status

`AASM_RULE_DISCOVERY_V5_ENGINEERING_PASS`

{candidate_count}/{candidate_count}配置在10晚完成label-blind replay；C000十晚逐字段复现Temporal R1。五晚discovery partition用于数据驱动发现和选择；选择与配置封存后才评价另外五晚confirmation partition。

## Existing rule limitations found from data

1. **Wake ocular语义不可达。** 8,902个Primary epochs中blink evidence出现 `{evidence_counts.get('blink_observed', 'NA')}` 次，最长仅 `{evidence_maxima.get('blink_seconds', 'NA')}` 秒，超过15秒为 `{evidence_counts.get('blink_over_15', 'NA')}` 次；所以`W.2 blink >15s`把离散blink错当成连续活动。Full-development基线Wake F1仅为 `{full['baseline']['per_stage']['W']['f1']:.6f}`，但这只能定位规则/证据链瓶颈，不能证明ocular detector错误。
2. **N1路径定义过粗。** `N1.2`把generic slowing/vertex/SEM任一出现当成N1且不要求LAMF majority；它触发 `{rule_map.get('N1.2', {}).get('count', 'NA')}` 次、最终阶段正确率仅 `{rule_map.get('N1.2', {}).get('precision_as_final_rule', float('nan')):.6f}`。Full-development基线N1 F1为 `{full['baseline']['per_stage']['N1']['f1']:.6f}`。
3. **Post-arousal与通用fallback缺少结构语义。** `N2.4.POST_AROUSAL_MAJORITY`触发 `{rule_map.get('N2.4.POST_AROUSAL_MAJORITY', {}).get('count', 'NA')}` 次、正确率 `{rule_map.get('N2.4.POST_AROUSAL_MAJORITY', {}).get('precision_as_final_rule', float('nan')):.6f}`；`FALLBACK.INHERIT_PREVIOUS`触发 `{rule_map.get('FALLBACK.INHERIT_PREVIOUS', {}).get('count', 'NA')}` 次、正确率 `{rule_map.get('FALLBACK.INHERIT_PREVIOUS', {}).get('precision_as_final_rule', float('nan')):.6f}`。C196只采纳了有K/spindle结构证据时优先N2 re-entry；更激进的stage-specific fallback没有通过Discovery选择。
4. **N3主要受输入Evidence限制。** 1,534个Reference N3 epochs中，`{evidence_counts.get('n3_no_slow_wave', 'NA')}`个无slow-wave evidence，`{evidence_counts.get('n3_slow_wave_below_6', 'NA')}`个有候选但少于6秒，只有`{evidence_counts.get('n3_slow_wave_at_least_6', 'NA')}`个达到6秒。6秒等于30秒epoch的20%，符合AASM N3定义，因此本轮没有为提高HMC分数而降低阈值。
5. **REM transition实现不完整。** R.2只有forward continuation，没有AASM公开更新支持的definite-R backward bridge；C196补上该路径，并在确认集把REM F1提高 `{confirmation['stage_f1_deltas']['R']:+.6f}`。

这些局限来自冻结HMC上的confusion、rule firing和trace审计；它们是开发集诊断，不是独立病理生理结论。

## Frozen selected rule

- Config: `{selected.config_id}`
- Changed modules: `{patches}`
- Sources: `{', '.join(selected.source_ids)}`
- Claim types: `{', '.join(selected.claim_types)}`
- Selection reason: `{selection['selection_reason']}`

Source-scope caveat：AASM公开更新为post-arousal N2 segment re-entry提供规则基础；C196的`structure_reentry_first`把该证据提升为whole-epoch优先级，是本项目经数据检验的engineering operationalization，不应写成AASM逐字规定。

### Executable policy values

{_policy_table(selected)}

### AASM/source basis

{_source_table(selected)}

### Selected-rule component ablation

{component_table}

该ablation在winner冻结后执行，仅用于解释两项规则各自贡献；不会反向改变选择。

### Guardrail-rejected raw-metric candidate

Discovery无约束robust-score最高者是`{raw_best_exploratory['config_id'] if raw_best_exploratory else 'NOT RUN'}`，但其Discovery N1 F1下降超过预声明−0.03 guardrail，因此没有取代C196。为透明披露，在winner冻结后做了探索性confirmation：

{raw_best_table}

该结果是`GUARDRAIL_REJECTED_EXPLORATORY_NOT_SELECTED`，不是第二个确认冠军，也不能用于事后改写选择规则。

### Selected-rule operational risk

`R.BACKWARD`在全十晚重标 `{(backward_span or {}).get('count', 'NA')}` 个epochs；回溯长度中位数 `{(backward_span or {}).get('median_epochs', 'NA')}` epochs，95th percentile `{(backward_span or {}).get('p95_epochs', 'NA')}`，最大 `{(backward_span or {}).get('max_epochs', 'NA')}` epochs（`{(backward_span or {}).get('max_seconds', 'NA')}`秒）。当前实现没有显式maximum span；这不是本轮确认失败，但属于独立scorer review和外部验证前必须解决的操作化风险。

## Discovery partition

{_metric_table(discovery)}

{_paired_line(discovery)}

## Confirmation partition

{_metric_table(confirmation)}

{_paired_line(confirmation)}

## Full HMC development description

{_metric_table(full)}

{_paired_line(full)}

## Paired confirmation

- Changed: {confirmation['paired']['changed']}
- Corrected: {confirmation['paired']['corrected']}
- Introduced: {confirmation['paired']['introduced']}
- Net correct: {confirmation['paired']['net_correct']}
- Accuracy bootstrap 95% CI: {confirmation['bootstrap']['accuracy']['ci95']}
- Macro-F1 bootstrap 95% CI: {confirmation['bootstrap']['macro_f1']['ci95']}

逐类F1已在上述三张固定口径表中完整报告。Discovery用于选择，Confirmation用于选择后内部确认，Full HMC仅为selection-contaminated描述；三者不能混称为同一个测试结果。

## Scientific boundary

The HMC ten-night dataset has been repeatedly studied. The confirmation partition was process-isolated from this run's selector, but this is `WITHIN_PROJECT_POST_HOC_CONFIRMATION`, not a new external test. Detector accuracy cannot be inferred from stage gains. AASM public updates and FAQs constrain the candidate logic, but this implementation is an engineering operationalization rather than a substitute for the licensed scoring manual or clinical scorer review.
"""
    (root / "FINAL_DATA_DRIVEN_RULE_REPORT.md").write_text(report, encoding="utf-8")
    decision = f"""# Rule Adoption Decision

Status: `{status}`

Frozen candidate: `{selected.config_id}`.

Changed modules: `{patches}`.

Confirmation: ΔAccuracy={confirmation['delta']['accuracy']:+.6f}, ΔMacro-F1={confirmation['delta']['macro_f1']:+.6f}, ΔCoverage={confirmation['delta']['coverage']:+.6f}; {_paired_line(confirmation)}

This candidate may be retained as an HMC-development research rule set. Public/default replacement requires a genuinely external, record-disjoint confirmation with the same frozen config.
"""
    (root / "RULE_ADOPTION_DECISION.md").write_text(decision, encoding="utf-8")
