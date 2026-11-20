from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core import (
    ALL_PREDICTIONS,
    EVENT_TYPE,
    EVIDENCE_KEYS,
    GROUPS,
    HANDLERS,
    RULE_PREFIXES,
    STAGES,
    arm_prediction_frame,
    bootstrap_summary,
    changed_evidence_keys,
    changed_temporal_keys,
    comparison_counts,
    confusion_rows,
    event_stats,
    evidence_signature,
    evidence_stats,
    first_rule_difference,
    flatten_metric,
    load_arm_payload,
    metric_for_arm,
    rule_firing_counts,
    sha256,
    sign_class,
    temporal_before_signature,
    write_json,
)


METRIC_NAMES = ("accuracy", "macro_f1", "coverage")
STAGE_METRICS = tuple(f"{stage}_f1" for stage in STAGES)
ALL_EFFECT_METRICS = METRIC_NAMES + STAGE_METRICS


def _flat_value(metric: dict[str, Any], name: str) -> float:
    if name.endswith("_f1") and name[:-3] in STAGES:
        return float(metric["per_stage"][name[:-3]]["f1"])
    return float(metric[name])


def _ensure_directories(output_root: Path) -> None:
    for directory in (
        "verification", "global_analysis", "single_swap", "grouped",
        "leave_one_out", "mechanisms", "statistics", "figures", "audits",
    ):
        (output_root / directory).mkdir(parents=True, exist_ok=True)


def _metric_recheck(
    source_root: Path,
    output_root: Path,
    arm_ids: list[str],
    frames: dict[str, pd.DataFrame],
    metrics: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    frozen = pd.read_csv(source_root / "evaluation" / "all_36_arm_metrics.csv")
    max_difference = 0.0
    recheck_rows: list[dict[str, Any]] = []
    confusion: list[dict[str, Any]] = []
    for scope, mask_column in (("primary", "in_primary"), ("full", "in_full_record")):
        for arm_id in arm_ids:
            metric = metrics[scope][arm_id]
            target = frozen[(frozen.scope == scope) & (frozen.arm_id == arm_id)].iloc[0]
            differences: dict[str, float] = {}
            for name in ("accuracy", "macro_f1", "kappa", "coverage", "conditional_accuracy"):
                differences[name] = abs(float(metric[name]) - float(target[name]))
            for stage in STAGES:
                differences[f"{stage}_f1"] = abs(
                    float(metric["per_stage"][stage]["f1"]) - float(target[f"{stage}_f1"])
                )
            maximum = max(differences.values())
            max_difference = max(max_difference, maximum)
            recheck_rows.append({
                "scope": scope, "arm_id": arm_id,
                "n": metric["n"], "maximum_metric_difference": maximum,
                "pass": maximum < 1e-12,
            })
            if arm_id in {"B000", "F_ALL15"}:
                confusion.extend(confusion_rows(arm_id, scope, frames[arm_id], mask_column))
    pd.DataFrame(confusion).to_csv(output_root / "verification" / "confusion_matrix_recheck.csv", index=False)
    result = {
        "status": "PASS" if max_difference < 1e-12 else "FAIL",
        "arm_count": len(arm_ids), "scope_count": 2,
        "maximum_metric_difference": max_difference,
        "rows": recheck_rows,
        "source": str(source_root / "evaluation" / "all_36_arm_metrics.csv"),
    }
    write_json(output_root / "verification" / "frozen_metric_recheck.json", result)
    return result


def _global_tradeoff(
    output_root: Path,
    frames: dict[str, pd.DataFrame],
    metrics: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    base, full = metrics["primary"]["B000"], metrics["primary"]["F_ALL15"]
    rows: list[dict[str, Any]] = []
    for name in ("accuracy", "macro_f1", "coverage", "conditional_accuracy") + STAGE_METRICS:
        base_value = _flat_value(base, name)
        full_value = _flat_value(full, name)
        rows.append({
            "metric": name, "B000_fixed_mask": base_value,
            "F_ALL15_fixed_mask": full_value, "delta_fixed": full_value - base_value,
            "note": "fixed common primary mask" if name != "conditional_accuracy" else "accepted denominators differ",
        })
    pd.DataFrame(rows).to_csv(output_root / "global_analysis" / "full_system_tradeoff.csv", index=False)

    left = frames["B000"].query("in_primary == True").copy()
    right = frames["F_ALL15"].query("in_primary == True").copy()
    merged = left.merge(
        right[["record_id", "epoch_index", "prediction", "rule_id", "abstention_reason", "decision_source"]],
        on=["record_id", "epoch_index"], suffixes=("_base", "_full"), validate="one_to_one",
    )
    merged["base_accepted"] = merged.prediction_base != "UNDETERMINED"
    merged["full_accepted"] = merged.prediction_full != "UNDETERMINED"
    merged["base_correct"] = merged.true_stage == merged.prediction_base
    merged["full_correct"] = merged.true_stage == merged.prediction_full
    merged["coverage_transition"] = np.select(
        [
            merged.base_accepted & merged.full_accepted,
            merged.base_accepted & ~merged.full_accepted,
            ~merged.base_accepted & merged.full_accepted,
        ],
        ["ACCEPTED_TO_ACCEPTED", "ACCEPTED_TO_UNDETERMINED", "UNDETERMINED_TO_ACCEPTED"],
        default="UNDETERMINED_TO_UNDETERMINED",
    )
    merged["corrected"] = (~merged.base_correct & merged.full_correct).astype(int)
    merged["introduced"] = (merged.base_correct & ~merged.full_correct).astype(int)
    merged["net_correct"] = merged.corrected - merged.introduced

    aggregate = (
        merged.groupby(["coverage_transition", "true_stage"], dropna=False)
        .agg(n=("epoch_index", "size"), corrected=("corrected", "sum"), introduced=("introduced", "sum"), net_correct=("net_correct", "sum"))
        .reset_index()
    )
    aggregate.to_csv(output_root / "global_analysis" / "corrected_introduced_by_stage.csv", index=False)

    transitions = (
        merged.groupby(["true_stage", "prediction_base", "prediction_full"], dropna=False)
        .agg(n=("epoch_index", "size"), corrected=("corrected", "sum"), introduced=("introduced", "sum"))
        .reset_index()
    )
    transitions.to_csv(output_root / "global_analysis" / "stage_error_transitions.csv", index=False)

    coverage_changed = merged[merged.coverage_transition != "ACCEPTED_TO_ACCEPTED"].copy()
    coverage_changed["direct_movement_abstention"] = (
        coverage_changed.rule_id_full.astype(str).str.startswith("MBM.2")
        | coverage_changed.abstention_reason_full.astype(str).str.contains("future nonmovement", case=False, na=False)
    )
    coverage_changed.to_csv(output_root / "global_analysis" / "coverage_loss_decomposition.csv", index=False)
    movement_direct = int(
        ((merged.coverage_transition == "ACCEPTED_TO_UNDETERMINED") & merged.rule_id_full.astype(str).str.startswith("MBM.2")).sum()
    )
    loss_by_stage = (
        merged[merged.coverage_transition == "ACCEPTED_TO_UNDETERMINED"]
        .groupby("true_stage").size().to_dict()
    )
    coverage_audit = {
        "primary_epochs": int(len(merged)),
        "accepted_to_accepted": int((merged.coverage_transition == "ACCEPTED_TO_ACCEPTED").sum()),
        "accepted_to_undetermined": int((merged.coverage_transition == "ACCEPTED_TO_UNDETERMINED").sum()),
        "undetermined_to_accepted": int((merged.coverage_transition == "UNDETERMINED_TO_ACCEPTED").sum()),
        "undetermined_to_undetermined": int((merged.coverage_transition == "UNDETERMINED_TO_UNDETERMINED").sum()),
        "net_accepted_change": int(full["accepted"] - base["accepted"]),
        "coverage_delta": float(full["coverage"] - base["coverage"]),
        "accepted_to_undetermined_by_reference_stage": {str(k): int(v) for k, v in loss_by_stage.items()},
        "direct_mbm2_unknown_count": movement_direct,
        "boundary": "Only direct MBM.2.UNKNOWN rows are attributed to movement; remaining coverage loss is temporal/multifactor.",
    }
    write_json(output_root / "verification" / "coverage_denominator_audit.json", coverage_audit)
    return {"merged": merged, "coverage_audit": coverage_audit}


def _single_and_loo(
    source_root: Path,
    output_root: Path,
    frames: dict[str, pd.DataFrame],
    metrics: dict[str, dict[str, dict[str, Any]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_metric = metrics["primary"]["B000"]
    full_metric = metrics["primary"]["F_ALL15"]
    single_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    rule_rows: list[dict[str, Any]] = []
    loo_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for handler in HANDLERS:
        single_arm, loo_arm = f"S_{handler}", f"L_{handler}"
        single_metric, loo_metric = metrics["primary"][single_arm], metrics["primary"][loo_arm]
        paired_single = comparison_counts(frames["B000"], frames[single_arm])
        paired_full = comparison_counts(frames[loo_arm], frames["F_ALL15"])
        base_events = event_stats(source_root, "B000", EVENT_TYPE[handler])
        single_events = event_stats(source_root, single_arm, EVENT_TYPE[handler])
        base_evidence = evidence_stats(source_root, "B000", EVIDENCE_KEYS[handler])
        single_evidence = evidence_stats(source_root, single_arm, EVIDENCE_KEYS[handler])
        base_rules = rule_firing_counts(source_root, "B000", RULE_PREFIXES[handler])
        single_rules = rule_firing_counts(source_root, single_arm, RULE_PREFIXES[handler])

        single_row: dict[str, Any] = {
            "handler_id": handler, "event_type": EVENT_TYPE[handler], "arm_id": single_arm,
            **paired_single,
            "event_count_base": base_events["event_count"],
            "event_count_single": single_events["event_count"],
            "event_count_delta": single_events["event_count"] - base_events["event_count"],
            "event_density_delta_per_hour": single_events["event_density_per_hour"] - base_events["event_density_per_hour"],
            "valid_duration_delta_s": single_events["event_union_duration_s"] - base_events["event_union_duration_s"],
            "associated_event_count_delta": single_events["associated_event_count"] - base_events["associated_event_count"],
            "evidence_epoch_delta": single_evidence["evidence_positive_epochs"] - base_evidence["evidence_positive_epochs"],
            "evidence_seconds_delta": single_evidence["evidence_total_seconds"] - base_evidence["evidence_total_seconds"],
            "predicate_activation_delta": sum(single_rules.values()) - sum(base_rules.values()),
            "first_rule_divergence_count": int(
                (frames["B000"].query("in_primary == True").rule_id.to_numpy()
                 != frames[single_arm].query("in_primary == True").rule_id.to_numpy()).sum()
            ),
        }
        for name in ALL_EFFECT_METRICS:
            single_row[f"delta_{name}"] = _flat_value(single_metric, name) - _flat_value(base_metric, name)
        single_rows.append(single_row)

        for key in EVIDENCE_KEYS[handler]:
            evidence_rows.append({
                "handler_id": handler, "evidence_key": key,
                "base_positive_epochs": base_evidence["evidence_key_positive_epochs"].get(key, 0),
                "single_positive_epochs": single_evidence["evidence_key_positive_epochs"].get(key, 0),
                "delta_positive_epochs": single_evidence["evidence_key_positive_epochs"].get(key, 0) - base_evidence["evidence_key_positive_epochs"].get(key, 0),
            })
        for rule in sorted(set(base_rules) | set(single_rules)):
            rule_rows.append({
                "handler_id": handler, "rule_id": rule,
                "base_final_firings": base_rules.get(rule, 0),
                "single_final_firings": single_rules.get(rule, 0),
                "delta_final_firings": single_rules.get(rule, 0) - base_rules.get(rule, 0),
            })

        loo_row: dict[str, Any] = {
            "handler_id": handler, "event_type": EVENT_TYPE[handler], "leave_one_out_arm": loo_arm,
            **paired_full,
        }
        for name in ALL_EFFECT_METRICS:
            loo_row[f"full_contribution_{name}"] = _flat_value(full_metric, name) - _flat_value(loo_metric, name)
        loo_rows.append(loo_row)

        comparison_rows.append({
            "handler_id": handler,
            "single_accuracy": single_row["delta_accuracy"],
            "full_contribution_accuracy": loo_row["full_contribution_accuracy"],
            "accuracy_context_class": sign_class(single_row["delta_accuracy"], loo_row["full_contribution_accuracy"]),
            "single_macro_f1": single_row["delta_macro_f1"],
            "full_contribution_macro_f1": loo_row["full_contribution_macro_f1"],
            "macro_f1_context_class": sign_class(single_row["delta_macro_f1"], loo_row["full_contribution_macro_f1"]),
            "metric_dependent_sign_reversal": (
                sign_class(single_row["delta_accuracy"], loo_row["full_contribution_accuracy"])
                != sign_class(single_row["delta_macro_f1"], loo_row["full_contribution_macro_f1"])
            ),
        })

    single = pd.DataFrame(single_rows)
    loo = pd.DataFrame(loo_rows)
    comparison = pd.DataFrame(comparison_rows)
    single.to_csv(output_root / "single_swap" / "all_15_effects.csv", index=False)
    pd.DataFrame(evidence_rows).to_csv(output_root / "single_swap" / "evidence_changes.csv", index=False)
    pd.DataFrame(rule_rows).to_csv(output_root / "single_swap" / "rule_changes.csv", index=False)
    loo.to_csv(output_root / "leave_one_out" / "all_15_marginal_effects.csv", index=False)
    comparison.to_csv(output_root / "leave_one_out" / "single_vs_full_comparison.csv", index=False)
    comparison[
        (comparison.accuracy_context_class.str.contains("FULL"))
        | (comparison.macro_f1_context_class.str.contains("FULL"))
        | comparison.metric_dependent_sign_reversal
    ].to_csv(output_root / "leave_one_out" / "context_reversal_cases.csv", index=False)
    return single, loo, comparison


def _group_interactions(
    source_root: Path,
    output_root: Path,
    metrics: dict[str, dict[str, dict[str, Any]]],
) -> pd.DataFrame:
    base = metrics["primary"]["B000"]
    rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    for group, members in GROUPS.items():
        group_metric = metrics["primary"][group]
        for name in ALL_EFFECT_METRICS:
            observed = _flat_value(group_metric, name) - _flat_value(base, name)
            summed = sum(
                _flat_value(metrics["primary"][f"S_{member}"], name) - _flat_value(base, name)
                for member in members
            )
            rows.append({
                "group_id": group, "members": ";".join(members), "metric": name,
                "observed_group_delta": observed, "sum_single_delta": summed,
                "descriptive_nonadditivity": observed - summed,
                "interpretation_boundary": "descriptive frozen-system nonadditivity; not direct physiological causality",
            })
        keys = sorted(set(key for member in members for key in EVIDENCE_KEYS[member]))
        base_ev = evidence_stats(source_root, "B000", keys)
        group_ev = evidence_stats(source_root, group, keys)
        prefixes = sorted(set(prefix for member in members for prefix in RULE_PREFIXES[member]))
        base_rules = rule_firing_counts(source_root, "B000", prefixes)
        group_rules = rule_firing_counts(source_root, group, prefixes)
        mechanism_rows.append({
            "group_id": group, "members": ";".join(members),
            "changed_evidence_positive_epochs": group_ev["evidence_positive_epochs"] - base_ev["evidence_positive_epochs"],
            "changed_evidence_seconds": group_ev["evidence_total_seconds"] - base_ev["evidence_total_seconds"],
            "changed_relevant_rule_firings": sum(group_rules.values()) - sum(base_rules.values()),
            "shared_model_warning": "E09-E11 share one model" if group == "G_EMG_MOV" else "",
        })
    result = pd.DataFrame(rows)
    result.to_csv(output_root / "grouped" / "group_interactions.csv", index=False)
    pd.DataFrame(mechanism_rows).to_csv(output_root / "grouped" / "grouped_mechanisms.csv", index=False)
    return result


def _trace_pairs(
    source_root: Path,
    output_root: Path,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    pairs: list[tuple[str, str, str, str]] = [("B000", "F_ALL15", "FULL15", "ADD_15")]
    pairs.extend(("B000", f"S_{handler}", handler, "ADD_SINGLE") for handler in HANDLERS)
    pairs.extend(("F_ALL15", f"L_{handler}", handler, "REMOVE_FROM_FULL") for handler in HANDLERS)
    truth = reference[reference.in_primary.astype(bool)].set_index(["record_id", "epoch_index"])["true_stage"].to_dict()
    rows: list[dict[str, Any]] = []
    summary: Counter[tuple[str, str]] = Counter()
    for baseline_arm, comparison_arm, source, direction in pairs:
        base_pred = load_arm_payload(source_root, baseline_arm, "predictions")
        comp_pred = load_arm_payload(source_root, comparison_arm, "predictions")
        base_ev = load_arm_payload(source_root, baseline_arm, "evidence")
        comp_ev = load_arm_payload(source_root, comparison_arm, "evidence")
        for record_id in sorted(base_pred):
            for index, (left, right, left_ev, right_ev) in enumerate(
                zip(base_pred[record_id], comp_pred[record_id], base_ev[record_id], comp_ev[record_id])
            ):
                key = (record_id, index)
                if key not in truth:
                    continue
                changed_evidence = changed_evidence_keys(left_ev, right_ev)
                temporal_keys = changed_temporal_keys(left, right)
                temporal_changed = bool(temporal_keys)
                stage_changed = left.get("stage") != right.get("stage")
                rule_changed = left.get("rule_id") != right.get("rule_id")
                if not (changed_evidence or temporal_changed or stage_changed or rule_changed):
                    continue
                first_rule = first_rule_difference(left, right)
                if str(left.get("rule_id", "")).startswith("MBM") or str(right.get("rule_id", "")).startswith("MBM"):
                    trace_kind = "RULE_PREEMPTION_MOVEMENT"
                    first_rule = f"MBM_PREEMPTION:{left.get('rule_id')}->{right.get('rule_id')}"
                elif changed_evidence and temporal_changed:
                    trace_kind = "MULTIFACTOR_DIRECT_AND_PROPAGATED"
                elif changed_evidence:
                    trace_kind = "DIRECT"
                elif temporal_changed:
                    trace_kind = "PROPAGATED"
                elif rule_changed:
                    trace_kind = "RULE_PRIORITY_OR_PREEMPTION"
                else:
                    trace_kind = "UNRESOLVED"
                confidence = "LOW" if source == "FULL15" or trace_kind.startswith("MULTIFACTOR") else ("HIGH" if trace_kind == "DIRECT" else "MEDIUM")
                summary[(f"{baseline_arm}_vs_{comparison_arm}", trace_kind)] += 1
                left_correct = left.get("stage") == truth[key]
                right_correct = right.get("stage") == truth[key]
                if source == "FULL15":
                    mapped_sources = sorted(
                        handler for handler, keys in EVIDENCE_KEYS.items()
                        if set(keys) & set(changed_evidence)
                    )
                    changed_sources = ";".join(mapped_sources) or "MULTISOURCE_UNRESOLVED"
                else:
                    changed_sources = source
                rows.append({
                    "comparison": f"{baseline_arm}_vs_{comparison_arm}",
                    "comparison_direction": direction, "record_id": record_id,
                    "epoch_index": index, "reference_stage": truth[key],
                    "baseline_prediction": left.get("stage"), "comparison_prediction": right.get("stage"),
                    "baseline_rule": left.get("rule_id"), "comparison_rule": right.get("rule_id"),
                    "changed_event_sources": changed_sources,
                    "changed_evidence": ";".join(changed_evidence),
                    "first_changed_rule": first_rule,
                    "first_changed_temporal_state": ";".join(temporal_keys),
                    "direct_or_propagated": trace_kind,
                    "attribution_confidence": confidence,
                    "stage_changed": stage_changed,
                    "corrected": (not left_correct and right_correct),
                    "introduced": (left_correct and not right_correct),
                    "provenance": f"traces/{baseline_arm}/{record_id}.json.gz|traces/{comparison_arm}/{record_id}.json.gz",
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "mechanisms" / "first_divergence_trace.csv", index=False)
    summary_rows = [
        {"comparison": pair, "direct_or_propagated": kind, "count": count}
        for (pair, kind), count in sorted(summary.items())
    ]
    pd.DataFrame(summary_rows).to_csv(output_root / "mechanisms" / "temporal_propagation.csv", index=False)
    return frame


def _predicate_value(row: dict[str, Any], prefix: str) -> str:
    for item in row.get("trace", []):
        if str(item.get("rule_id", "")).startswith(prefix):
            return str(item.get("value", ""))
    return "NOT_EVALUATED_PREEMPTED"


def _case_study_data(
    source_root: Path,
    reference: pd.DataFrame,
    handler: str,
    comparison_arm: str,
    evidence_key: str,
    predicate_prefix: str,
    threshold: float,
    inclusive: bool,
    threshold_label: str,
) -> dict[str, Any]:
    base_ev = load_arm_payload(source_root, "B000", "evidence")
    comp_ev = load_arm_payload(source_root, comparison_arm, "evidence")
    base_pred = load_arm_payload(source_root, "B000", "predictions")
    comp_pred = load_arm_payload(source_root, comparison_arm, "predictions")
    truth = reference[reference.in_primary.astype(bool)].set_index(["record_id", "epoch_index"])["true_stage"].to_dict()
    threshold_cross_down = threshold_cross_up = insufficient = 0
    predicate_transitions: Counter[str] = Counter()
    stage_transitions: Counter[str] = Counter()
    corrected = introduced = 0
    for record in sorted(base_ev):
        for index, (be, ce, bp, cp) in enumerate(zip(base_ev[record], comp_ev[record], base_pred[record], comp_pred[record])):
            if (record, index) not in truth:
                continue
            b_seconds = float(be.get("events", {}).get(evidence_key, {}).get("seconds") or 0.0)
            c_seconds = float(ce.get("events", {}).get(evidence_key, {}).get("seconds") or 0.0)
            b_qualified = b_seconds >= threshold if inclusive else b_seconds > threshold
            c_qualified = c_seconds >= threshold if inclusive else c_seconds > threshold
            threshold_cross_down += int(b_qualified and not c_qualified)
            threshold_cross_up += int(not b_qualified and c_qualified)
            insufficient += int(0 < c_seconds and not c_qualified)
            b_value, c_value = _predicate_value(bp, predicate_prefix), _predicate_value(cp, predicate_prefix)
            if b_value != c_value:
                predicate_transitions[f"{b_value}->{c_value}"] += 1
            if bp.get("stage") != cp.get("stage"):
                stage_transitions[f"{truth[(record, index)]}:{bp.get('stage')}->{cp.get('stage')}"] += 1
            b_ok = bp.get("stage") == truth[(record, index)]
            c_ok = cp.get("stage") == truth[(record, index)]
            corrected += int(not b_ok and c_ok)
            introduced += int(b_ok and not c_ok)
    return {
        "handler_id": handler,
        "threshold_label": threshold_label,
        "base_event": event_stats(source_root, "B000", EVENT_TYPE[handler]),
        "comparison_event": event_stats(source_root, comparison_arm, EVENT_TYPE[handler]),
        "threshold_cross_down": threshold_cross_down,
        "threshold_cross_up": threshold_cross_up,
        "comparison_insufficient_0_to_6s_epochs": insufficient,
        "predicate_transitions": dict(predicate_transitions),
        "top_stage_transitions": dict(stage_transitions.most_common(25)),
        "corrected": corrected, "introduced": introduced, "net_correct": corrected - introduced,
    }


def _write_case_studies(
    source_root: Path,
    output_root: Path,
    reference: pd.DataFrame,
    metrics: dict[str, dict[str, dict[str, Any]]],
    global_info: dict[str, Any],
) -> dict[str, Any]:
    e01 = _case_study_data(source_root, reference, "E01", "S_E01", "slow_wave", "N3.1", 6.0, True, "slow_wave >= 6 s")
    e02 = _case_study_data(source_root, reference, "E02", "S_E02", "n2_kcomplex", "N2.1", 0.0, False, "n2_kcomplex > 0 s")
    e13 = _case_study_data(source_root, reference, "E13", "S_E13", "mbm", "MBM", 15.0, False, "mbm > 15 s")
    ar = _case_study_data(source_root, reference, "AR", "S_AR", "arousal", "N2", 0.0, False, "arousal > 0 s")
    sp = _case_study_data(source_root, reference, "SP", "S_SP", "n2_spindle", "N2.1", 0.0, False, "n2_spindle > 0 s")
    cases = {"E01": e01, "E02": e02, "E13": e13, "AR": ar, "SP": sp}
    write_json(output_root / "mechanisms" / "case_study_counts.json", cases)

    def delta(arm: str, name: str) -> float:
        return _flat_value(metrics["primary"][arm], name) - _flat_value(metrics["primary"]["B000"], name)

    (output_root / "mechanisms" / "e01_slowwave_n3.md").write_text(f"""# E01 Slow-wave → N3 regression

## Observed chain

- Frozen event count: **{e01['base_event']['event_count']}**; SSSM event count: **{e01['comparison_event']['event_count']}**.
- Union duration changed by **{e01['comparison_event']['event_union_duration_s'] - e01['base_event']['event_union_duration_s']:+.1f} s**.
- Frozen `slow_wave >= 6 s` but SSSM `< 6 s`: **{e01['threshold_cross_down']} epochs**; reverse crossing: **{e01['threshold_cross_up']} epochs**.
- SSSM had a candidate but `< 6 s`: **{e01['comparison_insufficient_0_to_6s_epochs']} epochs**.
- Single-swap ΔN3 F1: **{delta('S_E01', 'N3_f1'):+.6f}**; corrected/introduced/net: **{e01['corrected']}/{e01['introduced']}/{e01['net_correct']}**.

This is a trace-supported rule-coverage failure: the replacement supplies less rule-qualifying slow-wave duration at the frozen six-second predicate. It is not proof that missing SSSM candidates are physiological false negatives because HMC has no independent slow-wave event truth.
""", encoding="utf-8")

    (output_root / "mechanisms" / "e02_kcomplex_n2.md").write_text(f"""# E02 K-complex → N2 / Macro-F1 tradeoff

## Observed chain

- Frozen event count: **{e02['base_event']['event_count']}**; K08 event count: **{e02['comparison_event']['event_count']}**.
- Associated-event count changed by **{e02['comparison_event']['associated_event_count'] - e02['base_event']['associated_event_count']:+d}**.
- Rule-qualifying `n2_kcomplex > 0 s` changed positive→zero in **{e02['threshold_cross_down']} epochs** and zero→positive in **{e02['threshold_cross_up']} epochs**.
- Single ΔAccuracy **{delta('S_E02', 'accuracy'):+.6f}**, ΔN2 F1 **{delta('S_E02', 'N2_f1'):+.6f}**, ΔREM F1 **{delta('S_E02', 'R_f1'):+.6f}**, ΔMacro-F1 **{delta('S_E02', 'macro_f1'):+.6f}**.
- Corrected/introduced/net: **{e02['corrected']}/{e02['introduced']}/{e02['net_correct']}**.

Macro-F1 improves because equal-weight REM/other-stage gains can outweigh N2 loss in the macro average, while Accuracy is dominated by the large N2 support. In Full context E02 remains Accuracy-negative but Macro-positive; the trace shows context-dependent reuse of K-complex evidence by N2 and REM paths rather than a global quality gain.
""", encoding="utf-8")

    movement = global_info["coverage_audit"]
    (output_root / "mechanisms" / "e13_movement_coverage.md").write_text(f"""# E13 Movement proxy → Coverage tradeoff

## Observed chain

- Frozen MBM count: **{e13['base_event']['event_count']}**; neural proxy count: **{e13['comparison_event']['event_count']}**.
- Proxy union duration changed by **{e13['comparison_event']['event_union_duration_s'] - e13['base_event']['event_union_duration_s']:+.1f} s**.
- Single ΔWake F1 **{delta('S_E13', 'W_f1'):+.6f}**, ΔMacro-F1 **{delta('S_E13', 'macro_f1'):+.6f}**, ΔCoverage **{delta('S_E13', 'coverage'):+.6f}**.
- Full system Accepted→UNDETERMINED: **{movement['accepted_to_undetermined']}**, of which directly traceable `MBM.2.UNKNOWN`: **{movement['direct_mbm2_unknown_count']}**.

The proxy can preempt ordinary rules and enter the movement pending/unknown path. This raises Wake/Macro-F1 on some epochs while converting many previously accepted epochs to abstentions. Only direct MBM rows are assigned to movement; remaining coverage loss is temporal or multifactor. E13 remains `MOVEMENT_PROXY`, not a validated MBM detector.
""", encoding="utf-8")

    (output_root / "mechanisms" / "arousal_positive_control.md").write_text(f"""# Arousal U-Net positive control

- Event count change: **{ar['comparison_event']['event_count'] - ar['base_event']['event_count']:+d}**.
- ΔAccuracy **{delta('S_AR', 'accuracy'):+.6f}**; ΔMacro-F1 **{delta('S_AR', 'macro_f1'):+.6f}**; ΔN1 F1 **{delta('S_AR', 'N1_f1'):+.6f}**.
- Corrected/introduced/net: **{ar['corrected']}/{ar['introduced']}/{ar['net_correct']}**.

The source changes event proposals that are directly consumed by transition/reset logic and produces a positive net correction, unlike duration-starved E01 or abstention-heavy E13. This is internal development-set evidence, not cross-center clinical validation.
""", encoding="utf-8")

    (output_root / "mechanisms" / "spindle_positive_control.md").write_text(f"""# SUMOv2 Spindle positive control

- Event count change: **{sp['comparison_event']['event_count'] - sp['base_event']['event_count']:+d}**.
- ΔAccuracy **{delta('S_SP', 'accuracy'):+.6f}**; ΔN2 F1 **{delta('S_SP', 'N2_f1'):+.6f}**; ΔMacro-F1 **{delta('S_SP', 'macro_f1'):+.6f}**.
- Corrected/introduced/net: **{sp['corrected']}/{sp['introduced']}/{sp['net_correct']}**.

Spindle evidence reaches the frozen N2 structural predicates with a positive Accuracy/N2 net effect. Its single-swap Macro-F1 is slightly negative, but its Full-context marginal contribution is positive, demonstrating context-dependent complementarity rather than universal benefit.
""", encoding="utf-8")
    return cases


def _per_night_and_bootstrap(
    output_root: Path,
    frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons: list[tuple[str, str, str]] = [("B000", "F_ALL15", "FULL15")]
    comparisons.extend(("B000", f"S_{handler}", f"SINGLE_{handler}") for handler in HANDLERS)
    comparisons.extend((f"L_{handler}", "F_ALL15", f"FULL_CONTRIBUTION_{handler}") for handler in HANDLERS)
    per_night: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for base_arm, comp_arm, comparison in comparisons:
        for record_id in sorted(frames[base_arm].record_id.unique()):
            base_subset = frames[base_arm][(frames[base_arm].record_id == record_id) & frames[base_arm].in_primary.astype(bool)]
            comp_subset = frames[comp_arm][(frames[comp_arm].record_id == record_id) & frames[comp_arm].in_primary.astype(bool)]
            base_metric = metric_for_arm(base_subset.assign(in_primary=True))
            comp_metric = metric_for_arm(comp_subset.assign(in_primary=True))
            row = {"comparison": comparison, "base_arm": base_arm, "comparison_arm": comp_arm, "record_id": record_id}
            for name in ALL_EFFECT_METRICS:
                row[f"delta_{name}"] = _flat_value(comp_metric, name) - _flat_value(base_metric, name)
            per_night.append(row)
    frame = pd.DataFrame(per_night)
    for comparison, group in frame.groupby("comparison"):
        for name in ALL_EFFECT_METRICS:
            values = group[f"delta_{name}"].to_numpy(float)
            summary = bootstrap_summary(values, int(config["bootstrap_draws"]), int(config["bootstrap_seed"]))
            summaries.append({"comparison": comparison, "metric": name, **summary, "n_records": len(values)})
    summary_frame = pd.DataFrame(summaries)
    frame.to_csv(output_root / "statistics" / "per_night_metrics.csv", index=False)
    summary_frame.to_csv(output_root / "statistics" / "cluster_bootstrap.csv", index=False)
    return frame, summary_frame


def _pareto(metrics: dict[str, dict[str, dict[str, Any]]], output_root: Path) -> pd.DataFrame:
    rows = []
    for arm, metric in metrics["primary"].items():
        rows.append({
            "arm_id": arm, "accuracy": metric["accuracy"], "macro_f1": metric["macro_f1"],
            "coverage": metric["coverage"], "N3_f1": metric["per_stage"]["N3"]["f1"],
            "R_f1": metric["per_stage"]["R"]["f1"],
        })
    frame = pd.DataFrame(rows)
    values = frame[["accuracy", "macro_f1", "coverage", "N3_f1", "R_f1"]].to_numpy()
    nondominated = []
    for index, value in enumerate(values):
        dominated = any(np.all(other >= value) and np.any(other > value) for j, other in enumerate(values) if j != index)
        nondominated.append(not dominated)
    frame["pareto_nondominated"] = nondominated
    frame.to_csv(output_root / "statistics" / "pareto_analysis.csv", index=False)
    return frame


def run_analysis(
    config: dict[str, Any],
    output_root: Path,
    reference: pd.DataFrame,
) -> dict[str, Any]:
    _ensure_directories(output_root)
    source_root = Path(config["source_root"])
    manifest = pd.read_csv(source_root / "arms" / "arm_manifest_36.csv")
    arm_ids = manifest.arm_id.tolist()
    if len(arm_ids) != 36 or len(pd.read_csv(source_root / "arms" / "arm_execution_10night.csv")) != 360:
        raise RuntimeError("Frozen 36-arm/360-night heritage is incomplete")
    frames = {arm: arm_prediction_frame(source_root, arm, reference) for arm in arm_ids}
    metrics: dict[str, dict[str, dict[str, Any]]] = {"primary": {}, "full": {}}
    for arm, frame in frames.items():
        metrics["primary"][arm] = metric_for_arm(frame, "in_primary")
        metrics["full"][arm] = metric_for_arm(frame, "in_full_record")
    metric_recheck = _metric_recheck(source_root, output_root, arm_ids, frames, metrics)
    global_info = _global_tradeoff(output_root, frames, metrics)
    single, loo, comparison = _single_and_loo(source_root, output_root, frames, metrics)
    interactions = _group_interactions(source_root, output_root, metrics)
    traces = _trace_pairs(source_root, output_root, reference)
    cases = _write_case_studies(source_root, output_root, reference, metrics, global_info)
    per_night, bootstrap = _per_night_and_bootstrap(output_root, frames, config)
    pareto = _pareto(metrics, output_root)
    write_json(output_root / "verification" / "all_recomputed_metrics.json", {
        scope: {arm: flatten_metric(metric) for arm, metric in values.items()}
        for scope, values in metrics.items()
    })
    return {
        "arm_ids": arm_ids, "frames": frames, "metrics": metrics,
        "metric_recheck": metric_recheck, "global": global_info,
        "single": single, "loo": loo, "comparison": comparison,
        "interactions": interactions, "traces": traces, "cases": cases,
        "per_night": per_night, "bootstrap": bootstrap, "pareto": pareto,
    }
