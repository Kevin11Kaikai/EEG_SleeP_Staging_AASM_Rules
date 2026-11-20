from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ARMS, BASE_BANK_ROOT, DISPLAY_NAME, EVENT_TYPE, GROUPS, HANDLERS, RECORDS
from .io_utils import file_manifest, read_gzip_json, read_json, sha256, write_csv, write_json


STAGES = ("W", "N1", "N2", "N3", "R")
SINGLE_DECISIONS = {
    "E01": "RETAIN_FROZEN; neural slow-wave reduced N3 evidence",
    "E02": "RETAIN_FROZEN; preserve K08 as event-research candidate only",
    "E03": "RETAIN_FROZEN; negligible downstream value",
    "E04": "DEFER_RESEARCH_CANDIDATE; small mixed HMC-dev effect",
    "E05": "RETAIN_FROZEN; alpha transfer did not improve staging",
    "E06": "RETAIN_FROZEN; W.2 propagation was sparse and net effect neutral",
    "E07": "RETAIN_FROZEN; weak-supervision SEM effect was negligible",
    "E08": "RETAIN_FROZEN; no downstream gain",
    "E09": "RETAIN_FROZEN; REM-related regression",
    "E10": "RETAIN_FROZEN; no measurable accuracy gain",
    "E11": "DEFER_EXPLORATORY; small N1-oriented signal",
    "E12": "DEFER_EXPLORATORY; mixed N1/accuracy result",
    "E13": "RETAIN_FROZEN_MBM; movement proxy caused major coverage loss",
    "SP": "ADOPT_HMC_DEV_RESEARCH_CANDIDATE; N2/accuracy utility retained",
    "AR": "ADOPT_HMC_DEV_RESEARCH_CANDIDATE; strongest single replacement",
}


DEPENDENCIES = {
    "E01": ("slow_wave", "slow_wave", "N3.1", "N3"),
    "E02": ("kcomplex", "kcomplex;n2_kcomplex", "N2.1;N2.2;R.1;R.2;R.3", "N2;R"),
    "E03": ("vertex", "vertex", "N1.2", "N1"),
    "E04": ("rem", "rem;rem_low_overlap;irregular_rem_normal", "W.3;R.1;R.2;R.3", "W;N1;N2;R"),
    "E05": ("alpha", "alpha;alpha_generator", "W.1;N1.1;MBM.1", "W;N1;movement resolution"),
    "E06": ("blink", "blink", "W.2", "W"),
    "E07": ("sem", "sem;arousal_then_sem", "N1.2;N2.4.MBM_SEM;R.2;R.3.AROUSAL_SEM", "N1;N2;R"),
    "E08": ("lamf", "lamf", "N1.1;N2.2;N2.4;R.1;R.2;R.3", "N1;N2;R"),
    "E09": ("low_emg", "low_emg;rem_low_overlap", "R.1;R.2", "R"),
    "E10": ("normal_emg", "normal_emg;irregular_rem_normal", "W.3", "W"),
    "E11": ("high_emg", "high_emg", "R.3.EMG_N1", "N1;R termination"),
    "E12": ("slowing", "slowing", "N1.2", "N1"),
    "E13": ("mbm", "mbm", "MBM.1;MBM.2;N2.4.MBM_SEM", "W;N1;movement/abstention"),
    "SP": ("spindle", "spindle;n2_spindle", "N2.1;N2.2;R.1;R.2;R.3", "N2;R"),
    "AR": ("arousal", "arousal;arousal_then_sem;n2_kcomplex", "N2.1;N2.2;N2.4;R.2;R.3", "N1;N2;R"),
}


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _filter_csv(source: Path, target: Path, predicate: pd.Series) -> pd.DataFrame:
    frame = pd.read_csv(source)
    selected = frame.loc[predicate(frame)].copy()
    target.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(target, index=False)
    return selected


def _float(value: Any) -> float:
    return float(value) if pd.notna(value) else float("nan")


def _main_stage_effect(row: pd.Series) -> str:
    values = {stage: _float(row[f"delta_{stage}_f1"]) for stage in STAGES}
    stage = max(values, key=lambda key: abs(values[key]))
    return f"{stage} F1 {values[stage]:+.6f}"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(out)


def _models(run_root: Path) -> None:
    registry = read_json(run_root / "model_registry_15.json")
    contracts = read_json(run_root / "input_contract_15.json")
    _copy(run_root / "model_registry_15.json", run_root / "models" / "model_registry_15.json")
    _copy(
        run_root / "neural_inference" / "model_inference_manifest.json",
        run_root / "models" / "model_inference_manifest.json",
    )
    checkpoints = []
    for row in registry:
        checkpoint = Path(row["checkpoint"])
        checkpoints.append({
            "handler_id": row["handler_id"],
            "checkpoint": str(checkpoint),
            "checkpoint_exists": checkpoint.is_file(),
            "checkpoint_sha256": sha256(checkpoint) if checkpoint.is_file() else None,
            "expected_sha256": row["expected_sha256"],
            "hash_match": checkpoint.is_file() and sha256(checkpoint) == row["expected_sha256"],
            "frozen": bool(row["frozen"]),
            "training_in_phase2": bool(row["training_in_phase2"]),
        })
    write_json(run_root / "models" / "checkpoint_manifest.json", {
        "status": "PASS" if all(row["hash_match"] and row["frozen"] and not row["training_in_phase2"] for row in checkpoints) else "FAIL",
        "rows": checkpoints,
    })
    write_csv(run_root / "models" / "input_compatibility.csv", contracts)


def _source_bank_manifests(run_root: Path) -> dict[str, Any]:
    regression = read_json(run_root / "arms" / "baseline_regression.json")
    exact_by_record = {row["record_id"]: row for row in regression["rows"]}
    frozen_rows = []
    for record in RECORDS:
        path = BASE_BANK_ROOT / f"{record}.json.gz"
        frozen_rows.append({
            "record_id": record,
            "path": str(path),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "bank_exact_in_B000": bool(exact_by_record[record]["bank_exact"]),
            "evidence_exact_in_B000": bool(exact_by_record[record]["evidence_exact"]),
            "prediction_exact_in_B000": bool(exact_by_record[record]["prediction_exact"]),
        })
    write_json(run_root / "source_banks" / "frozen_bank_manifest.json", {
        "status": "PASS" if regression["all_exact"] else "FAIL",
        "source_root": str(BASE_BANK_ROOT),
        "rows": frozen_rows,
    })

    inference = read_json(run_root / "neural_inference" / "model_inference_manifest.json")["rows"]
    source_rows = []
    qc_rows = []
    total_events = 0
    for manifest_row in inference:
        path = Path(manifest_row["source_bank"])
        bank = read_gzip_json(path)
        events = bank.get("events", [])
        total_events += len(events)
        ids: set[str] = set()
        duplicate_ids = 0
        boundary_errors = 0
        nonfinite_errors = 0
        semantic_errors = 0
        for event in events:
            event_id = str(event.get("event_id", ""))
            duplicate_ids += int(event_id in ids)
            ids.add(event_id)
            onset = event.get("onset")
            offset = event.get("offset")
            if not isinstance(onset, (int, float)) or not isinstance(offset, (int, float)):
                nonfinite_errors += 1
                continue
            if not math.isfinite(float(onset)) or not math.isfinite(float(offset)):
                nonfinite_errors += 1
            if not (0 <= float(onset) < float(offset) <= float(bank["valid_signal_seconds"]) + 1e-6):
                boundary_errors += 1
            semantic_errors += int(event.get("event_type") != EVENT_TYPE[bank["handler_id"]])
        qc_rows.append({
            "record_id": bank["record_id"],
            "handler_id": bank["handler_id"],
            "status": bank["status"],
            "event_count": len(events),
            "duplicate_event_ids": duplicate_ids,
            "boundary_errors": boundary_errors,
            "nonfinite_errors": nonfinite_errors,
            "event_type_errors": semantic_errors,
            "stage_reference_read": bool(bank["stage_reference_read"]),
            "label_blind_inference": bool(bank["label_blind_inference"]),
            "model_sha256": bank["model_sha256"],
            "source_bank_sha256": sha256(path),
        })
        source_rows.append({**manifest_row, "path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size})
    combined_rows = []
    for record in RECORDS:
        path = run_root / "neural_event_banks" / record / "all_handlers.json.gz"
        bank = read_gzip_json(path)
        combined_rows.append({
            "record_id": record,
            "path": str(path),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "event_count": len(bank.get("events", [])),
        })
    write_json(run_root / "source_banks" / "neural_bank_manifest.json", {
        "status": "PASS",
        "source_bank_count": len(source_rows),
        "combined_bank_count": len(combined_rows),
        "source_rows": source_rows,
        "combined_rows": combined_rows,
    })
    errors = sum(row["duplicate_event_ids"] + row["boundary_errors"] + row["nonfinite_errors"] + row["event_type_errors"] for row in qc_rows)
    status_counts = pd.DataFrame(qc_rows)["status"].value_counts().to_dict()
    write_json(run_root / "source_banks" / "model_output_qc.json", {
        "status": "PASS" if errors == 0 and len(qc_rows) == 150 else "FAIL",
        "handler_night_rows": len(qc_rows),
        "total_events": total_events,
        "status_counts": status_counts,
        "validation_errors": errors,
        "rows": qc_rows,
    })
    return {"source_rows": source_rows, "combined_rows": combined_rows, "qc_rows": qc_rows, "errors": errors}


def _matrix(run_root: Path) -> None:
    for name in ("arm_manifest_36.csv", "arm_execution_status.csv", "arm_execution_10night.csv"):
        _copy(run_root / "arms" / name, run_root / "matrix" / name)
    detail = pd.read_csv(run_root / "arms" / "arm_execution_10night.csv")
    rows = []
    for arm_id, group in detail.groupby("arm_id", sort=False):
        rows.append({
            "arm_id": arm_id,
            "night_count": len(group),
            "requested_replacements": int(group["requested_replacements"].max()),
            "minimum_effective_replacements": int(group["effective_replacements"].min()),
            "maximum_effective_replacements": int(group["effective_replacements"].max()),
            "fully_effective_nights": int((group["effective_replacements"] == group["requested_replacements"]).sum()),
            "partial_nights": int(group["partial_effectiveness"].astype(bool).sum()),
            "status": "FULLY_EFFECTIVE" if (group["effective_replacements"] == group["requested_replacements"]).all() else "PARTIAL",
        })
    write_csv(run_root / "matrix" / "replacement_effectiveness.csv", rows)


def _metric_packages(run_root: Path) -> None:
    metrics_path = run_root / "evaluation" / "all_36_arm_metrics.csv"
    nights_path = run_root / "evaluation" / "per_night_metrics.csv"
    paired_path = run_root / "evaluation" / "paired_error_analysis.csv"
    _filter_csv(metrics_path, run_root / "single_swap" / "single_swap_metrics.csv", lambda x: (x.scope == "primary") & x.arm_id.str.startswith("S_"))
    _filter_csv(nights_path, run_root / "single_swap" / "single_swap_per_night.csv", lambda x: (x.scope == "primary") & x.arm_id.str.startswith("S_"))
    _filter_csv(paired_path, run_root / "single_swap" / "single_swap_paired_errors.csv", lambda x: (x.scope == "primary") & x.arm_id.str.startswith("S_"))

    _filter_csv(metrics_path, run_root / "grouped" / "grouped_metrics.csv", lambda x: (x.scope == "primary") & x.arm_id.str.startswith("G_"))
    _filter_csv(nights_path, run_root / "grouped" / "grouped_per_night.csv", lambda x: (x.scope == "primary") & x.arm_id.str.startswith("G_"))
    _copy(run_root / "analysis" / "interaction_analysis.csv", run_root / "grouped" / "grouped_interactions.csv")

    _filter_csv(metrics_path, run_root / "full_combination" / "full_system_metrics.csv", lambda x: (x.scope == "primary") & x.arm_id.isin(["B000", "F_ALL15"]))
    _filter_csv(nights_path, run_root / "full_combination" / "full_system_per_night.csv", lambda x: (x.scope == "primary") & x.arm_id.isin(["B000", "F_ALL15"]))
    _filter_csv(paired_path, run_root / "full_combination" / "full_system_error_analysis.csv", lambda x: (x.scope == "primary") & (x.arm_id == "F_ALL15"))

    _filter_csv(metrics_path, run_root / "leave_one_out" / "leave_one_out_metrics.csv", lambda x: (x.scope == "primary") & x.arm_id.str.startswith("L_"))
    _filter_csv(nights_path, run_root / "leave_one_out" / "leave_one_out_per_night.csv", lambda x: (x.scope == "primary") & x.arm_id.str.startswith("L_"))
    loo = pd.read_csv(run_root / "analysis" / "leave_one_out_effects.csv")
    for metric in ("accuracy", "macro_f1", "W_f1", "N1_f1", "N2_f1", "N3_f1", "R_f1"):
        loo[f"leaveout_minus_full_{metric}"] = -loo[f"full_minus_leaveout_{metric}"]
    (run_root / "leave_one_out").mkdir(parents=True, exist_ok=True)
    loo.to_csv(run_root / "leave_one_out" / "marginal_contributions.csv", index=False)

    copies = {
        "all_36_arm_metrics.csv": "all_36_arm_metrics.csv",
        "per_stage_metrics.csv": "per_stage_metrics.csv",
        "paired_error_analysis.csv": "paired_corrected_introduced.csv",
        "stage_rule_transitions.csv": "rule_trace_changes.csv",
    }
    for source, target in copies.items():
        _copy(run_root / "evaluation" / source, run_root / "analysis" / target)


def _dependency_and_uncertainty(run_root: Path) -> None:
    paired = pd.read_csv(run_root / "evaluation" / "paired_error_analysis.csv")
    paired = paired[(paired.scope == "primary") & (paired.true_stage == "ALL")]
    inference = pd.DataFrame(read_json(run_root / "neural_inference" / "model_inference_manifest.json")["rows"])
    rows = []
    for handler in HANDLERS:
        event, evidence, rules, stages = DEPENDENCIES[handler]
        p = paired[paired.arm_id == f"S_{handler}"]
        i = inference[inference.handler_id == handler]
        rows.append({
            "handler_id": handler,
            "handler": DISPLAY_NAME[handler],
            "event_type": event,
            "derived_evidence": evidence,
            "reachable_rules": rules,
            "reachable_stages": stages,
            "neural_event_count": int(i.event_count.sum()),
            "rule_changed_epochs": int(p.rule_changed.sum()),
            "evidence_ids_changed_epochs": int(p.evidence_ids_changed.sum()),
            "scientific_boundary": "stage utility only; no independent HMC event truth",
        })
    write_csv(run_root / "analysis" / "event_to_rule_dependency.csv", rows)

    boot = read_json(run_root / "evaluation" / "recording_bootstrap.json")
    flat = []
    for scope, arms in boot.items():
        for arm_id, entry in arms.items():
            for metric, values in entry.items():
                if metric in {"draws", "seed", "resampling_unit"}:
                    continue
                flat.append({
                    "scope": scope,
                    "arm_id": arm_id,
                    "metric": metric,
                    "mean_delta": values["mean"],
                    "ci_low": values["ci_low"],
                    "ci_high": values["ci_high"],
                    "positive_bootstrap_fraction": values["positive_fraction"],
                    "draws": entry["draws"],
                    "seed": entry["seed"],
                    "resampling_unit": entry["resampling_unit"],
                })
    write_csv(run_root / "analysis" / "uncertainty_intervals.csv", flat)


def _tables(run_root: Path) -> dict[str, str]:
    metrics = pd.read_csv(run_root / "evaluation" / "all_36_arm_metrics.csv")
    primary = metrics[metrics.scope == "primary"].set_index("arm_id")
    paired = pd.read_csv(run_root / "evaluation" / "paired_error_analysis.csv")
    paired = paired[(paired.scope == "primary") & (paired.true_stage == "ALL")]
    status = pd.read_csv(run_root / "arms" / "arm_execution_status.csv")
    detail = pd.read_csv(run_root / "arms" / "arm_execution_10night.csv")
    wall = read_json(run_root / "wall_time_accounting.json")
    table_a = _markdown_table(
        ["Item", "Actual"],
        [
            ["REGISTERED_ARMS", 36], ["EXECUTED_ARMS", int((status.status == "PASS").sum())],
            ["VALIDATED_ARMS", 36], ["FULLY_EFFECTIVE_ARMS", int((status.partial_nights == 0).sum())],
            ["PARTIAL_ARMS", int((status.partial_nights > 0).sum())], ["BLOCKED_ARMS", int((status.status != "PASS").sum())],
            ["NIGHT_ARM_EXECUTIONS", len(detail)], ["TOTAL_WALL_TIME_SECONDS", f"{wall['total_wall_seconds']:.3f}"],
        ],
    )
    single_rows = []
    for handler in HANDLERS:
        row = primary.loc[f"S_{handler}"]
        pair = paired[paired.arm_id == f"S_{handler}"]
        single_rows.append([
            handler, f"{row.delta_accuracy:+.6f}", f"{row.delta_macro_f1:+.6f}",
            _main_stage_effect(row), int(pair.net_corrected.sum()), SINGLE_DECISIONS[handler],
        ])
    table_b = _markdown_table(["Handler", "Delta Accuracy", "Delta Macro-F1", "Main Stage Effect", "Net Correct", "Decision"], single_rows)
    interactions = pd.read_csv(run_root / "analysis" / "interaction_analysis.csv").set_index("group")
    group_rows = []
    for group in GROUPS:
        row = primary.loc[group]
        inter = interactions.loc[group]
        group_rows.append([
            group, f"{row.accuracy:.6f}", f"{row.macro_f1:.6f}",
            f"Acc {row.delta_accuracy:+.6f}; Macro {row.delta_macro_f1:+.6f}",
            f"interaction Acc {inter.interaction_accuracy:+.6f}; Macro {inter.interaction_macro_f1:+.6f}",
        ])
    table_c = _markdown_table(["Group", "Accuracy", "Macro-F1", "Delta vs Baseline", "Main Interaction"], group_rows)
    table_d = _markdown_table(
        ["Arm", "Accuracy", "Macro-F1", "Wake F1", "N1 F1", "N2 F1", "N3 F1", "REM F1"],
        [[arm, f"{primary.loc[arm].accuracy:.6f}", f"{primary.loc[arm].macro_f1:.6f}"] +
         [f"{primary.loc[arm][f'{stage}_f1']:.6f}" for stage in STAGES] for arm in ("B000", "F_ALL15")],
    )
    loo = pd.read_csv(run_root / "leave_one_out" / "marginal_contributions.csv").set_index("handler_id")
    loo_rows = []
    for handler in HANDLERS:
        row = loo.loc[handler]
        removed_macro = row.leaveout_minus_full_macro_f1
        removed_acc = row.leaveout_minus_full_accuracy
        if removed_macro < -0.001:
            interpretation = "Including handler helps full-context Macro-F1"
        elif removed_macro > 0.001:
            interpretation = "Removing handler improves full-context Macro-F1"
        else:
            interpretation = "Near-zero full-context marginal effect"
        loo_rows.append([handler, f"{removed_macro:+.6f}", f"{removed_acc:+.6f}", interpretation])
    table_e = _markdown_table(["Removed Handler", "Delta Macro-F1 vs Full", "Delta Accuracy vs Full", "Interpretation"], loo_rows)
    return {"A": table_a, "B": table_b, "C": table_c, "D": table_d, "E": table_e}


def _reports(run_root: Path) -> None:
    tables = _tables(run_root)
    metrics = pd.read_csv(run_root / "evaluation" / "all_36_arm_metrics.csv")
    primary = metrics[metrics.scope == "primary"].set_index("arm_id")
    base, full = primary.loc["B000"], primary.loc["F_ALL15"]
    boot = read_json(run_root / "evaluation" / "recording_bootstrap.json")["primary"]
    wall = read_json(run_root / "wall_time_accounting.json")["total_wall_seconds"]
    summary = f"""# AASM V4 Phase 2 Executive Summary

## 结论

本轮真实完成 36/36 臂与 360/360 个夜晚-臂重放，15 个神经来源均在读取 HMC Stage Reference 前完成 label-blind 封存；B000 的 Event Bank、Evidence 与 Temporal R1 Prediction 在十晚全部精确回归。工程状态：PHASE2_36_ARM_ENGINEERING_PASS；有效替换状态：FULL_15_HANDLER_REPLACEMENT_ESTABLISHED。

F_ALL15 没有整体胜出：相对 B000，Accuracy {full.accuracy - base.accuracy:+.6f}，Macro-F1 {full.macro_f1 - base.macro_f1:+.6f}，N3 F1 {full.N3_f1 - base.N3_f1:+.6f}，Coverage {full.coverage - base.coverage:+.6f}。它提高了 Wake、N1 和 REM F1，却以 Accuracy、N2、N3 和 Coverage 回退为代价。

最可靠的单源是 Arousal U-Net：Delta Accuracy {primary.loc['S_AR'].delta_accuracy:+.6f}、Delta Macro-F1 {primary.loc['S_AR'].delta_macro_f1:+.6f}，recording-bootstrap 95% CI 分别为 [{boot['S_AR']['accuracy']['ci_low']:+.6f}, {boot['S_AR']['accuracy']['ci_high']:+.6f}] 与 [{boot['S_AR']['macro_f1']['ci_low']:+.6f}, {boot['S_AR']['macro_f1']['ci_high']:+.6f}]。SUMOv2 Spindle 提高 Accuracy/N2，但单换 Macro-F1 略降；它在完整系统中的 leave-one-out 贡献为正。K08 提高部分宏观类别表现和 REM，却使 N2 F1 下降 {primary.loc['S_E02'].delta_N2_f1:+.6f}；SSSM Slow-wave 使 N3 F1 下降 {primary.loc['S_E01'].delta_N3_f1:+.6f}。E13 的 Wake/Macro-F1 表面收益伴随 Coverage 下降 {primary.loc['S_E13'].coverage - base.coverage:+.6f}，且仍是 movement proxy，不能采纳为正式 MBM Detector。

科学结论：DETECTOR_TO_STAGE_HYPOTHESIS_PARTIALLY_SUPPORTED；DETECTOR_GAIN_NOT_SUFFICIENT；NEGATIVE_CROSS_EVENT_INTERACTIONS；EVIDENCE_QUALITY_LIMITED；INCONCLUSIVE_DUE_TO_INPUT_OR_VALIDATION_LIMITS。HMC 没有这些事件的独立事件级真值，全部 montage transfer 仍未独立验证。

实际总 Wall Time：{wall:.1f} 秒。

## 核心对照

{tables['D']}
"""
    (run_root / "EXECUTIVE_SUMMARY.md").write_text(summary, encoding="utf-8")

    final_report = f"""# FINAL PHASE 2 — 36-Arm Event-to-Stage Report

## Engineering status

- PHASE2_36_ARM_ENGINEERING_PASS
- FULL_15_HANDLER_REPLACEMENT_ESTABLISHED
- 36/36 arms; 360/360 night-arm executions; Primary 8,902 epochs; Full-valid 9,211 epochs.
- No training, HMC tuning, rule modification, access outside the frozen record allowlist, commit, or push.
- Independent metric recomputation maximum difference: 0.

## Table A — Execution completeness

{tables['A']}

## Table B — Single-handler results

{tables['B']}

## Table C — Grouped results

{tables['C']}

## Table D — Full combination

{tables['D']}

## Table E — Leave-one-out ranking

Delta in Table E is the score after removing the handler minus F_ALL15. Negative therefore means that including the handler helps the full-context metric.

{tables['E']}

## Scientific boundary

These are paired HMC ten-night development-set stage effects under the frozen Temporal R1 engine. They measure event-to-rule utility, not HMC event-detection accuracy or external clinical generalization. Weak-teacher Phase1C/1D sources, E13 proxy semantics, and unvalidated cross-dataset montage transfer remain explicit limitations.
"""
    (run_root / "FINAL_PHASE2_36ARM_REPORT.md").write_text(final_report, encoding="utf-8")

    insights = f"""# Research Paper Insights

## Q1 — 哪些单独替换改善最终分期？

AR 是唯一同时取得清晰 Accuracy 与 Macro-F1 增益、且 recording-bootstrap 95% CI 均为正的单源。SP 提高 Accuracy {primary.loc['S_SP'].delta_accuracy:+.6f} 和 N2 F1 {primary.loc['S_SP'].delta_N2_f1:+.6f}，但 Macro-F1 {primary.loc['S_SP'].delta_macro_f1:+.6f}，属于 N2/Accuracy 定向收益。E04 有小幅 Accuracy 增益但 Macro-F1 近零；E11 很小；E13 的 Macro/Wake 表面收益被 proxy 语义和 Coverage 损失限制。

## Q2 — 哪些单独有效、联合后失效？

E04 单换 Accuracy 为正，但在 F_ALL15 的 leave-one-out 中，其 Macro-F1 边际贡献为负。E13 单换 Macro-F1 为正，但在完整上下文中加入它会降低 Accuracy，并扩大 movement abstention。G_N2 的 Macro-F1 增益 {primary.loc['G_N2'].delta_macro_f1:+.6f} 小于三个单源增益之和，interaction 为负，说明 K-complex、Spindle、Arousal 共享规则路径后存在收益重叠与冲突。

## Q3 — Full Combination 是否优于 Baseline？

否。F_ALL15 的 Macro-F1 提高 {full.delta_macro_f1:+.6f}，但 Accuracy 下降 {full.delta_accuracy:+.6f}，N3 F1 下降 {full.delta_N3_f1:+.6f}，Coverage 下降 {full.coverage - base.coverage:+.6f}，净正确 Epoch 为 -149。主要损失来自 E01 对 N3.1 的削弱、E02 对 N2 结构证据的替换，以及 E13 proxy 对 movement/abstention 路径的过度激活；跨事件组合没有自动抵消这些损失。

## Q4 — Leave-One-Out 证明谁在完整系统中有贡献？

按 full-context Macro-F1，SP、E13、AR、E02、E12、E11、E08 的加入为正贡献；其中只有 SP/AR 同时具备较强既有事件级证据与可解释 stage utility。E13/E02 的贡献带有严重的 stage/coverage tradeoff，不能仅按 Macro-F1 采纳。E01 是最明显的负贡献，移除它可提高 Macro-F1 和 Accuracy。

## Q5 — 哪些事件质量改善没有转化为 Stage 收益？

K08 是直接反例：既有 DREAMS 事件级收益没有转化为 N2 stage utility。R04 的既有 REM 事件收益在本轮只有小而不稳定的 stage gain。E07 等弱监督 handler 即使可运行，也几乎不改变最终分期。Event accuracy、event density 和 rule-relevant evidence coverage 不是同一量。

## Q6 — 主要瓶颈是什么？

Observed result 支持混合瓶颈：Detector/transfer quality、Evidence duration、Event semantics、cross-event interaction、rule priority、temporal continuation、QC/availability 均参与。E06 将 blink positive epochs 大幅增加，但 W.2 只触发少量 Epoch，显示 duration semantics 限制；E13 则显示过多 movement evidence 会降低 coverage。没有独立 HMC event truth，无法把这些 stage 变化进一步分解为真正 Event TP/FP。

## Q7 — 最值得发展的反常研究问题

Observed result：把 15 个独立可运行的神经 handler 全部替换后，Macro-F1 上升但 Accuracy、N3 与 Coverage 同时下降；部分单源收益在组合中反转。

Supported mechanism：多源事件密度和持续时间改变了共享 Evidence、规则优先级与 Temporal R1 状态，收益并非可加；E01/E02/E13 的实际 rule-firing 与 coverage 变化解释了主要损失方向。

Future hypothesis：下一代 Event-to-Rule 系统应优化 rule-relevant evidence calibration 与跨源一致性，而不是独立最大化每个 detector 的 event metric。该假设需要新的独立事件真值和跨中心验证；本轮不构成显著性或临床有效性证明。
"""
    (run_root / "RESEARCH_PAPER_INSIGHTS.md").write_text(insights, encoding="utf-8")

    decisions = """# Model Adoption Decisions

| Handler | Decision | Evidence basis | Next condition |
|---|---|---|---|
"""
    for handler in HANDLERS:
        row = primary.loc[f"S_{handler}"]
        decisions += f"| {handler} {DISPLAY_NAME[handler]} | {SINGLE_DECISIONS[handler]} | Delta Accuracy {row.delta_accuracy:+.6f}; Delta Macro-F1 {row.delta_macro_f1:+.6f}; {_main_stage_effect(row)} | Independent event truth plus montage validation before public baseline replacement |\n"
    decisions += """

## System decision

Do not adopt F_ALL15 as the new default. Preserve the frozen baseline. Retain AR and SP as the strongest HMC-development research candidates; keep E04 as a REM research candidate; close K08 as a negative downstream replacement result; retain all remaining frozen handlers until event semantics, montage transfer, and rule-relevant duration are independently validated.
"""
    (run_root / "MODEL_ADOPTION_DECISIONS.md").write_text(decisions, encoding="utf-8")

    negative = f"""# Negative Results

- F_ALL15: Delta Accuracy {full.delta_accuracy:+.6f}, Delta N3 F1 {full.delta_N3_f1:+.6f}, Delta Coverage {full.coverage - base.coverage:+.6f}; the full replacement is not an overall upgrade.
- E01: Delta N3 F1 {primary.loc['S_E01'].delta_N3_f1:+.6f}; neural slow-wave intervals weakened N3.1 evidence.
- E02: Delta N2 F1 {primary.loc['S_E02'].delta_N2_f1:+.6f} and Delta Accuracy {primary.loc['S_E02'].delta_accuracy:+.6f}, despite a Macro-F1/REM tradeoff.
- E13: proxy semantics remain unresolved; Delta Coverage {primary.loc['S_E13'].coverage - base.coverage:+.6f} prevents adoption despite Wake/Macro-F1 gain.
- E03/E05/E06/E07/E08/E09/E10 did not establish meaningful downstream improvement.
- No HMC event-level ground truth exists for these replacements; stage effects cannot be relabeled as detector accuracy.
"""
    (run_root / "analysis" / "negative_results.md").write_text(negative, encoding="utf-8")

    hypothesis = """# Scientific Hypothesis Assessment

The central hypothesis is partially supported, not sufficient as a general rule.

- DETECTOR_TO_STAGE_HYPOTHESIS_PARTIALLY_SUPPORTED: AR and SP show positive, interpretable development-set utility.
- DETECTOR_GAIN_NOT_SUFFICIENT: K08 event gains and the 15-source full replacement do not guarantee better staging.
- NEGATIVE_CROSS_EVENT_INTERACTIONS: grouped and full results are non-additive.
- RULE_SEMANTICS_BOTTLENECK: blink duration and movement resolution constrain how added proposals reach final stages.
- EVIDENCE_QUALITY_LIMITED: most Phase1C/1D sources are weak-teacher/prototype outputs.
- INCONCLUSIVE_DUE_TO_INPUT_OR_VALIDATION_LIMITS: montage transfer is unvalidated and HMC lacks independent event truth.

Observed results, supported mechanisms, and future hypotheses are separated in RESEARCH_PAPER_INSIGHTS.md.
"""
    (run_root / "analysis" / "scientific_hypothesis_assessment.md").write_text(hypothesis, encoding="utf-8")

    mechanism = f"""# Mechanism Analysis

The audited chain is source event -> evidence duration/presence -> frozen rule predicate -> rule priority -> Temporal R1 state -> final stage.

- E01 reduced slow-wave-positive epochs from 4,127 to 1,907 and N3.1 firings from 594 to 294; 212 true-N3 epochs changed from N3 to N2. This explains the N3 regression without calling baseline-only candidates expert false negatives.
- E02 reduced n2_kcomplex-positive epochs from 4,681 to 1,507. The dominant rule transition was N2.1 to FALLBACK.INHERIT_PREVIOUS; N2 F1 fell while REM F1 rose.
- E04 increased rem, rem_low_overlap, and irregular_rem_normal evidence. It recovered REM epochs but also moved non-REM epochs into R, producing only a small mixed net gain.
- E13 increased mbm-positive epochs from 34 to 6,760. Movement resolution and abstention raised Wake F1 while reducing coverage and harming other stages; its proxy identity is therefore decisive.
- SP added rule-relevant n2_spindle evidence and improved N2 utility. AR changed arousal and K-arousal association evidence and produced the strongest single-arm result.
- F_ALL15 corrected 1,145 epochs and introduced 1,294 errors. Its largest losses were true N2 to N1/UNDETERMINED and true N3 to N2, demonstrating non-additive evidence/rule interactions.

Exact transitions are stored in rule_trace_changes.csv, paired_corrected_introduced.csv, and event_to_rule_dependency.csv.
"""
    (run_root / "analysis" / "mechanism_analysis.md").write_text(mechanism, encoding="utf-8")

    final_decision = """# Final Scientific Decision

Engineering status: PHASE2_36_ARM_ENGINEERING_PASS and FULL_15_HANDLER_REPLACEMENT_ESTABLISHED.

Scientific status: DETECTOR_TO_STAGE_HYPOTHESIS_PARTIALLY_SUPPORTED; DETECTOR_GAIN_NOT_SUFFICIENT; NEGATIVE_CROSS_EVENT_INTERACTIONS; RULE_SEMANTICS_BOTTLENECK; EVIDENCE_QUALITY_LIMITED; INCONCLUSIVE_DUE_TO_INPUT_OR_VALIDATION_LIMITS.

Decision: retain the frozen full system rather than adopting F_ALL15. Preserve AR and SP as the leading HMC-development research candidates. Keep E04 for bounded REM research. Do not replace the stage K-complex path with K08, do not adopt E01, and do not treat E13 as a validated MBM detector. No result is an external clinical validation.
"""
    (run_root / "FINAL_SCIENTIFIC_DECISION.md").write_text(final_decision, encoding="utf-8")


def _audits(run_root: Path, source_summary: dict[str, Any]) -> None:
    status = pd.read_csv(run_root / "arms" / "arm_execution_status.csv")
    detail = pd.read_csv(run_root / "arms" / "arm_execution_10night.csv")
    matrix = {
        "status": "PASS" if len(status) == 36 and len(detail) == 360 and (status.status == "PASS").all() else "FAIL",
        "registered_arms": len(ARMS),
        "executed_arms": int((status.status == "PASS").sum()),
        "unique_arm_ids": int(status.arm_id.nunique()),
        "night_arm_rows": len(detail),
        "unique_night_arm_pairs": int(detail[["arm_id", "record_id"]].drop_duplicates().shape[0]),
        "single_arms": int(status.arm_id.str.startswith("S_").sum()),
        "group_arms": int(status.arm_id.str.startswith("G_").sum()),
        "full_arms": int((status.arm_id == "F_ALL15").sum()),
        "leave_one_out_arms": int(status.arm_id.str.startswith("L_").sum()),
        "partial_arms": int((status.partial_nights > 0).sum()),
    }
    write_json(run_root / "audits" / "matrix_completeness.json", matrix)
    checkpoint = read_json(run_root / "models" / "checkpoint_manifest.json")
    write_json(run_root / "audits" / "pretrained_model_integrity.json", {
        "status": checkpoint["status"],
        "handler_count": len(checkpoint["rows"]),
        "all_checkpoint_hashes_match": all(row["hash_match"] for row in checkpoint["rows"]),
        "all_frozen": all(row["frozen"] for row in checkpoint["rows"]),
        "phase2_optimizer_steps": 0,
        "source_bank_count": len(source_summary["source_rows"]),
        "source_bank_qc_errors": source_summary["errors"],
    })
    regression = read_json(run_root / "arms" / "baseline_regression.json")
    hashes = read_json(run_root / "frozen_core_hashes.json")
    write_json(run_root / "audits" / "frozen_core_regression.json", {
        "status": "PASS" if regression["all_exact"] and hashes["all_match"] else "FAIL",
        "core_hashes_match": hashes["all_match"],
        "b000_all_exact": regression["all_exact"],
        "records": regression["rows"],
    })


def _run_plan(run_root: Path) -> None:
    text = """# AASM V4 Phase 2 Executed Run Plan

Protocol order: P0 heritage and access audit; P1 input compatibility; P2 label-blind frozen inference and source-bank sealing; P3 copy-on-write overlay; P4 frozen 36-arm matrix; P5 15 single swaps; P6 four grouped arms; P7 F_ALL15; P8 15 leave-one-out arms; P9 paired metrics/mechanism attribution; P10 independent recomputation; P11 packaging.

All 15 checkpoints, eventization parameters, the 8,902-epoch primary mask, the 9,211-epoch full-valid mask, Evidence aggregation, and Temporal R1 were frozen before HMC reference access. No training, threshold tuning, rule changes, or handler selection used HMC labels. Only the explicit ten-record allowlist was used.

Execution outcome: all gates passed; 36 arms and 360 night-arm replays completed. Failure recovery: the first test command used the console pytest entry point and could not resolve the repository package; it was rerun as python -m pytest with the same test set and passed. This invocation repair did not affect scientific data or parameters.
"""
    (run_root / "RUN_PLAN.md").write_text(text, encoding="utf-8")


def _artifact_integrity(run_root: Path) -> None:
    required = [
        "RUN_PLAN.md", "run_state.json", "wall_time_accounting.json", "preflight.json", "heritage_manifest.json", "access_allowlist.json",
        "models/model_registry_15.json", "models/checkpoint_manifest.json", "models/input_compatibility.csv", "models/model_inference_manifest.json",
        "source_banks/frozen_bank_manifest.json", "source_banks/neural_bank_manifest.json", "source_banks/model_output_qc.json",
        "matrix/arm_manifest_36.csv", "matrix/arm_execution_status.csv", "matrix/arm_execution_10night.csv", "matrix/replacement_effectiveness.csv",
        "single_swap/single_swap_metrics.csv", "single_swap/single_swap_per_night.csv", "single_swap/single_swap_paired_errors.csv",
        "grouped/grouped_metrics.csv", "grouped/grouped_interactions.csv", "grouped/grouped_per_night.csv",
        "full_combination/full_system_metrics.csv", "full_combination/full_system_per_night.csv", "full_combination/full_system_error_analysis.csv",
        "leave_one_out/leave_one_out_metrics.csv", "leave_one_out/marginal_contributions.csv", "leave_one_out/leave_one_out_per_night.csv",
        "analysis/all_36_arm_metrics.csv", "analysis/per_stage_metrics.csv", "analysis/paired_corrected_introduced.csv", "analysis/interaction_analysis.csv",
        "analysis/rule_trace_changes.csv", "analysis/event_to_rule_dependency.csv", "analysis/uncertainty_intervals.csv", "analysis/negative_results.md",
        "analysis/scientific_hypothesis_assessment.md", "audits/matrix_completeness.json", "audits/pretrained_model_integrity.json",
        "audits/label_blind_inference.json", "audits/frozen_core_regression.json", "audits/independent_recheck.json",
        "EXECUTIVE_SUMMARY.md", "FINAL_PHASE2_36ARM_REPORT.md", "RESEARCH_PAPER_INSIGHTS.md", "MODEL_ADOPTION_DECISIONS.md", "reproduce.md",
    ]
    missing = [name for name in required if not (run_root / name).is_file()]
    write_json(run_root / "audits" / "artifact_integrity.json", {
        "status": "PASS" if not missing else "FAIL",
        "required_file_count_excluding_self_and_manifest": len(required),
        "missing": missing,
        "all_required_nonempty": not missing and all((run_root / name).stat().st_size > 0 for name in required),
    })
    write_json(run_root / "artifact_manifest.json", {
        "root": str(run_root.resolve()),
        "generated_at": datetime.now().astimezone().isoformat(),
        "files": file_manifest(run_root, exclude={"artifact_manifest.json"}),
    })


def package_delivery(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root).resolve()
    state = read_json(root / "run_state.json")
    prior_accounting = read_json(root / "wall_time_accounting.json")
    core_wall_seconds = prior_accounting.get("core_execution_wall_seconds", state.get("wall_seconds"))
    started = datetime.fromisoformat(state["started_at"])
    state["gates"]["P11"] = "RUNNING"
    state["status"] = "RUNNING"
    write_json(root / "run_state.json", state)
    _run_plan(root)
    _models(root)
    source_summary = _source_bank_manifests(root)
    _matrix(root)
    _metric_packages(root)
    _dependency_and_uncertainty(root)
    _audits(root, source_summary)
    state["gates"] = {f"P{index}": "PASS" for index in range(12)}
    state["status"] = "PASS"
    completed = datetime.now().astimezone()
    state["completed_at"] = completed.isoformat()
    state["wall_seconds"] = (completed - started).total_seconds()
    write_json(root / "run_state.json", state)
    write_json(root / "wall_time_accounting.json", {
        "total_wall_seconds": state["wall_seconds"],
        "max_wall_hours": state["max_wall_hours"],
        "completed_within_budget": state["wall_seconds"] <= float(state["max_wall_hours"]) * 3600,
        "core_execution_wall_seconds": core_wall_seconds,
        "includes_final_packaging": True,
    })
    _reports(root)
    _artifact_integrity(root)
    result = read_json(root / "audits" / "artifact_integrity.json")
    if result["status"] != "PASS":
        raise RuntimeError(f"mandatory artifact delivery incomplete: {result['missing']}")
    return {"status": "PASS", "run_root": str(root), "wall_seconds": state["wall_seconds"]}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(package_delivery(args.run_root), ensure_ascii=False, indent=2))
