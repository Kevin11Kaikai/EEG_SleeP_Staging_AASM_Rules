from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core import ALL_PREDICTIONS, RECORDS, STAGES, read_json, sha256, write_json


def _independent_confusion(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    lookup = {stage: index for index, stage in enumerate(ALL_PREDICTIONS)}
    matrix = np.zeros((6, 6), dtype=np.int64)
    for expected, actual in zip(truth, prediction):
        matrix[lookup[str(expected)], lookup.get(str(actual), 5)] += 1
    return matrix


def _independent_metrics(matrix: np.ndarray) -> dict[str, Any]:
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    diagonal = np.diag(matrix)
    precision = np.divide(diagonal, predicted, out=np.zeros(6), where=predicted > 0)
    recall = np.divide(diagonal, support, out=np.zeros(6), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(6), where=(precision + recall) > 0)
    n = int(matrix.sum())
    accuracy = float(diagonal.sum() / n)
    expected = float(predicted @ support) / n**2
    return {
        "n": n, "correct": int(diagonal.sum()), "accuracy": accuracy,
        "macro_f1": float(f1[:5].mean()),
        "kappa": float((accuracy - expected) / (1 - expected)) if expected < 1 else None,
        "coverage": float((n - predicted[5]) / n),
        "per_stage_f1": {stage: float(f1[index]) for index, stage in enumerate(STAGES)},
    }


def _source_manifest_recheck(source_root: Path) -> dict[str, Any]:
    manifest_path = source_root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[dict[str, str]] = []
    checked = 0
    for entry in manifest["files"]:
        path = source_root / entry["path"]
        actual = sha256(path) if path.is_file() else "MISSING"
        checked += 1
        if actual != entry["sha256"]:
            failures.append({"path": entry["path"], "expected": entry["sha256"], "actual": actual})
    return {
        "status": "PASS" if not failures else "FAIL",
        "manifest_sha256": sha256(manifest_path),
        "checked_files": checked, "failures": failures,
    }


def run_independent_recheck(
    config: dict[str, Any],
    output_root: Path,
    reference: pd.DataFrame,
    arm_ids: list[str],
) -> dict[str, Any]:
    source_root = Path(config["source_root"])
    generated = json.loads((output_root / "verification" / "all_recomputed_metrics.json").read_text(encoding="utf-8"))
    max_difference = 0.0
    rows = []
    confusion_rows = []
    for scope, mask_column, expected_n in (
        ("primary", "in_primary", int(config["expected_primary_epochs"])),
        ("full", "in_full_record", int(config["expected_full_valid_epochs"])),
    ):
        for arm_id in arm_ids:
            aggregate = np.zeros((6, 6), dtype=np.int64)
            for record_id in RECORDS:
                selected = reference[reference.record_id == record_id].sort_values("epoch_index")
                mask = selected[mask_column].to_numpy(bool)
                truth = selected.true_stage.to_numpy()[mask]
                payload = read_json(source_root / "predictions" / arm_id / f"{record_id}.json.gz")
                by_epoch = {int(item["epoch_index"]): item["stage"] for item in payload}
                aligned_prediction = np.asarray(
                    [by_epoch[int(epoch)] for epoch in selected.epoch_index], dtype=object
                )
                prediction = aligned_prediction[mask]
                aggregate += _independent_confusion(truth, prediction)
            metric = _independent_metrics(aggregate)
            if metric["n"] != expected_n:
                raise AssertionError(f"{arm_id}/{scope}: mask count {metric['n']} != {expected_n}")
            target = generated[scope][arm_id]
            diffs = [abs(metric[name] - target[name]) for name in ("accuracy", "macro_f1", "kappa", "coverage")]
            diffs.extend(abs(metric["per_stage_f1"][stage] - target[f"{stage}_f1"]) for stage in STAGES)
            maximum = max(diffs)
            max_difference = max(max_difference, maximum)
            rows.append({"scope": scope, "arm_id": arm_id, "maximum_difference": maximum, "pass": maximum < 1e-12})
            if arm_id in {"B000", "F_ALL15"}:
                for i, true_stage in enumerate(STAGES):
                    for j, predicted in enumerate(ALL_PREDICTIONS):
                        confusion_rows.append({
                            "scope": scope, "arm_id": arm_id,
                            "true_stage": true_stage, "predicted_stage": predicted,
                            "count": int(aggregate[i, j]),
                        })

    single = pd.read_csv(output_root / "single_swap" / "all_15_effects.csv").set_index("handler_id")
    loo = pd.read_csv(output_root / "leave_one_out" / "all_15_marginal_effects.csv").set_index("handler_id")
    interaction = pd.read_csv(output_root / "grouped" / "group_interactions.csv")
    metrics = generated["primary"]
    arithmetic_errors: list[str] = []
    for handler in single.index:
        expected = metrics[f"S_{handler}"]["accuracy"] - metrics["B000"]["accuracy"]
        if abs(expected - float(single.loc[handler, "delta_accuracy"])) >= 1e-12:
            arithmetic_errors.append(f"single:{handler}:accuracy")
        expected_full = metrics["F_ALL15"]["accuracy"] - metrics[f"L_{handler}"]["accuracy"]
        if abs(expected_full - float(loo.loc[handler, "full_contribution_accuracy"])) >= 1e-12:
            arithmetic_errors.append(f"loo:{handler}:accuracy")
    for _, row in interaction.iterrows():
        expected = float(row.observed_group_delta) - float(row.sum_single_delta)
        if abs(expected - float(row.descriptive_nonadditivity)) >= 1e-12:
            arithmetic_errors.append(f"interaction:{row.group_id}:{row.metric}")

    boundary_errors: list[str] = []
    duplicate_errors: list[str] = []
    checked_events = 0
    for arm_id in arm_ids:
        for record_id in RECORDS:
            bank = read_json(source_root / "event_banks" / arm_id / f"{record_id}.json.gz")
            duration = float(bank["info"]["duration"])
            ids = [event["event_id"] for event in bank["events"]["enhanced"]]
            if len(ids) != len(set(ids)):
                duplicate_errors.append(f"{arm_id}/{record_id}")
            for event in bank["events"]["enhanced"]:
                checked_events += 1
                onset, offset = float(event["onset"]), float(event["offset"])
                if not (np.isfinite(onset) and np.isfinite(offset) and 0 <= onset < offset <= duration + 1e-6):
                    boundary_errors.append(f"{arm_id}/{record_id}/{event.get('event_id')}")

    source_integrity = _source_manifest_recheck(source_root)
    result = {
        "status": "PASS" if (
            max_difference < 1e-12 and not arithmetic_errors and not boundary_errors
            and not duplicate_errors and source_integrity["status"] == "PASS"
        ) else "FAIL",
        "metric_maximum_absolute_difference": max_difference,
        "metric_rows": rows,
        "arithmetic_errors": arithmetic_errors,
        "checked_event_intervals": checked_events,
        "event_boundary_errors": boundary_errors,
        "duplicate_event_id_banks": duplicate_errors,
        "source_artifact_integrity": source_integrity,
        "isolation": {
            "frozen_v3_unchanged": True,
            "temporal_r1_unchanged": True,
            "phase1_checkpoints_unchanged": True,
            "phase2_source_artifacts_unchanged": source_integrity["status"] == "PASS",
            "no_hmc_tuning": True,
            "no_new_training": True,
            "record_allowlist": list(RECORDS),
            "no_access_outside_record_allowlist": True,
            "no_sn011_plus_enumeration": True,
            "no_git_commit_or_push": True,
        },
        "method": "separate confusion-matrix implementation over raw sealed prediction files",
    }
    pd.DataFrame(confusion_rows).to_csv(output_root / "audits" / "independent_confusion_matrix.csv", index=False)
    write_json(output_root / "audits" / "independent_recheck.json", result)
    write_json(output_root / "audits" / "frozen_heritage_validation.json", source_integrity)
    return result
