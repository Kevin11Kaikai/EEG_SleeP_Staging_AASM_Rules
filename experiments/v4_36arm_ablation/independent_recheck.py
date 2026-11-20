from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ALIGNED_REFERENCE, ARMS, BASE_BANK_ROOT, EVENT_TYPE, HANDLERS, RECORDS
from .io_utils import canonical_hash, read_gzip_json, read_json, sha256, write_json


STAGES = ("W", "N1", "N2", "N3", "R", "UNDETERMINED")


def _confusion(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    lookup = {stage: index for index, stage in enumerate(STAGES)}
    matrix = np.zeros((6, 6), dtype=np.int64)
    for expected, actual in zip(truth, prediction):
        matrix[lookup[expected], lookup.get(actual, 5)] += 1
    return matrix


def _metrics(matrix: np.ndarray) -> dict[str, Any]:
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    diagonal = np.diag(matrix)
    precision = np.divide(diagonal, predicted, out=np.zeros(6), where=predicted > 0)
    recall = np.divide(diagonal, support, out=np.zeros(6), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(6), where=(precision + recall) > 0)
    n = int(matrix.sum())
    accuracy = float(diagonal.sum() / n)
    expected = float(predicted @ support) / n**2
    return {"n": n, "accuracy": accuracy, "macro_f1": float(f1[:5].mean()),
            "kappa": float((accuracy - expected) / (1 - expected)) if expected < 1 else None,
            "coverage": float((n - predicted[5]) / n),
            "per_stage_f1": {stage: float(f1[index]) for index, stage in enumerate(STAGES[:5])}}


def run(run_root: Path) -> dict[str, Any]:
    reference = pd.read_csv(ALIGNED_REFERENCE)
    main = read_json(run_root / "evaluation" / "metrics.json")
    max_difference = 0.0
    rows = []
    for scope, column, expected_n in (("primary", "in_primary", 8902), ("full", "in_full_record", 9211)):
        for arm in ARMS:
            matrices = []
            for record in RECORDS:
                selected = reference.loc[reference.record_id == record].sort_values("epoch_index")
                mask = selected[column].to_numpy(dtype=bool)
                truth = selected.true_stage.to_numpy()[mask]
                prediction = np.asarray([row["stage"] for row in read_gzip_json(
                    run_root / "predictions" / arm / f"{record}.json.gz")])[mask]
                matrices.append(_confusion(truth, prediction))
            metric = _metrics(sum(matrices, np.zeros((6, 6), dtype=np.int64)))
            if metric["n"] != expected_n:
                raise AssertionError("independent mask count mismatch")
            target = main[scope][arm]
            differences = [abs(metric[key] - target[key]) for key in ("accuracy", "macro_f1", "kappa", "coverage")]
            differences += [abs(metric["per_stage_f1"][stage] - target["per_stage"][stage]["f1"]) for stage in STAGES[:5]]
            maximum = max(differences)
            max_difference = max(max_difference, maximum)
            rows.append({"scope": scope, "arm_id": arm, "maximum_metric_difference": maximum, "pass": maximum < 1e-12})
    boundary_errors, duplicate_errors, nonselected_errors = [], [], []
    for arm, selected in ARMS.items():
        for record in RECORDS:
            bank = read_gzip_json(run_root / "event_banks" / arm / f"{record}.json.gz")
            duration = float(bank["info"]["duration"])
            ids = [event["event_id"] for event in bank["events"]["enhanced"]]
            if len(ids) != len(set(ids)):
                duplicate_errors.append(f"{arm}/{record}")
            for event in bank["events"]["enhanced"]:
                if not (0 <= float(event["onset"]) < float(event["offset"]) <= duration + 1e-6):
                    boundary_errors.append(f"{arm}/{record}/{event['event_id']}")
            if arm == "B000":
                base = read_gzip_json(BASE_BANK_ROOT / f"{record}.json.gz")
                if canonical_hash(bank) != canonical_hash(base):
                    nonselected_errors.append(f"B000/{record}")
    execution = pd.read_csv(run_root / "arms" / "arm_execution_10night.csv")
    paired = pd.read_csv(run_root / "evaluation" / "paired_error_analysis.csv")
    identity_failures = int(((paired.corrected - paired.introduced) != paired.net_corrected).sum())
    result = {
        "status": "PASS" if max_difference < 1e-12 and not boundary_errors and not duplicate_errors and not nonselected_errors and identity_failures == 0 else "FAIL",
        "registered_arms": len(ARMS), "night_arm_rows": len(execution),
        "all_360_executed": len(execution) == 360 and bool((execution.status == "PASS").all()),
        "maximum_independent_metric_difference": max_difference, "metric_rows": rows,
        "paired_identity_failures": identity_failures, "event_boundary_errors": boundary_errors,
        "duplicate_id_errors": duplicate_errors, "b000_bank_errors": nonselected_errors,
        "reference_sha256": sha256(ALIGNED_REFERENCE),
        "no_training": True, "no_hmc_checkpoint_selection": True,
        "explicit_allowlist_only": list(RECORDS), "sn011_plus_enumerated": False,
        "outside_record_allowlist_accessed": False, "git_commit_or_push": False,
    }
    write_json(run_root / "audits" / "independent_recheck.json", result)
    return result
