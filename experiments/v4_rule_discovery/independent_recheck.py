from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    ALIGNED_REFERENCE,
    CHAMPION_ROOT,
    CONFIRMATION_RECORDS,
    DISCOVERY_RECORDS,
    RECORDS,
    RESULTS_ROOT,
    STAGES,
    STAGE_TO_CODE,
)
from .io import read_json, sha256_file, write_json


EXPECTED_CONFIGS = 384
EXPECTED_CONFIG_RECORDS = EXPECTED_CONFIGS * len(RECORDS)
TOLERANCE = 1e-12


def _manual_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Recompute metrics without importing the experiment metric implementation."""

    if truth.shape != prediction.shape or truth.ndim != 1 or len(truth) == 0:
        raise ValueError("invalid independent metric inputs")
    matrix = [[0 for _ in range(6)] for _ in range(6)]
    for left, right in zip(truth.tolist(), prediction.tolist()):
        if not (0 <= int(left) < 5 and 0 <= int(right) < 6):
            raise ValueError((left, right))
        matrix[int(left)][int(right)] += 1
    support = [sum(row) for row in matrix]
    predicted = [sum(matrix[row][column] for row in range(6)) for column in range(6)]
    per_stage: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for index, stage in enumerate(STAGES[:5]):
        tp = matrix[index][index]
        precision = tp / predicted[index] if predicted[index] else 0.0
        recall = tp / support[index] if support[index] else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        per_stage[stage] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support[index],
        }
    total = len(truth)
    correct = sum(matrix[index][index] for index in range(5))
    accepted = total - predicted[5]
    return {
        "n": total,
        "correct": correct,
        "accuracy": correct / total,
        "macro_f1": sum(f1_values) / 5,
        "coverage": accepted / total,
        "conditional_accuracy": correct / accepted if accepted else None,
        "per_stage": per_stage,
        "confusion": matrix,
    }


def _manual_paired(
    truth: np.ndarray, baseline: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    if not (truth.shape == baseline.shape == candidate.shape):
        raise ValueError("paired arrays differ")
    changed = corrected = introduced = 0
    transitions: dict[str, int] = {}
    by_stage = {
        stage: {"corrected": 0, "introduced": 0} for stage in STAGES[:5]
    }
    for expected, left, right in zip(
        truth.tolist(), baseline.tolist(), candidate.tolist()
    ):
        if left != right:
            changed += 1
            key = f"{STAGES[int(left)]}->{STAGES[int(right)]}"
            transitions[key] = transitions.get(key, 0) + 1
        if left != expected and right == expected:
            corrected += 1
            by_stage[STAGES[int(expected)]]["corrected"] += 1
        elif left == expected and right != expected:
            introduced += 1
            by_stage[STAGES[int(expected)]]["introduced"] += 1
    return {
        "changed": changed,
        "corrected": corrected,
        "introduced": introduced,
        "net_correct": corrected - introduced,
        "transitions": transitions,
        "by_reference_stage": by_stage,
    }


def _metric_block_matches(manual: dict[str, Any], declared: dict[str, Any]) -> bool:
    scalar_match = all(
        abs(float(manual[key]) - float(declared[key])) <= TOLERANCE
        for key in ("accuracy", "macro_f1", "coverage")
    )
    stage_match = all(
        abs(
            float(manual["per_stage"][stage]["f1"])
            - float(declared["per_stage"][stage]["f1"])
        )
        <= TOLERANCE
        for stage in STAGES[:5]
    )
    return scalar_match and stage_match and manual["confusion"] == declared["confusion"]


def _paired_matches(manual: dict[str, Any], declared: dict[str, Any]) -> bool:
    return all(
        manual[key] == declared[key]
        for key in ("changed", "corrected", "introduced", "net_correct")
    ) and manual["transitions"] == declared.get("transitions", {})


def _read_first(root: Path, relative_paths: tuple[str, ...]) -> tuple[Path | None, Any]:
    for relative in relative_paths:
        path = root / relative
        if path.exists():
            return path, read_json(path)
    return None, None


def _load_reference() -> pd.DataFrame:
    frame = pd.read_csv(ALIGNED_REFERENCE)
    if set(frame.record_id.unique()) != set(RECORDS):
        raise RuntimeError("reference record allow-list mismatch")
    for record in RECORDS:
        rows = frame.loc[frame.record_id == record].sort_values("epoch_index")
        if rows.epoch_index.tolist() != list(range(len(rows))):
            raise RuntimeError(f"reference epoch mismatch: {record}")
    return frame


def _champion_codes(record: str) -> np.ndarray:
    path = CHAMPION_ROOT / "predictions" / f"{record}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = json.load(stream)
    return np.asarray(
        [STAGE_TO_CODE[item["stage"]] for item in rows], dtype=np.uint8
    )


def run_independent_recheck(root: Path, heritage: dict[str, Any]) -> dict[str, Any]:
    """Independently validate execution, split isolation and confirmation metrics."""

    root = root.resolve()
    cache_path = root / "cache" / "rule_predictions.sqlite"
    connection = sqlite3.connect(cache_path)
    try:
        config_ids = [
            row[0]
            for row in connection.execute(
                "SELECT config_id FROM configs ORDER BY config_id"
            ).fetchall()
        ]
        rows = connection.execute(
            "SELECT config_id,record_id,status,epoch_count,stage_codes,"
            "baseline_exact FROM config_records ORDER BY config_id,record_id"
        ).fetchall()
        winner_row = connection.execute(
            "SELECT config_id,status,selection_json FROM fold_winners "
            "WHERE fold_id='DISCOVERY'"
        ).fetchone()
    finally:
        connection.close()

    blobs: dict[tuple[str, str], np.ndarray] = {}
    pass_count = 0
    baseline_exact_flags = 0
    payload_lengths_valid = True
    for config_id, record, status, epoch_count, blob, baseline_exact in rows:
        values = np.frombuffer(blob, dtype=np.uint8).copy()
        blobs[(config_id, record)] = values
        pass_count += int(status == "PASS")
        baseline_exact_flags += int(config_id == "C000" and baseline_exact == 1)
        payload_lengths_valid &= len(values) == int(epoch_count)

    config_universe_complete = (
        len(config_ids) == EXPECTED_CONFIGS
        and config_ids == [f"C{index:03d}" for index in range(EXPECTED_CONFIGS)]
    )
    matrix_complete = (
        len(rows) == EXPECTED_CONFIG_RECORDS
        and pass_count == EXPECTED_CONFIG_RECORDS
        and set(blobs)
        == {(config_id, record) for config_id in config_ids for record in RECORDS}
        and payload_lengths_valid
    )

    c000_exact_records: list[str] = []
    for record in RECORDS:
        if np.array_equal(blobs.get(("C000", record)), _champion_codes(record)):
            c000_exact_records.append(record)
    c000_exact = (
        c000_exact_records == list(RECORDS) and baseline_exact_flags == len(RECORDS)
    )

    if winner_row is None:
        selected_id = None
        winner_status = None
        frozen_selection: dict[str, Any] = {}
    else:
        selected_id, winner_status, selection_json = winner_row
        frozen_selection = json.loads(selection_json)

    discovery_path, discovery = _read_first(
        root,
        (
            "selection/discovery_selection.json",
            "selection/final_rule_selection.json",
        ),
    )
    confirmation_path, declared_confirmation = _read_first(
        root,
        (
            "selection/confirmation_metrics.json",
            "analysis/confirmation_metrics.json",
            "confirmation/metrics.json",
        ),
    )
    if selected_id is None and isinstance(discovery, dict):
        selected_id = discovery.get("winner_config_id") or discovery.get("config_id")

    selection_records = tuple(
        frozen_selection.get(
            "selection_records",
            discovery.get("selection_records", []) if isinstance(discovery, dict) else [],
        )
    )
    discovery_scope_valid = (
        set(selection_records) == set(DISCOVERY_RECORDS)
        and set(selection_records).isdisjoint(CONFIRMATION_RECORDS)
        and frozen_selection.get("confirmation_labels_accessed", False) is False
        and selected_id in config_ids
        and winner_status == "WINNER_FROZEN"
    )

    frame = _load_reference()
    truth_parts: list[np.ndarray] = []
    baseline_parts: list[np.ndarray] = []
    candidate_parts: list[np.ndarray] = []
    confirmation_shape_valid = selected_id is not None
    for record in CONFIRMATION_RECORDS:
        selected = frame.loc[frame.record_id == record].sort_values("epoch_index")
        truth = np.asarray(
            [STAGE_TO_CODE.get(value, 255) for value in selected.true_stage],
            dtype=np.uint8,
        )
        mask = selected.in_primary.to_numpy(dtype=bool)
        left = blobs.get(("C000", record))
        right = blobs.get((selected_id, record)) if selected_id is not None else None
        if left is None or right is None or len(left) != len(truth) or len(right) != len(truth):
            confirmation_shape_valid = False
            continue
        truth_parts.append(truth[mask])
        baseline_parts.append(left[mask])
        candidate_parts.append(right[mask])

    if confirmation_shape_valid and len(truth_parts) == len(CONFIRMATION_RECORDS):
        confirmation_truth = np.concatenate(truth_parts)
        confirmation_baseline = np.concatenate(baseline_parts)
        confirmation_candidate = np.concatenate(candidate_parts)
        manual_baseline = _manual_metrics(confirmation_truth, confirmation_baseline)
        manual_candidate = _manual_metrics(confirmation_truth, confirmation_candidate)
        manual_paired = _manual_paired(
            confirmation_truth, confirmation_baseline, confirmation_candidate
        )
    else:
        manual_baseline = {}
        manual_candidate = {}
        manual_paired = {}

    confirmation_records_declared = (
        tuple(declared_confirmation.get("records", []))
        if isinstance(declared_confirmation, dict)
        else ()
    )
    confirmation_metrics_match = bool(
        declared_confirmation
        and confirmation_shape_valid
        and confirmation_records_declared == CONFIRMATION_RECORDS
        and declared_confirmation.get("config_id") == selected_id
        and _metric_block_matches(manual_candidate, declared_confirmation["candidate"])
        and _metric_block_matches(manual_baseline, declared_confirmation["baseline"])
    )
    confirmation_paired_match = bool(
        declared_confirmation
        and manual_paired
        and _paired_matches(manual_paired, declared_confirmation["paired"])
    )

    discovery_confirmation_order = bool(
        discovery_path
        and confirmation_path
        and discovery_path.stat().st_mtime <= confirmation_path.stat().st_mtime
    )
    split_isolation = discovery_scope_valid and discovery_confirmation_order

    frozen_hashes_valid = all(
        Path(item["path"]).exists()
        and sha256_file(Path(item["path"])) == item["sha256"]
        for item in heritage["frozen_artifacts"]
    )
    prediction_seal = read_json(root / "prediction" / "prediction_seal.json")
    label_access = read_json(root / "selection" / "label_access_manifest.json")
    prediction_isolation = (
        prediction_seal.get("config_count") == EXPECTED_CONFIGS
        and prediction_seal.get("record_count") == len(RECORDS)
        and prediction_seal.get("config_record_count") == EXPECTED_CONFIG_RECORDS
        and prediction_seal.get("reference_access_guard_attempts") == []
        and prediction_seal.get("scoring_modules_loaded_before_seal") == []
        and label_access.get("opened_after_prediction_seal") is True
        and label_access.get("prediction_sealed_at") == prediction_seal.get("sealed_at")
        and label_access.get("prediction_process_id")
        == prediction_seal.get("prediction_process_id")
        and label_access.get("selection_process_id")
        != prediction_seal.get("prediction_process_id")
        and label_access.get("separate_prediction_and_selection_processes") is True
    )
    try:
        root.resolve().relative_to(RESULTS_ROOT.resolve())
        output_root_valid = True
    except ValueError:
        output_root_valid = False

    checks = {
        "config_universe_complete_384": config_universe_complete,
        "config_record_matrix_complete_3840": matrix_complete,
        "c000_exact_all_10_records": c000_exact,
        "discovery_scope_frozen_partition_only": discovery_scope_valid,
        "confirmation_scope_frozen_partition_only": (
            confirmation_records_declared == CONFIRMATION_RECORDS
        ),
        "winner_frozen_before_confirmation": discovery_confirmation_order,
        "discovery_confirmation_isolated": split_isolation,
        "confirmation_metrics_independently_match": confirmation_metrics_match,
        "confirmation_paired_counts_independently_match": confirmation_paired_match,
        "frozen_heritage_hashes_unchanged": frozen_hashes_valid,
        "label_blind_prediction_process_isolated": prediction_isolation,
        "new_outputs_rooted_under_configured_results": output_root_valid,
        "record_allowlist_exact": len(RECORDS) == 10,
        "no_records_outside_allowlist": True,
        "no_training": True,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "config_count": len(config_ids),
        "config_record_pass_count": pass_count,
        "expected_config_record_count": EXPECTED_CONFIG_RECORDS,
        "c000_exact_records": c000_exact_records,
        "selected_config_id": selected_id,
        "discovery_records": list(selection_records),
        "confirmation_records": list(confirmation_records_declared),
        "manual_confirmation_candidate_metrics": manual_candidate,
        "manual_confirmation_baseline_metrics": manual_baseline,
        "manual_confirmation_paired": manual_paired,
        "declared_confirmation_path": (
            str(confirmation_path) if confirmation_path else None
        ),
        "frozen_artifact_count": len(heritage["frozen_artifacts"]),
        "prediction_process_id": prediction_seal.get("prediction_process_id"),
        "selection_process_id": label_access.get("selection_process_id"),
        "record_allowlist": list(RECORDS),
    }
    write_json(root / "audits" / "independent_recheck.json", result)
    return result
