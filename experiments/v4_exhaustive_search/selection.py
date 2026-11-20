from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sqlite3
from typing import Any, Iterable

import numpy as np
import pandas as pd

from experiments.v4_subset_search.metrics import build_nested_splits

from .config import (
    ALIGNED_REFERENCE,
    HANDLERS,
    RECORDS,
    STAGE_NAMES,
    STAGE_TO_CODE,
    ExtendedLimits,
)


class ReferenceData:
    def __init__(self) -> None:
        frame = pd.read_csv(ALIGNED_REFERENCE)
        self.truth: dict[str, np.ndarray] = {}
        self.primary: dict[str, np.ndarray] = {}
        for record in RECORDS:
            rows = frame.loc[frame.record_id == record].sort_values("epoch_index")
            truth = np.asarray(
                [STAGE_TO_CODE.get(value, 255) for value in rows.true_stage], dtype=np.uint8
            )
            primary = rows.in_primary.to_numpy(dtype=bool)
            if np.any(truth[primary] > 4):
                raise RuntimeError(f"invalid scored truth: {record}")
            self.truth[record] = truth
            self.primary[record] = primary
        if sum(int(self.primary[record].sum()) for record in RECORDS) != 8902:
            raise RuntimeError("primary mask changed")


def confusion_from_codes(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    encoded = truth.astype(np.int64) * 6 + prediction.astype(np.int64)
    return np.bincount(encoded, minlength=36).reshape(6, 6).astype(np.int32)


def load_confusion_cube(
    cache_path: Path,
    masks: Iterable[int],
    reference: ReferenceData,
) -> tuple[np.ndarray, np.ndarray]:
    mask_values = np.asarray(sorted(set(int(mask) for mask in masks)), dtype=np.int32)
    mask_index = {int(mask): index for index, mask in enumerate(mask_values)}
    record_index = {record: index for index, record in enumerate(RECORDS)}
    cube = np.zeros((len(mask_values), len(RECORDS), 6, 6), dtype=np.int32)
    seen = np.zeros((len(mask_values), len(RECORDS)), dtype=bool)
    connection = sqlite3.connect(
        f"file:{cache_path.resolve().as_posix()}?mode=ro&immutable=1", uri=True
    )
    try:
        rows = connection.execute(
            """SELECT ar.mask,ar.record_id,ar.stage_codes,ar.epoch_count
               FROM arm_records ar JOIN arms a ON a.mask=ar.mask
               WHERE a.status='PASS' AND ar.status='PASS'
               ORDER BY ar.mask,ar.record_id"""
        )
        for mask, record, blob, epoch_count in rows:
            index = mask_index.get(int(mask))
            if index is None:
                continue
            prediction = np.frombuffer(blob, dtype=np.uint8)
            truth = reference.truth[record]
            valid = reference.primary[record]
            if len(prediction) != int(epoch_count) or len(prediction) != len(truth):
                raise RuntimeError(f"prediction/reference mismatch: {mask}/{record}")
            cube[index, record_index[record]] = confusion_from_codes(
                truth[valid], prediction[valid]
            )
            seen[index, record_index[record]] = True
    finally:
        connection.close()
    missing = np.argwhere(~seen)
    if len(missing):
        examples = [
            (int(mask_values[left]), RECORDS[right]) for left, right in missing[:20]
        ]
        raise RuntimeError(f"complete mask lacks record confusion: {examples}")
    return mask_values, cube


def vector_metrics(matrices: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(matrices, dtype=np.int64)
    if values.ndim == 2:
        values = values[None, :, :]
    diagonal = np.diagonal(values, axis1=1, axis2=2)
    support = values.sum(axis=2)
    predicted = values.sum(axis=1)
    n = values.sum(axis=(1, 2))
    correct = diagonal.sum(axis=1)
    precision = np.divide(
        diagonal,
        predicted,
        out=np.zeros_like(diagonal, dtype=np.float64),
        where=predicted > 0,
    )
    recall = np.divide(
        diagonal,
        support,
        out=np.zeros_like(diagonal, dtype=np.float64),
        where=support > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(diagonal, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "macro_f1": f1[:, :5].mean(axis=1),
        "coverage": 1.0 - predicted[:, 5] / n,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "support": support,
    }


def metric_dict(matrix: np.ndarray) -> dict[str, Any]:
    arrays = vector_metrics(matrix)
    return {
        "n": int(arrays["n"][0]),
        "correct": int(arrays["correct"][0]),
        "accuracy": float(arrays["accuracy"][0]),
        "macro_f1": float(arrays["macro_f1"][0]),
        "coverage": float(arrays["coverage"][0]),
        "per_stage": {
            stage: {
                "precision": float(arrays["precision"][0, index]),
                "recall": float(arrays["recall"][0, index]),
                "f1": float(arrays["f1"][0, index]),
                "support": int(arrays["support"][0, index]),
            }
            for index, stage in enumerate(STAGE_NAMES[:5])
        },
        "confusion": np.asarray(matrix, dtype=np.int64).tolist(),
    }


def _candidate_indices(mask_values: np.ndarray, masks: set[int]) -> np.ndarray:
    return np.asarray(
        [index for index, mask in enumerate(mask_values) if int(mask) in masks], dtype=np.int32
    )


def select_outer_winners(
    mask_values: np.ndarray,
    cube: np.ndarray,
    structural_masks: set[int],
    safe_warm_masks: dict[str, set[int]],
    limits: ExtendedLimits,
    exhaustive: bool,
) -> list[dict[str, Any]]:
    mask_to_index = {int(mask): index for index, mask in enumerate(mask_values)}
    baseline_index = mask_to_index[0]
    output: list[dict[str, Any]] = []
    for access in build_nested_splits(limits.seed):
        allowed = set(map(int, mask_values)) if exhaustive else (
            structural_masks | safe_warm_masks[access.fold_id]
        )
        allowed &= set(map(int, mask_values))
        candidates = _candidate_indices(mask_values, allowed)
        selection_records = [RECORDS.index(record) for record in access.selection_records]
        pooled = cube[candidates][:, selection_records].sum(axis=1, dtype=np.int64)
        pooled_metric = vector_metrics(pooled)
        baseline = metric_dict(cube[baseline_index, selection_records].sum(axis=0))
        inner_accuracy = []
        inner_macro = []
        for records in access.inner_validation_records:
            record_indices = [RECORDS.index(record) for record in records]
            current = vector_metrics(
                cube[candidates][:, record_indices].sum(axis=1, dtype=np.int64)
            )
            base = metric_dict(cube[baseline_index, record_indices].sum(axis=0))
            inner_accuracy.append(current["accuracy"] - base["accuracy"])
            inner_macro.append(current["macro_f1"] - base["macro_f1"])
        inner_accuracy_values = np.stack(inner_accuracy, axis=1)
        inner_macro_values = np.stack(inner_macro, axis=1)
        eligible = (
            (pooled_metric["coverage"] + limits.epsilon >= limits.coverage_floor)
            & (pooled_metric["accuracy"] + limits.epsilon >= baseline["accuracy"])
            & (pooled_metric["macro_f1"] + limits.epsilon >= baseline["macro_f1"])
        )
        eligible_positions = np.flatnonzero(eligible)
        if not len(eligible_positions):
            eligible_positions = np.asarray(
                [int(np.flatnonzero(mask_values[candidates] == 0)[0])], dtype=np.int64
            )
        def key(position: int) -> tuple[Any, ...]:
            mask = int(mask_values[candidates[position]])
            return (
                float(inner_accuracy_values[position].mean()),
                float(inner_macro_values[position].mean()),
                float(inner_accuracy_values[position].min()),
                -mask.bit_count(),
                -mask,
            )
        best_position = max((int(value) for value in eligible_positions), key=key)
        winner_mask = int(mask_values[candidates[best_position]])
        output.append({
            "fold_id": access.fold_id,
            "heldout_record": access.heldout_record,
            "selection_records": list(access.selection_records),
            "inner_validation_records": [list(values) for values in access.inner_validation_records],
            "candidate_universe_count": int(len(candidates)),
            "winner_mask": winner_mask,
            "winner_handlers": [
                handler for bit, handler in enumerate(HANDLERS) if winner_mask & (1 << bit)
            ],
            "winner_summary": {
                "accuracy": float(pooled_metric["accuracy"][best_position]),
                "macro_f1": float(pooled_metric["macro_f1"][best_position]),
                "coverage": float(pooled_metric["coverage"][best_position]),
                "mean_inner_accuracy_delta": float(inner_accuracy_values[best_position].mean()),
                "mean_inner_macro_delta": float(inner_macro_values[best_position].mean()),
                "worst_inner_accuracy_delta": float(inner_accuracy_values[best_position].min()),
                "eligible": bool(eligible[best_position]),
            },
        })
    return output


def score_frozen_winners(
    mask_values: np.ndarray,
    cube: np.ndarray,
    winners: list[dict[str, Any]],
    limits: ExtendedLimits,
) -> dict[str, Any]:
    mask_to_index = {int(mask): index for index, mask in enumerate(mask_values)}
    rows = []
    candidate_matrices = []
    baseline_matrices = []
    for winner in winners:
        record = winner["heldout_record"]
        record_index = RECORDS.index(record)
        candidate = cube[mask_to_index[int(winner["winner_mask"])], record_index].astype(np.int64)
        baseline = cube[mask_to_index[0], record_index].astype(np.int64)
        candidate_metric = metric_dict(candidate)
        baseline_metric = metric_dict(baseline)
        rows.append({
            **winner,
            "candidate_metrics": candidate_metric,
            "baseline_metrics": baseline_metric,
            "delta_accuracy": candidate_metric["accuracy"] - baseline_metric["accuracy"],
            "delta_macro_f1": candidate_metric["macro_f1"] - baseline_metric["macro_f1"],
            "delta_coverage": candidate_metric["coverage"] - baseline_metric["coverage"],
        })
        candidate_matrices.append(candidate)
        baseline_matrices.append(baseline)
    pooled_candidate = sum(candidate_matrices, np.zeros((6, 6), dtype=np.int64))
    pooled_baseline = sum(baseline_matrices, np.zeros((6, 6), dtype=np.int64))
    candidate_metric = metric_dict(pooled_candidate)
    baseline_metric = metric_dict(pooled_baseline)
    rng = np.random.default_rng(limits.seed)
    samples = {"accuracy": [], "macro_f1": [], "coverage": []}
    for _ in range(2000):
        indices = rng.integers(0, len(RECORDS), size=len(RECORDS))
        left = sum(
            (baseline_matrices[index] for index in indices), np.zeros((6, 6), dtype=np.int64)
        )
        right = sum(
            (candidate_matrices[index] for index in indices), np.zeros((6, 6), dtype=np.int64)
        )
        left_metric = metric_dict(left)
        right_metric = metric_dict(right)
        for key in samples:
            samples[key].append(right_metric[key] - left_metric[key])
    bootstrap = {
        key: {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        }
        for key, values in samples.items()
    }
    return {
        "rows": rows,
        "candidate": candidate_metric,
        "baseline": baseline_metric,
        "delta": {
            key: candidate_metric[key] - baseline_metric[key]
            for key in ("accuracy", "macro_f1", "coverage")
        },
        "bootstrap": bootstrap,
    }


def full_development_results(
    mask_values: np.ndarray,
    cube: np.ndarray,
    limits: ExtendedLimits,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    matrices = cube.sum(axis=1, dtype=np.int64)
    metric = vector_metrics(matrices)
    baseline_position = int(np.flatnonzero(mask_values == 0)[0])
    baseline_accuracy = float(metric["accuracy"][baseline_position])
    baseline_macro = float(metric["macro_f1"][baseline_position])
    eligible = (
        (metric["coverage"] + limits.epsilon >= limits.coverage_floor)
        & (metric["accuracy"] + limits.epsilon >= baseline_accuracy)
        & (metric["macro_f1"] + limits.epsilon >= baseline_macro)
    )
    rows = []
    for index, mask_value in enumerate(mask_values):
        mask = int(mask_value)
        rows.append({
            "mask": mask,
            "arm_id": f"M_{mask:04X}",
            "handlers": ";".join(
                handler for bit, handler in enumerate(HANDLERS) if mask & (1 << bit)
            ),
            "cardinality": mask.bit_count(),
            "accuracy": float(metric["accuracy"][index]),
            "macro_f1": float(metric["macro_f1"][index]),
            "coverage": float(metric["coverage"][index]),
            "delta_accuracy": float(metric["accuracy"][index] - baseline_accuracy),
            "delta_macro_f1": float(metric["macro_f1"][index] - baseline_macro),
            "eligible": bool(eligible[index]),
        })
    leaderboard = pd.DataFrame(rows)
    accuracy_row = leaderboard.sort_values(
        ["accuracy", "macro_f1", "coverage", "cardinality", "mask"],
        ascending=[False, False, False, True, True],
    ).iloc[0].to_dict()
    eligible_rows = leaderboard.loc[leaderboard.eligible]
    balanced_row = eligible_rows.sort_values(
        ["accuracy", "macro_f1", "coverage", "cardinality", "mask"],
        ascending=[False, False, False, True, True],
    ).iloc[0].to_dict()
    macro_row = leaderboard.sort_values(
        ["macro_f1", "accuracy", "coverage", "cardinality", "mask"],
        ascending=[False, False, False, True, True],
    ).iloc[0].to_dict()
    return leaderboard, {
        "accuracy_champion": accuracy_row,
        "balanced_champion": balanced_row,
        "macro_f1_champion_unconstrained": macro_row,
        "balanced_champion_metrics": metric_dict(
            matrices[int(np.flatnonzero(mask_values == int(balanced_row["mask"]))[0])]
        ),
    }


def nested_paired_counts(
    cache_path: Path,
    winners: list[dict[str, Any]],
    reference: ReferenceData,
) -> dict[str, int]:
    connection = sqlite3.connect(
        f"file:{cache_path.resolve().as_posix()}?mode=ro&immutable=1", uri=True
    )
    corrected = introduced = both_correct = both_incorrect = 0
    try:
        for row in winners:
            record = row["heldout_record"]
            winner = int(row["winner_mask"])
            blobs = dict(connection.execute(
                "SELECT mask,stage_codes FROM arm_records WHERE record_id=? AND mask IN (?,?)",
                (record, 0, winner),
            ).fetchall())
            truth = reference.truth[record][reference.primary[record]]
            baseline = np.frombuffer(blobs[0], dtype=np.uint8)[reference.primary[record]]
            candidate = np.frombuffer(blobs[winner], dtype=np.uint8)[reference.primary[record]]
            left = baseline == truth
            right = candidate == truth
            corrected += int(np.sum(~left & right))
            introduced += int(np.sum(left & ~right))
            both_correct += int(np.sum(left & right))
            both_incorrect += int(np.sum(~left & ~right))
    finally:
        connection.close()
    return {
        "corrected": corrected,
        "introduced": introduced,
        "net_correct": corrected - introduced,
        "both_correct": both_correct,
        "both_incorrect": both_incorrect,
    }
