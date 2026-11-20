from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .cache import PredictionCache
from .config import STAGES, STAGE_TO_CODE, RuleConfig


def confusion_from_codes(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    if truth.shape != prediction.shape:
        raise ValueError((truth.shape, prediction.shape))
    matrix = np.zeros((6, 6), dtype=np.int64)
    np.add.at(matrix, (truth, prediction), 1)
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (6, 6) or int(matrix.sum()) == 0:
        raise ValueError("invalid confusion matrix")
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    diagonal = matrix.diagonal()
    precision = np.divide(diagonal, predicted, out=np.zeros(6), where=predicted > 0)
    recall = np.divide(diagonal, support, out=np.zeros(6), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(6), where=precision + recall > 0)
    total = int(matrix.sum())
    correct = int(diagonal[:5].sum())
    accepted = total - int(predicted[5])
    expected = float(predicted @ support) / total**2
    return {
        "n": total,
        "correct": correct,
        "accuracy": correct / total,
        "macro_f1": float(f1[:5].mean()),
        "coverage": accepted / total,
        "conditional_accuracy": None if accepted == 0 else correct / accepted,
        "abstention_rate": int(predicted[5]) / total,
        "kappa": (correct / total - expected) / (1 - expected) if expected < 1 else None,
        "per_stage": {
            stage: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, stage in enumerate(STAGES[:5])
        },
        "confusion": matrix.tolist(),
    }


class ReferenceData:
    def __init__(self, path: Path, allowed_records: Iterable[str]) -> None:
        self.allowed_records = tuple(allowed_records)
        frame = pd.read_csv(path)
        observed = tuple(sorted(frame.record_id.unique()))
        if set(observed) != set(self.allowed_records):
            raise RuntimeError(f"reference shard mismatch: {observed} != {self.allowed_records}")
        self.rows: dict[str, dict[str, np.ndarray]] = {}
        for record in self.allowed_records:
            selected = frame.loc[frame.record_id == record].sort_values("epoch_index")
            if selected.epoch_index.tolist() != list(range(len(selected))):
                raise RuntimeError(f"reference epoch index mismatch: {record}")
            truth = np.asarray([STAGE_TO_CODE.get(value, 255) for value in selected.true_stage], dtype=np.uint8)
            primary = selected.in_primary.to_numpy(dtype=bool)
            full = selected.in_full_record.to_numpy(dtype=bool)
            if np.any(truth[primary] > 4) or np.any(truth[full] > 4):
                raise RuntimeError(f"invalid scored truth: {record}")
            self.rows[record] = {"truth": truth, "primary": primary, "full": full}

    def matrix(self, cache: PredictionCache, config_id: str, record: str, scope: str = "primary") -> np.ndarray:
        if record not in self.rows:
            raise PermissionError(f"record not present in reference shard: {record}")
        data = self.rows[record]
        prediction = cache.stage_codes(config_id, record)
        mask = data[scope]
        if len(prediction) != len(data["truth"]):
            raise RuntimeError(f"prediction/reference mismatch: {config_id}/{record}")
        return confusion_from_codes(data["truth"][mask], prediction[mask])

    def score(self, cache: PredictionCache, config_id: str, records: Iterable[str], scope: str = "primary") -> dict[str, Any]:
        selected = tuple(records)
        if not selected or not set(selected) <= set(self.allowed_records):
            raise PermissionError(f"invalid record selection: {selected}")
        matrix = sum((self.matrix(cache, config_id, record, scope) for record in selected), np.zeros((6, 6), dtype=np.int64))
        result = metrics_from_confusion(matrix)
        result["records"] = list(selected)
        return result

    def paired(self, cache: PredictionCache, config_id: str, records: Iterable[str]) -> dict[str, Any]:
        corrected = introduced = changed = 0
        transitions: dict[str, int] = {}
        by_stage = {stage: {"corrected": 0, "introduced": 0} for stage in STAGES[:5]}
        for record in records:
            data = self.rows[record]
            mask = data["primary"]
            truth = data["truth"][mask]
            baseline = cache.stage_codes("C000", record)[mask]
            candidate = cache.stage_codes(config_id, record)[mask]
            changed += int(np.sum(baseline != candidate))
            for y, left, right in zip(truth.tolist(), baseline.tolist(), candidate.tolist()):
                if left != right:
                    key = f"{STAGES[left]}->{STAGES[right]}"
                    transitions[key] = transitions.get(key, 0) + 1
                if left != y and right == y:
                    corrected += 1
                    by_stage[STAGES[y]]["corrected"] += 1
                elif left == y and right != y:
                    introduced += 1
                    by_stage[STAGES[y]]["introduced"] += 1
        return {
            "changed": changed,
            "corrected": corrected,
            "introduced": introduced,
            "net_correct": corrected - introduced,
            "transitions": transitions,
            "by_reference_stage": by_stage,
        }


def candidate_summary(
    reference: ReferenceData,
    cache: PredictionCache,
    config: RuleConfig,
    records: tuple[str, ...],
    coverage_tolerance: float = 0.005,
    stage_f1_guardrail: float = 0.03,
    worst_night_accuracy: float = -0.05,
) -> dict[str, Any]:
    candidate = reference.score(cache, config.config_id, records)
    baseline = reference.score(cache, "C000", records)
    per_record = []
    for record in records:
        current = reference.score(cache, config.config_id, (record,))
        base = reference.score(cache, "C000", (record,))
        per_record.append(
            {
                "record_id": record,
                "delta_accuracy": current["accuracy"] - base["accuracy"],
                "delta_macro_f1": current["macro_f1"] - base["macro_f1"],
                "delta_coverage": current["coverage"] - base["coverage"],
            }
        )
    acc = np.asarray([row["delta_accuracy"] for row in per_record], dtype=float)
    macro = np.asarray([row["delta_macro_f1"] for row in per_record], dtype=float)
    balanced = (acc + macro) / 2
    stage_deltas = {
        stage: candidate["per_stage"][stage]["f1"] - baseline["per_stage"][stage]["f1"]
        for stage in STAGES[:5]
    }
    record_equal_accuracy = float(acc.mean())
    record_equal_macro = float(macro.mean())
    standard_error = float(balanced.std(ddof=1) / np.sqrt(len(balanced))) if len(balanced) > 1 else 0.0
    eligible = (
        config.config_id != "C000"
        and candidate["coverage"] >= baseline["coverage"] - coverage_tolerance
        and record_equal_accuracy >= 0
        and record_equal_macro >= 0
        and min(stage_deltas.values()) >= -stage_f1_guardrail
        and int(np.sum(acc >= 0)) >= 3
        and float(acc.min()) >= worst_night_accuracy
    )
    return {
        "config_id": config.config_id,
        "accuracy": candidate["accuracy"],
        "macro_f1": candidate["macro_f1"],
        "coverage": candidate["coverage"],
        "delta_accuracy": candidate["accuracy"] - baseline["accuracy"],
        "delta_macro_f1": candidate["macro_f1"] - baseline["macro_f1"],
        "delta_coverage": candidate["coverage"] - baseline["coverage"],
        "stage_f1_deltas": stage_deltas,
        "record_equal_accuracy_delta": record_equal_accuracy,
        "record_equal_macro_f1_delta": record_equal_macro,
        "worst_night_accuracy_delta": float(acc.min()),
        "nonnegative_accuracy_nights": int(np.sum(acc >= 0)),
        "balanced_standard_error": standard_error,
        "robust_score": float(balanced.mean() - standard_error - 0.00025 * config.complexity),
        "eligible": bool(eligible),
        "complexity": config.complexity,
        "per_record": per_record,
    }


def choose_config(summaries: list[dict[str, Any]]) -> tuple[str, str]:
    eligible = [row for row in summaries if row["eligible"]]
    if not eligible:
        return "C000", "NO_ELIGIBLE_CANDIDATE_RETAIN_R1"
    winner = max(
        eligible,
        key=lambda row: (
            row["robust_score"],
            row["record_equal_macro_f1_delta"],
            row["record_equal_accuracy_delta"],
            -row["complexity"],
            row["config_id"],
        ),
    )
    return winner["config_id"], "DISCOVERY_ROBUST_SCORE_WITH_GUARDRAILS"
