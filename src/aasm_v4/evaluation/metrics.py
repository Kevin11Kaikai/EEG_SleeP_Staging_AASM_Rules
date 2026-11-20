"""Small dependency-free metric recheck used by the public release.

``UNDETERMINED`` is included as a prediction column but is never a reference
stage.  Fixed-mask accuracy therefore counts abstentions as errors, while
coverage reports their frequency explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..contracts import STAGES, UNDETERMINED


PREDICTION_LABELS: tuple[str, ...] = (*STAGES, UNDETERMINED)


def confusion_matrix(
    reference: Sequence[str], prediction: Sequence[str]
) -> list[list[int]]:
    """Return a 5-by-6 fixed-label confusion matrix."""

    if len(reference) != len(prediction):
        raise ValueError("reference and prediction lengths differ")
    stage_index = {label: index for index, label in enumerate(STAGES)}
    prediction_index = {label: index for index, label in enumerate(PREDICTION_LABELS)}
    matrix = [[0 for _ in PREDICTION_LABELS] for _ in STAGES]
    for truth, guess in zip(reference, prediction):
        if truth not in stage_index:
            raise ValueError(f"unsupported reference stage: {truth}")
        if guess not in prediction_index:
            raise ValueError(f"unsupported predicted stage: {guess}")
        matrix[stage_index[truth]][prediction_index[guess]] += 1
    return matrix


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def compute_fixed_mask_metrics(
    reference: Sequence[str], prediction: Sequence[str]
) -> dict[str, Any]:
    """Compute accuracy, macro-F1, coverage, and per-stage metrics."""

    matrix = confusion_matrix(reference, prediction)
    total = len(reference)
    accepted = sum(guess != UNDETERMINED for guess in prediction)
    correct = sum(matrix[index][index] for index in range(len(STAGES)))
    per_stage: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(STAGES):
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        precision = _safe_ratio(true_positive, predicted)
        recall = _safe_ratio(true_positive, support)
        f1 = _safe_ratio(2 * true_positive, predicted + support)
        per_stage[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return {
        "n": total,
        "correct": correct,
        "accuracy": _safe_ratio(correct, total),
        "macro_f1": sum(float(per_stage[label]["f1"]) for label in STAGES) / len(STAGES),
        "coverage": _safe_ratio(accepted, total),
        "conditional_accuracy": _safe_ratio(correct, accepted),
        "per_stage": per_stage,
        "confusion": matrix,
    }
