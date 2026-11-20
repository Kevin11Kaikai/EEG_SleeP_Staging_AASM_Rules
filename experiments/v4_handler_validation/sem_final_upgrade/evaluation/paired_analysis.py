from __future__ import annotations

from typing import Any

from ..common import match_intervals


def compare_teacher_and_neural(
    reference: list[tuple[float, float]],
    neural: list[tuple[float, float]],
    minimum_iou: float,
) -> dict[str, Any]:
    result = match_intervals(reference, neural, minimum_iou)
    return {
        "teacher_events": len(reference),
        "neural_events": len(neural),
        "shared_matched_events": result["tp"],
        "teacher_only_events": result["fn"],
        "neural_only_events": result["fp"],
        "temporal_agreement_mean_iou": result["mean_matched_iou"],
    }
