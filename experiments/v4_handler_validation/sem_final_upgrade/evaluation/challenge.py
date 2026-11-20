from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..common import eventize, match_intervals, reference_intervals
from ..data.audit import RecordData


def infer_full_record(
    model: torch.nn.Module,
    signal: np.ndarray,
    device: torch.device,
    window_seconds: int,
    stride_seconds: int,
    batch_size: int,
) -> np.ndarray:
    input_window = window_seconds * 100
    output_window = window_seconds * 10
    duration_seconds = signal.shape[-1] // 100
    output_length = duration_seconds * 10
    starts = list(range(0, max(1, duration_seconds - window_seconds + 1), stride_seconds))
    final = max(0, duration_seconds - window_seconds)
    if not starts or starts[-1] != final:
        starts.append(final)
    accumulator = np.zeros(output_length, dtype=np.float64)
    weights = np.zeros(output_length, dtype=np.float64)
    taper = 0.25 + 0.75 * np.hanning(output_window)
    model.eval()
    for batch_start in range(0, len(starts), batch_size):
        batch_starts = starts[batch_start : batch_start + batch_size]
        chunks = []
        for start in batch_starts:
            left = start * 100
            chunk = signal[:, left : left + input_window]
            if chunk.shape[-1] < input_window:
                chunk = np.pad(chunk, ((0, 0), (0, input_window - chunk.shape[-1])))
            chunks.append(chunk)
        tensor = torch.from_numpy(np.stack(chunks)).float().to(device)
        with torch.inference_mode():
            probability = torch.sigmoid(model(tensor))[:, 0].cpu().numpy()
        for index, start in enumerate(batch_starts):
            left = start * 10
            right = min(output_length, left + output_window)
            width = right - left
            accumulator[left:right] += probability[index, :width] * taper[:width]
            weights[left:right] += taper[:width]
    if np.any(weights <= 0):
        raise AssertionError("Overlap-add SEM inference left uncovered samples")
    output = (accumulator / weights).astype(np.float32)
    if not np.isfinite(output).all():
        raise FloatingPointError("Non-finite SEM probability")
    return output


def _evaluate_one(
    record: RecordData,
    probability: np.ndarray,
    threshold: float,
    eventization: dict[str, float],
    minimum_iou: float,
) -> tuple[dict[str, Any], list[tuple[float, float]], list[tuple[float, float]]]:
    truth = reference_intervals(record.target, record.scoring_mask, 10)
    predicted = eventize(
        probability,
        record.scoring_mask,
        threshold,
        10,
        float(eventization["minimum_seconds"]),
        float(eventization["maximum_seconds"]),
        float(eventization["merge_gap_seconds"]),
    )
    matched = match_intervals(truth, predicted, minimum_iou)
    scored_hours = float(record.scoring_mask.sum() / 10 / 3600)
    return (
        {
            "record_id": record.record_id,
            "dataset_id": record.dataset_id,
            "split_role": record.split_role,
            "reference_events": len(truth),
            "predicted_events": len(predicted),
            "tp": matched["tp"],
            "fp": matched["fp"],
            "fn": matched["fn"],
            "precision": matched["precision"],
            "recall": matched["recall"] if truth else None,
            "f1": matched["f1"] if truth else None,
            "mean_matched_iou": matched["mean_matched_iou"],
            "fp_per_hour": matched["fp"] / scored_hours if scored_hours else None,
            "scored_hours": scored_hours,
            "probability_mean": float(np.mean(probability)),
            "probability_std": float(np.std(probability)),
            "threshold": float(threshold),
        },
        truth,
        predicted,
    )


def evaluate_records(
    records: dict[str, RecordData],
    record_ids: list[str],
    probabilities: dict[str, np.ndarray],
    threshold: float,
    eventization: dict[str, float],
    minimum_iou: float,
) -> dict[str, Any]:
    per_record = []
    truth_total = 0
    predicted_total = 0
    tp_total = 0
    scored_hours = 0.0
    intervals = {}
    for record_id in record_ids:
        row, truth, predicted = _evaluate_one(
            records[record_id],
            probabilities[record_id],
            threshold,
            eventization,
            minimum_iou,
        )
        per_record.append(row)
        truth_total += len(truth)
        predicted_total += len(predicted)
        tp_total += int(row["tp"])
        scored_hours += float(row["scored_hours"])
        intervals[record_id] = {"truth": truth, "predicted": predicted}
    fp = predicted_total - tp_total
    fn = truth_total - tp_total
    precision = tp_total / predicted_total if predicted_total else 0.0
    recall = tp_total / truth_total if truth_total else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if recall is not None and precision + recall
        else (0.0 if recall is not None else None)
    )
    return {
        "aggregate": {
            "reference_events": truth_total,
            "predicted_events": predicted_total,
            "tp": tp_total,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fp_per_hour": fp / scored_hours if scored_hours else None,
            "scored_hours": scored_hours,
            "metric_status": (
                "WEAK_TEACHER_FIDELITY"
                if truth_total
                else "NOT_DEFINED_NO_POSITIVES"
            ),
        },
        "per_record": per_record,
        "intervals": intervals,
    }


def select_validation_threshold(
    records: dict[str, RecordData],
    record_ids: list[str],
    probabilities: dict[str, np.ndarray],
    thresholds: list[float],
    eventization: dict[str, float],
    minimum_iou: float,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    rows = []
    best = None
    for threshold in thresholds:
        result = evaluate_records(
            records,
            record_ids,
            probabilities,
            threshold,
            eventization,
            minimum_iou,
        )
        row = {"threshold": threshold, **result["aggregate"]}
        rows.append(row)
        key = (
            float(row["f1"] or 0.0),
            float(row["precision"]),
            -float(row["fp_per_hour"] or 0.0),
            float(threshold),
        )
        if best is None or key > best[0]:
            best = (key, float(threshold), result)
    if best is None:
        raise AssertionError("No validation thresholds evaluated")
    return best[1], best[2], rows
