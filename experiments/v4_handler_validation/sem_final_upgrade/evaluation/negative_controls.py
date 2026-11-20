from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..common import eventize, match_intervals, reference_intervals
from ..data.audit import RecordData
from .challenge import evaluate_records, infer_full_record


def time_shift_control(
    records: dict[str, RecordData],
    ids: list[str],
    probabilities: dict[str, np.ndarray],
    threshold: float,
    eventization: dict[str, float],
    minimum_iou: float,
    seconds: int = 60,
) -> dict[str, Any]:
    shifted = {}
    width = seconds * 10
    for record_id in ids:
        values = np.zeros_like(probabilities[record_id])
        values[width:] = probabilities[record_id][:-width]
        shifted[record_id] = values
    return evaluate_records(
        records, ids, shifted, threshold, eventization, minimum_iou
    )["aggregate"]


def constant_negative_control(
    records: dict[str, RecordData], ids: list[str]
) -> dict[str, Any]:
    reference_count = sum(
        len(reference_intervals(records[key].target, records[key].scoring_mask, 10))
        for key in ids
    )
    return {
        "reference_events": reference_count,
        "predicted_events": 0,
        "tp": 0,
        "fp": 0,
        "fn": reference_count,
        "precision": 0.0,
        "recall": 0.0 if reference_count else None,
        "f1": 0.0 if reference_count else None,
    }


def overlap_confusion(
    record: RecordData,
    predicted: list[tuple[float, float]],
    mask: np.ndarray,
    label: str,
) -> dict[str, Any]:
    reference = reference_intervals(mask, record.signal_qc, 10)
    matched = match_intervals(reference, predicted, 0.01)
    return {
        "kind": label,
        "reference_intervals": len(reference),
        "sem_predictions": len(predicted),
        "overlapping_predictions": matched["tp"],
        "overlap_fraction_of_sem_predictions": (
            matched["tp"] / len(predicted) if predicted else 0.0
        ),
    }


def channel_transform_audit(
    model: torch.nn.Module,
    record: RecordData,
    base_probability: np.ndarray,
    device: torch.device,
    model_config: dict[str, Any],
    threshold: float,
    eventization: dict[str, float],
) -> dict[str, Any]:
    variants = {
        "channel_swap": record.signal[[1, 0]],
        "right_polarity_flip": np.stack([record.signal[0], -record.signal[1]]),
    }
    base_events = eventize(
        base_probability,
        record.signal_qc,
        threshold,
        10,
        eventization["minimum_seconds"],
        eventization["maximum_seconds"],
        eventization["merge_gap_seconds"],
    )
    rows = {}
    for name, signal in variants.items():
        probability = infer_full_record(
            model,
            signal,
            device,
            int(model_config["window_seconds"]),
            int(model_config["inference_stride_seconds"]),
            int(model_config["batch_size"]),
        )
        events = eventize(
            probability,
            record.signal_qc,
            threshold,
            10,
            eventization["minimum_seconds"],
            eventization["maximum_seconds"],
            eventization["merge_gap_seconds"],
        )
        agreement = match_intervals(base_events, events, 0.2)
        rows[name] = {
            "events": len(events),
            "agreement_with_base_f1": agreement["f1"],
            "probability_mean_absolute_delta": float(
                np.mean(np.abs(probability - base_probability))
            ),
        }
    return rows
