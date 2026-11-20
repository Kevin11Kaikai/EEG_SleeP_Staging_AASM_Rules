from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import binary_counts, intervals_from_mask, match_intervals, rasterize
from .models import FAMILY_SPECS


def apply_output_semantics(probability: np.ndarray, family: str) -> np.ndarray:
    output = np.asarray(probability, dtype=np.float32).copy()
    if family == "chin_emg":
        # Frozen V3 semantics require high_emg to also satisfy normal_emg.
        output[1] = np.maximum(output[1], output[2])
    return output


def infer_full_record(
    model: torch.nn.Module,
    signal: np.ndarray,
    family: str,
    window_seconds: int,
    device: torch.device,
    *,
    batch_size: int = 16,
) -> np.ndarray:
    spec = FAMILY_SPECS[family]
    input_window = window_seconds * spec.input_rate_hz
    output_window = window_seconds * spec.output_rate_hz
    duration_seconds = signal.shape[-1] // spec.input_rate_hz
    output_length = duration_seconds * spec.output_rate_hz
    stride_seconds = max(1, window_seconds // 2)
    starts = list(range(0, max(1, duration_seconds - window_seconds + 1), stride_seconds))
    final_start = max(0, duration_seconds - window_seconds)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    accumulator = np.zeros((len(spec.output_heads), output_length), dtype=np.float64)
    weights = np.zeros(output_length, dtype=np.float64)
    taper = 0.25 + 0.75 * np.hanning(output_window)
    model.eval()
    for batch_start in range(0, len(starts), batch_size):
        batch_starts = starts[batch_start : batch_start + batch_size]
        chunks = []
        for start in batch_starts:
            left = start * spec.input_rate_hz
            chunk = signal[:, left : left + input_window]
            if chunk.shape[-1] < input_window:
                chunk = np.pad(chunk, ((0, 0), (0, input_window - chunk.shape[-1])))
            chunks.append(chunk)
        tensor = torch.from_numpy(np.stack(chunks)).float().to(device)
        with torch.inference_mode():
            probability = torch.sigmoid(model(tensor)).cpu().numpy()
        for index, start in enumerate(batch_starts):
            output_start = start * spec.output_rate_hz
            right = min(output_length, output_start + output_window)
            width = right - output_start
            accumulator[:, output_start:right] += probability[index, :, :width] * taper[:width]
            weights[output_start:right] += taper[:width]
    if np.any(weights <= 0):
        raise AssertionError("Overlap-add inference left uncovered output samples")
    output = apply_output_semantics(
        (accumulator / weights[None, :]).astype(np.float32), family
    )
    if not np.isfinite(output).all():
        raise FloatingPointError("Non-finite probabilities during full-record inference")
    return output


def eventize_head(
    probability: np.ndarray,
    threshold: float,
    rate_hz: int,
    config: dict[str, float],
) -> list[tuple[float, float]]:
    return intervals_from_mask(
        probability >= threshold,
        rate_hz,
        minimum_seconds=float(config["minimum_seconds"]),
        merge_gap_seconds=float(config["merge_gap_seconds"]),
    )


def evaluate_probability(
    probability: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    rate_hz: int,
    eventization: dict[str, float],
) -> dict[str, Any]:
    prediction = probability >= threshold
    temporal = binary_counts(target, prediction, valid)
    reference_intervals = intervals_from_mask(
        (target > 0) & (valid > 0), rate_hz, minimum_seconds=1 / rate_hz
    )
    predicted_intervals = eventize_head(
        probability, threshold, rate_hz, eventization
    )
    event = match_intervals(reference_intervals, predicted_intervals, minimum_iou=0.2)
    return {
        **{f"temporal_{key}": value for key, value in temporal.items()},
        **{f"event_{key}": value for key, value in event.items()},
        "teacher_interval_count": len(reference_intervals),
        "prediction_interval_count": len(predicted_intervals),
        "prediction_fraction": float(np.mean(prediction)),
        "probability_mean": float(np.mean(probability)),
        "probability_std": float(np.std(probability)),
        "threshold": float(threshold),
    }


def build_event_rows(
    record_id: str,
    handler_id: str,
    event_type: str,
    probability: np.ndarray,
    threshold: float,
    rate_hz: int,
    eventization: dict[str, float],
    checkpoint_sha256: str,
    source_channel: str,
    source_sampling_rate_hz: float,
) -> list[dict[str, Any]]:
    intervals = eventize_head(probability, threshold, rate_hz, eventization)
    rows = []
    for index, (onset, offset) in enumerate(intervals):
        left = int(math.floor(onset * rate_hz))
        right = max(left + 1, int(math.ceil(offset * rate_hz)))
        confidence = float(np.mean(probability[left:right]))
        rows.append(
            {
                "event_id": f"{record_id}:{handler_id}:{index:07d}",
                "event_type": event_type,
                "record_id": record_id,
                "onset": onset,
                "offset": offset,
                "confidence": confidence,
                "channel": source_channel,
                "detector_version": "phase1c-neural-v1",
                "adapter_version": "phase1c-adapter-v1",
                "quality_status": "AVAILABLE",
                "evidence_status": "CANDIDATE_PROXY",
                "measurements": {
                    "mean_model_probability": confidence,
                    "threshold": float(threshold),
                    "output_rate_hz": rate_hz,
                },
                "source_model_sha256": checkpoint_sha256,
                "source_signal_pointer": {
                    "record_id": record_id,
                    "channel": source_channel,
                    "sampling_rate_hz": source_sampling_rate_hz,
                    "start_sample": int(round(onset * source_sampling_rate_hz)),
                    "end_sample": int(round(offset * source_sampling_rate_hz)),
                },
            }
        )
    return rows
