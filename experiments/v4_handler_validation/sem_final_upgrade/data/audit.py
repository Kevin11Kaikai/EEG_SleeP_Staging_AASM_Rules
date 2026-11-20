from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import find_peaks, peak_widths

from experiments.v4_handler_validation.phase1c_handlers.edf import (
    read_channels,
    read_header,
    unit_to_microvolts,
)

from ..common import rasterize, sha256


@dataclass
class RecordData:
    record_id: str
    dataset_id: str
    split_role: str
    group_id: str
    signal: np.ndarray
    signal_qc: np.ndarray
    target: np.ndarray
    scoring_mask: np.ndarray
    blink_mask: np.ndarray
    rem_mask: np.ndarray
    hard_negative_mask: np.ndarray
    teacher_intervals: list[tuple[float, float]]
    teacher_details: list[dict[str, Any]]
    blink_intervals: list[tuple[float, float]]
    metadata: dict[str, Any]


def teacher_fidelity_scoring_mask(signal_qc: np.ndarray) -> np.ndarray:
    """Score the frozen teacher on every QC-valid sample.

    The teacher is executed over the complete record, so its absence is the
    weak-label negative for teacher-fidelity evaluation.  A mask constructed
    from the teacher target would leak reference locations into eventization
    and can hide label-blind false proposals.
    """
    return np.asarray(signal_qc, dtype=np.uint8).copy()


def _frozen_functions(config: dict[str, Any]):
    root = Path(config["frozen_heritage_root"])
    source = str((root / "src").resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from aasm_v2.events.morphology import ocular_candidates
    from aasm_v2.preprocessing import band_filter, prepare, quality
    from aasm_v2.temporal import runs

    with (root / "configs" / "frozen_detector_config.json").open(
        "r", encoding="utf-8"
    ) as stream:
        frozen_config = json.load(stream)
    return quality, prepare, ocular_candidates, band_filter, runs, frozen_config


def _rem_intervals(record: dict[str, Any], config: dict[str, Any]) -> list[tuple[float, float]]:
    if record["source_task"] != "rem":
        return []
    manifest_path = Path(config["historical_tournament"]) / "data_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    excerpt = record["record_id"].replace("rem_excerpt", "excerpt")
    row = manifest["records"]["rem"][excerpt]
    return [
        (float(event["onset_s"]), float(event["offset_s"]))
        for event in row.get("expert1_events", [])
        if float(event["offset_s"]) > float(event["onset_s"])
    ]


def build_record_cache(
    record: dict[str, Any], config: dict[str, Any]
) -> RecordData:
    quality, prepare, ocular_candidates, band_filter, runs, frozen = _frozen_functions(config)
    path = Path(record["file_path"])
    header = read_header(path)
    channels = (record["left_channel"], record["right_channel"])
    raw = read_channels(path, channels, header=header)
    goods = []
    teacher_signals = []
    neural_signals = []
    slow_signals = []
    for channel in channels:
        values, source_rate = raw[channel]
        detail = next(item for item in header.signals if item.name == channel)
        scale = unit_to_microvolts(detail.unit)
        good = quality(
            values,
            source_rate,
            "eog",
            (detail.physical_min * scale, detail.physical_max * scale),
            frozen["qc"],
        )
        goods.append(good)
        teacher_signals.append(
            prepare(
                values,
                source_rate,
                frozen["sampling_hz"],
                good,
                frozen["filters"]["eog"],
                frozen["filters"]["order"],
            )
        )
        neural_signals.append(
            np.clip(
                prepare(values, source_rate, 100, good, (0.3, 15.0), 4) / 200.0,
                -10.0,
                10.0,
            ).astype(np.float32)
        )
        slow_signals.append(prepare(values, source_rate, 100, good, (0.3, 5.0), 4))
    joint_good = np.logical_and(goods[0], goods[1])
    ocular = ocular_candidates(
        teacher_signals[0],
        teacher_signals[1],
        frozen["sampling_hz"],
        joint_good,
        frozen["enhanced"],
    )
    sem_intervals = [(float(a), float(b)) for a, b, *_ in ocular["sem"]]
    sem_details = [dict(row[2]) for row in ocular["sem"]]
    gate_counts = {
        "amplitude_prominence_peaks": 0,
        "duration_1_to_5s_pass": 0,
        "negative_correlation_pass_within_duration": 0,
        "initial_deflection_pass_within_duration": 0,
        "all_gates_pass": 0,
    }
    sem_cfg = frozen["enhanced"]["sem"]
    for lo, hi in runs(joint_good, 1 / 30):
        origin = round(lo * frozen["sampling_hz"])
        left = band_filter(
            teacher_signals[0][origin : round(hi * frozen["sampling_hz"])],
            frozen["sampling_hz"],
            (0.3, 5.0),
        )
        right = band_filter(
            teacher_signals[1][origin : round(hi * frozen["sampling_hz"])],
            frozen["sampling_hz"],
            (0.3, 5.0),
        )
        difference_signal = (left - right) / 2
        peaks, _ = find_peaks(
            np.abs(difference_signal),
            height=sem_cfg["minimum_uv"],
            prominence=sem_cfg["minimum_uv"] / 2,
            distance=round(sem_cfg["duration_s"][0] * frozen["sampling_hz"]),
        )
        gate_counts["amplitude_prominence_peaks"] += len(peaks)
        if not len(peaks):
            continue
        widths = peak_widths(np.abs(difference_signal), peaks, rel_height=0.9)
        for index, peak in enumerate(peaks):
            a = int(np.floor(widths[2][index]))
            b = int(np.ceil(widths[3][index])) + 1
            duration = (b - a) / frozen["sampling_hz"]
            if not sem_cfg["duration_s"][0] <= duration <= sem_cfg["duration_s"][1]:
                continue
            gate_counts["duration_1_to_5s_pass"] += 1
            correlation = float(np.corrcoef(left[a:b], right[a:b])[0, 1])
            correlation_pass = bool(
                np.isfinite(correlation)
                and correlation <= sem_cfg["negative_correlation"]
            )
            initial_pass = bool(
                (peak - widths[2][index]) / frozen["sampling_hz"]
                >= sem_cfg["minimum_initial_deflection_s"]
            )
            gate_counts["negative_correlation_pass_within_duration"] += int(
                correlation_pass
            )
            gate_counts["initial_deflection_pass_within_duration"] += int(initial_pass)
            gate_counts["all_gates_pass"] += int(correlation_pass and initial_pass)
    blink_intervals = [(float(a), float(b)) for a, b, *_ in ocular["blink"]]
    duration_seconds = int(len(joint_good) * 30)
    output_length = duration_seconds * 10
    target = rasterize(sem_intervals, output_length, 10)
    blink = rasterize(blink_intervals, output_length, 10)
    rem_intervals = _rem_intervals(record, config)
    rem = rasterize(rem_intervals, output_length, 10)
    signal_qc = np.repeat(joint_good.astype(np.uint8), 300)[:output_length]
    scoring = teacher_fidelity_scoring_mask(signal_qc)
    hard_negative = signal_qc & ((blink > 0) | (rem > 0)) & (target == 0)
    input_length = duration_seconds * 100
    signal = np.stack([value[:input_length] for value in neural_signals]).astype(np.float32)
    metadata = {
        **record,
        "source_edf_sha256": sha256(path),
        "duration_seconds": duration_seconds,
        "source_sampling_rates_hz": [float(raw[name][1]) for name in channels],
        "neural_sampling_rate_hz": 100,
        "teacher_output_rate_hz": 10,
        "signal_unit": "uV before fixed /200 scaling",
        "good_epochs": int(joint_good.sum()),
        "total_complete_epochs": int(len(joint_good)),
        "sem_events": len(sem_intervals),
        "sem_positive_samples_10hz": int(target.sum()),
        "sem_positive_seconds": float(target.sum() / 10),
        "teacher_gate_counts": gate_counts,
        "blink_events": len(blink_intervals),
        "rem_hard_negative_events": len(rem_intervals),
        "signal_qc_seconds": float(signal_qc.sum() / 10),
        "scored_seconds": float(scoring.sum() / 10),
        "unlabeled_seconds": 0.0,
        "invalid_seconds": float((~signal_qc.astype(bool)).sum() / 10),
    }
    return RecordData(
        record_id=record["record_id"],
        dataset_id=record["dataset_id"],
        split_role=record["split_role"],
        group_id=record["group_id"],
        signal=signal,
        signal_qc=signal_qc.astype(np.uint8),
        target=target.astype(np.uint8),
        scoring_mask=scoring.astype(np.uint8),
        blink_mask=blink.astype(np.uint8),
        rem_mask=rem.astype(np.uint8),
        hard_negative_mask=hard_negative.astype(np.uint8),
        teacher_intervals=sem_intervals,
        teacher_details=sem_details,
        blink_intervals=blink_intervals,
        metadata=metadata,
    )


def teacher_gate_audit(record: RecordData) -> dict[str, Any]:
    gates = record.metadata["teacher_gate_counts"]
    return {
        "record_id": record.record_id,
        **gates,
        "accepted_exact_frozen_teacher_events": len(record.teacher_intervals),
        "note": "Exact replay of the unchanged frozen 128-Hz SEM teacher gates",
    }
