from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from .common import intervals_from_mask, rasterize, sha256, write_json
from .data import resolve_role_channels
from .edf import read_channel, read_channels, read_header, unit_to_microvolts


HANDLER_RATES = {
    "E05": 1,
    "E06": 10,
    "E07": 10,
    "E08": 1,
    "E09": 2,
    "E10": 2,
    "E11": 2,
    "E12": 1,
    "E13": 1,
}

EVENT_TYPES = {
    "E05": "alpha",
    "E06": "blink",
    "E07": "sem",
    "E08": "lamf",
    "E09": "low_emg",
    "E10": "normal_emg",
    "E11": "high_emg",
    "E12": "slowing",
    "E13": "mbm",
}

REQUIRED_ROLES = {
    "E05": ("o",),
    "E06": ("l", "r", "m"),
    "E07": ("l", "r"),
    "E08": ("c",),
    "E09": ("m",),
    "E10": ("m",),
    "E11": ("m",),
    "E12": ("c", "o"),
    "E13": ("f", "c", "m"),
}


def _v3_header(path: str | Path) -> dict[str, Any]:
    header = read_header(path)
    by_name = {signal.name: signal for signal in header.signals}
    roles = resolve_role_channels(header)
    channels = []
    for channel in roles.values():
        if channel not in by_name:
            continue
        signal = by_name[channel]
        factor = unit_to_microvolts(signal.unit)
        channels.append(
            {
                "name": signal.name,
                "unit": signal.unit,
                "sf": signal.sampling_rate,
                "min_uv": signal.physical_min * factor,
                "max_uv": signal.physical_max * factor,
                "prefilter": signal.prefilter,
                "polarity": 1,
            }
        )
    return {
        "record_id": Path(path).stem,
        "duration": header.duration_seconds,
        "channels": channels,
        "size": Path(path).stat().st_size,
        "roles": roles,
        "identity": "DREAMS_SOURCE_NIGHT_GROUP; person identity unavailable",
    }


def _v3_read_channel(path: str | Path, role: str, info: dict) -> tuple:
    name = info["roles"][role]
    if not name:
        return None, None
    return read_channel(path, name)


def _load_frozen_module(heritage_root: Path):
    source = str((heritage_root / "src").resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    return importlib.import_module("aasm_v2.events")


def _robust_score(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - median))
    return (values - median) / max(scale, 1e-6)


def _safe_negative_masks(path: Path, duration_seconds: int) -> dict[str, np.ndarray]:
    header = read_header(path)
    roles = resolve_role_channels(header)
    selected = [roles[key] for key in ("l", "r", "f", "c", "m")]
    if any(not channel for channel in selected):
        return {}
    raw = read_channels(
        path,
        selected,
        header=header,
    )
    eog = []
    for channel in (roles["l"], roles["r"]):
        values, rate = raw[channel]
        sos = butter(4, (0.3, 5.0), btype="bandpass", fs=rate, output="sos")
        eog.append(sosfiltfilt(sos, values.astype(np.float64)))
    eog_rate = raw[roles["l"]][1]
    frame = int(round(eog_rate / 10))
    usable = min(map(len, eog)) // frame * frame
    left = eog[0][:usable].reshape(-1, frame)
    right = eog[1][:usable].reshape(-1, frame)
    common_peak = np.max(np.abs((left + right) / 2), axis=1)
    difference_peak = np.max(np.abs((left - right) / 2), axis=1)
    one_second = {}
    framed = {}
    for role in ("f", "c", "m"):
        channel = roles[role]
        values, rate = raw[channel]
        width = int(round(rate))
        usable = min(len(values), int(duration_seconds * rate)) // width * width
        framed[role] = values[:usable].astype(np.float64).reshape(-1, width)
        one_second[role] = np.sqrt(np.mean(framed[role] ** 2, axis=1))
    common_length = min(map(len, one_second.values()))
    f_score = _robust_score(one_second["f"][:common_length])
    c_score = _robust_score(one_second["c"][:common_length])
    m_score = _robust_score(one_second["m"][:common_length])
    c_frames = framed["c"][:common_length]
    c_rate = raw[roles["c"]][1]
    frequency, power = welch(c_frames, fs=c_rate, nperseg=int(c_rate), axis=1)
    total = np.maximum(power[:, (frequency >= 0.3) & (frequency <= 35)].sum(axis=1), 1e-30)
    beta = power[:, (frequency >= 16) & (frequency <= 30)].sum(axis=1) / total
    movement_proxy = (
        ((f_score >= 2.0) | (c_score >= 2.0))
        & (m_score >= 1.5)
        & (beta >= 0.05)
    ).astype(np.uint8)
    movement_negative = (
        (f_score < 1.0) & (c_score < 1.0) & (m_score < 1.0)
    ).astype(np.uint8)
    return {
        "E06": (common_peak < 20.0).astype(np.uint8),
        "E07": (difference_peak < 15.0).astype(np.uint8),
        "E13": movement_negative,
        "E13_proxy_target": movement_proxy,
    }


def _role_valid(bank: dict, roles: tuple[str, ...], rate_hz: int, length: int) -> np.ndarray:
    epoch_valid = np.logical_and.reduce(
        [np.asarray(bank["goods"][role], dtype=bool) for role in roles]
    )
    repeated = np.repeat(epoch_valid, 30 * rate_hz)
    output = np.zeros(length, dtype=np.uint8)
    output[: min(length, len(repeated))] = repeated[:length].astype(np.uint8)
    return output


def generate_teacher_record(
    record: dict[str, Any], heritage_root: str | Path, cache_dir: str | Path
) -> dict[str, Any]:
    source = Path(record["file_path"])
    root = Path(heritage_root)
    config_path = root / "configs" / "frozen_detector_config.json"
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    events = _load_frozen_module(root)
    original_header = events.header
    original_reader = events.read_channel
    original_yasa = events.yasa_candidates
    events.header = _v3_header
    events.read_channel = _v3_read_channel
    events.yasa_candidates = lambda *args, **kwargs: ([], [])
    try:
        bank = events.detect(source, config)
    finally:
        events.header = original_header
        events.read_channel = original_reader
        events.yasa_candidates = original_yasa
    duration = int(bank["info"]["duration"])
    eligible_handlers = set(record["eligible_handlers"])
    target_events = [
        event
        for event in bank["events"]["enhanced"]
        if event["event_type"] in {
            EVENT_TYPES[handler_id] for handler_id in eligible_handlers
        }
    ]
    safe_negative = _safe_negative_masks(source, duration)
    arrays: dict[str, np.ndarray] = {}
    handler_rows = []
    for handler_id in sorted(eligible_handlers):
        event_type = EVENT_TYPES[handler_id]
        rate = HANDLER_RATES[handler_id]
        length = duration * rate
        intervals = [
            (float(event["onset"]), float(event["offset"]))
            for event in target_events
            if event["event_type"] == event_type
        ]
        target = rasterize(intervals, length, rate)
        label_source = "FROZEN_V3_WEAK_TEACHER"
        frozen_positive_samples = int(target.sum())
        if handler_id == "E13" and frozen_positive_samples == 0:
            proxy = safe_negative.get("E13_proxy_target", np.zeros(length, dtype=np.uint8))
            target = np.zeros(length, dtype=np.uint8)
            target[: min(length, len(proxy))] = proxy[:length]
            label_source = "PHYSIOLOGICAL_CONSTRAINT_WEAK_LABEL"
        label_event_count = (
            len(intervals)
            if label_source == "FROZEN_V3_WEAK_TEACHER"
            else len(
                intervals_from_mask(
                    target.astype(bool), rate, minimum_seconds=1 / rate
                )
            )
        )
        valid = _role_valid(bank, REQUIRED_ROLES[handler_id], rate, length)
        if handler_id in safe_negative:
            negative = safe_negative[handler_id]
            padded = np.zeros(length, dtype=np.uint8)
            padded[: min(length, len(negative))] = negative[:length]
            if handler_id == "E13":
                # Frozen MBM intentionally uses raw movement features even when
                # ordinary physiological-channel QC is rejected.
                valid = ((target > 0) | (padded > 0)).astype(np.uint8)
            else:
                valid = valid & ((target > 0) | (padded > 0))
        arrays[handler_id + "_target"] = target.astype(np.uint8)
        arrays[handler_id + "_valid"] = valid.astype(np.uint8)
        handler_rows.append(
            {
                "handler_id": handler_id,
                "event_type": event_type,
                "output_rate_hz": rate,
                "events": label_event_count,
                "positive_samples": int(np.sum(target & valid)),
                "valid_samples": int(valid.sum()),
                "availability": bank["availability"]["enhanced"].get(event_type),
                "negative_semantics": (
                    "PHYSIOLOGICAL_MARGIN_NEGATIVE"
                    if handler_id in safe_negative
                    else "WEAK_TEACHER_PROXY_NEGATIVE"
                ),
                "label_source": label_source,
                "frozen_v3_positive_samples_before_fallback": frozen_positive_samples,
            }
        )
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache / f"{record['recording_id']}.npz", **arrays)
    bank_path = cache / f"{record['recording_id']}.teacher.json"
    write_json(
        bank_path,
        {
            "schema_version": 1,
            "teacher_schema_version": 3 if "E13" in eligible_handlers else 1,
            "recording_id": record["recording_id"],
            "source_edf": str(source.resolve()),
            "source_edf_sha256": sha256(source),
            "label_provenance": "WEAK_TEACHER_LABEL",
            "teacher": (
                "Frozen V3 enhanced detector target subset; E13 uses an explicitly "
                "isolated physiology-constraint proxy only when Frozen V3 has zero positives"
            ),
            "teacher_config": str(config_path.resolve()),
            "teacher_config_sha256": sha256(config_path),
            "teacher_source_sha256": sha256(
                root / "src" / "aasm_v2" / "events" / "__init__.py"
            ),
            "unrelated_yasa_detectors_skipped": True,
            "stage_or_hypnogram_access": False,
            "handlers": handler_rows,
            "availability": {
                key: bank["availability"]["enhanced"].get(EVENT_TYPES[key])
                for key in eligible_handlers
            },
            "background_diagnostics": bank["diagnostics"]["backgrounds"],
            "blink_artifact_rejected": bank["diagnostics"].get(
                "blink_artifact_rejected"
            ),
            "limitations": [
                "DREAMS EMG1 is consistent with the published submental channel, but the EDF label does not spell out chin",
                "weak-teacher absence is not an expert-validated clinical negative",
                "DREAMS montage is adapted without modifying frozen V3 source",
                "E13 uses a separately identified physiological-constraint proxy only when the unchanged Frozen V3 montage transfer yields no positive sample",
            ],
        },
    )
    return {
        "recording_id": record["recording_id"],
        "cache": str((cache / f"{record['recording_id']}.npz").resolve()),
        "teacher_manifest": str(bank_path.resolve()),
        "handlers": handler_rows,
    }


def label_registry(heritage_root: str | Path) -> dict[str, Any]:
    root = Path(heritage_root)
    source = root / "src" / "aasm_v2" / "events" / "__init__.py"
    config = root / "configs" / "frozen_detector_config.json"
    return {
        "schema_version": 1,
        "label_source_level": "MIXED_LEVEL_C_WEAK_TARGETS",
        "expert_target_annotations_available": False,
        "stage_annotations_used": False,
        "teacher_entrypoint": "aasm_v2.events.detect target subset",
        "teacher_source": str(source.resolve()),
        "teacher_source_sha256": sha256(source),
        "teacher_config": str(config.resolve()),
        "teacher_config_sha256": sha256(config),
        "handler_rates_hz": HANDLER_RATES,
        "event_types": EVENT_TYPES,
        "emg_semantics": "multi-label: high_emg is a subset of normal_emg under frozen V3",
        "test_interpretation": (
            "E05-E12 frozen-teacher agreement and E13 physiology-constraint proxy "
            "agreement only; not independent expert accuracy"
        ),
        "per_handler_label_source": {
            **{handler: "FROZEN_V3_WEAK_TEACHER" for handler in EVENT_TYPES if handler != "E13"},
            "E13": "PHYSIOLOGICAL_CONSTRAINT_WEAK_LABEL_WITH_FROZEN_V3_ZERO_POSITIVE_RETAINED",
        },
        "e13_proxy_definition": (
            "1-second positive when (frontal robust RMS z >= 2.0 or central robust RMS "
            "z >= 2.0) and EMG robust RMS z >= 1.5 and central 16-30 Hz power fraction "
            ">= 0.05; robust statistics are fit within each signal record"
        ),
        "e13_proxy_selection_scope": (
            "recovered after Frozen V3 yielded zero positives; feasibility checked on "
            "train record kc_excerpt1 only, without validation/test or stage labels"
        ),
    }
