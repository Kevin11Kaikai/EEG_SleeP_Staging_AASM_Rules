from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt
import torch

from experiments.aasm_full_detector_integration.adapters.edf_signal import edf_layout, read_edf_channel_uv
from experiments.multi_event_dl_audit_v3.sssm.inference_adapter import LABEL_LONG, load_frozen_model
from experiments.v4_handler_validation.phase1c_handlers.inference import build_event_rows, infer_full_record
from experiments.v4_handler_validation.phase1c_handlers.models import FAMILY_SPECS, build_model
from experiments.v4_handler_validation.sem_final_upgrade.handler.adapter import E07SEMNeuralHandler

from .config import (
    ALLOWLIST_MANIFEST, E07_CHECKPOINT, E07_SHA256, EVENT_TYPE, EXPECTED_MODEL_SHA,
    FAMILIES, HANDLERS, PHASE1C_CONFIG, PRECOMPUTED, RECORDS, SSSM_CHECKPOINT,
    SSSM_SHA256,
)
from .io_utils import canonical_hash, read_json, sha256, write_gzip_json, write_json


BANDS = {"eeg": (0.3, 35.0), "eog": (0.3, 15.0), "emg": (10.0, 45.0)}
SCALES = {"eeg": 100.0, "eog": 200.0, "emg": 50.0}


def _load_allowlist() -> dict[str, dict[str, Any]]:
    payload = read_json(ALLOWLIST_MANIFEST)
    rows = {row["record_id"]: row for row in payload["records"]}
    if tuple(sorted(rows)) != RECORDS:
        raise PermissionError("HMC manifest differs from the configured record allowlist")
    for record, row in rows.items():
        if Path(row["edf_path"]).name != f"{record}.edf":
            raise PermissionError(f"unexpected signal file for {record}")
        if sha256(row["edf_path"]) != row["edf_sha256"]:
            raise RuntimeError(f"signal hash mismatch for {record}")
    return rows


def _prepare_channel(path: Path, channel: str, kind: str, target_hz: int = 100) -> tuple[np.ndarray, float]:
    values, rate, _ = read_edf_channel_uv(path, channel)
    sos = butter(4, BANDS[kind], btype="bandpass", fs=rate, output="sos")
    filtered = sosfiltfilt(sos, values.astype(np.float64, copy=False))
    if abs(rate - target_hz) > 1e-6:
        filtered = resample_poly(filtered, target_hz, int(round(rate)))
    output = np.clip(filtered / SCALES[kind], -10.0, 10.0).astype(np.float32)
    if not np.isfinite(output).all():
        raise FloatingPointError(f"non-finite preprocessed channel: {channel}")
    return output, rate


def _canonical_validate(events: list[dict[str, Any]], handler: str, record: str, duration: float) -> None:
    expected = EVENT_TYPE[handler]
    identifiers: set[str] = set()
    for event in events:
        if event.get("event_type") != expected:
            raise ValueError(f"{handler}/{record}: {event.get('event_type')} != {expected}")
        if event.get("event_id") in identifiers:
            raise ValueError(f"{handler}/{record}: duplicate event id")
        identifiers.add(event["event_id"])
        onset, offset = float(event["onset"]), float(event["offset"])
        if not (math.isfinite(onset) and math.isfinite(offset) and 0 <= onset < offset <= duration + 1e-6):
            raise ValueError(f"{handler}/{record}: invalid interval {onset}-{offset}/{duration}")
        if "source_model_sha256" not in event:
            event["source_model_sha256"] = event.get("measurements", {}).get("source_model_sha256")
        if event.get("source_model_sha256") != EXPECTED_MODEL_SHA[handler]:
            raise ValueError(f"{handler}/{record}: model identity mismatch")
        if "source_signal_pointer" not in event:
            event["source_signal_pointer"] = event.get("measurements", {}).get("source_signal_pointer")
        if not event.get("source_signal_pointer"):
            raise ValueError(f"{handler}/{record}: missing signal provenance")
        event.setdefault("parameters", {})
        event.setdefault("adapter_version", event.get("measurements", {}).get("adapter_version", "heritage-adapter"))
        event.setdefault("quality_status", "PARTIALLY_SUPPORTED")
        event.setdefault("evidence_status", "CANDIDATE_PROXY")
        event.setdefault("measurements", {})


def _source_payload(handler: str, record: str, events: list[dict[str, Any]], duration: float,
                    *, status: str, model: str, channels: list[str], provenance: dict[str, Any],
                    joint_valid_mask: list[bool] | None = None) -> dict[str, Any]:
    _canonical_validate(events, handler, record, duration)
    payload = {
        "schema_version": 1, "record_id": record, "handler_id": handler,
        "event_type": EVENT_TYPE[handler], "model": model,
        "model_sha256": EXPECTED_MODEL_SHA[handler], "channels": channels,
        "status": status,
        "overlay_availability": "AVAILABLE" if handler in {"E05", "SP"} else "PARTIALLY_SUPPORTED",
        "valid_signal_seconds": duration, "event_count": len(events), "events": events,
        "label_blind_inference": True, "stage_reference_read": False,
        "montage_status": "MONTAGE_TRANSFER_UNVALIDATED", "provenance": provenance,
    }
    if joint_valid_mask is not None:
        payload["joint_valid_mask"] = [bool(x) for x in joint_valid_mask]
    return payload


def _load_precomputed(handler: str, record: str, duration: float) -> dict[str, Any]:
    source = Path(str(PRECOMPUTED[handler]).format(record=record))
    payload = read_json(source)
    events = payload["events"]
    for event in events:
        event.setdefault("source_model_sha256", payload.get("model_sha256") or event.get("measurements", {}).get("source_model_sha256"))
        event.setdefault("source_signal_pointer", event.get("measurements", {}).get("source_signal_pointer"))
        event.setdefault("adapter_version", payload.get("adapter_version", "heritage-adapter"))
    return _source_payload(
        handler, record, events, duration, status=payload.get("status", "PARTIALLY_SUPPORTED"),
        model=payload.get("detector", handler),
        channels=list(payload.get("channels") or ([payload["channel"]] if payload.get("channel") else [])),
        provenance={"heritage_path": str(source.resolve()), "heritage_sha256": sha256(source)},
        joint_valid_mask=payload.get("joint_valid_mask") if handler == "AR" else None,
    )


def _family_models(device: torch.device) -> dict[str, torch.nn.Module]:
    output = {}
    for name, config in FAMILIES.items():
        if sha256(config.checkpoint) != config.expected_sha256:
            raise RuntimeError(f"checkpoint mismatch: {name}")
        checkpoint = torch.load(config.checkpoint, map_location="cpu", weights_only=True)
        model = build_model(name)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        output[name] = model.to(device).eval()
    return output


def _run_family(record: str, row: dict[str, Any], family_name: str, model: torch.nn.Module,
                device: torch.device, raw_root: Path) -> dict[str, dict[str, Any]]:
    config = FAMILIES[family_name]
    source = Path(row["edf_path"])
    values, native_rates = [], []
    for channel, kind in zip(config.channels, config.kinds):
        value, native_rate = _prepare_channel(source, channel, kind)
        values.append(value)
        native_rates.append(native_rate)
    length = min(map(len, values))
    signal = np.stack([value[:length] for value in values])
    before = sha256(config.checkpoint)
    prediction = infer_full_record(model, signal, family_name, config.window_seconds, device,
                                   batch_size=config.batch_size)
    if before != sha256(config.checkpoint):
        raise RuntimeError(f"checkpoint mutated: {family_name}")
    raw_root.mkdir(parents=True, exist_ok=True)
    raw_path = raw_root / f"{record}_{family_name}.npz"
    np.savez_compressed(raw_path, probability=prediction.astype(np.float16),
                        temporal_resolution_seconds=np.asarray([1 / FAMILY_SPECS[family_name].output_rate_hz]),
                        handlers=np.asarray(FAMILY_SPECS[family_name].output_heads))
    outputs = {}
    duration = float(row["duration_s"])
    for handler in config.handlers:
        threshold, output_rate, eventization = PHASE1C_CONFIG[handler]
        head = FAMILY_SPECS[family_name].output_heads.index(handler)
        events = build_event_rows(record, handler, EVENT_TYPE[handler], prediction[head], threshold,
                                  output_rate, eventization, config.expected_sha256,
                                  "+".join(config.channels), native_rates[0])
        for event in events:
            event["parameters"] = {"threshold": threshold, **eventization,
                                   "montage_status": "MONTAGE_TRANSFER_UNVALIDATED"}
            event["source_model_sha256"] = config.expected_sha256
            event["source_signal_pointer"]["channels"] = list(config.channels)
            event["source_signal_pointer"]["edf_path"] = str(source.resolve())
        outputs[handler] = _source_payload(
            handler, record, events, duration, status="AVAILABLE" if events else "NO_CANDIDATE",
            model=f"Phase1C {family_name}", channels=list(config.channels),
            provenance={"checkpoint": str(config.checkpoint.resolve()),
                        "checkpoint_sha256": config.expected_sha256,
                        "raw_output": str(raw_path.resolve()),
                        "shared_forward_handlers": list(config.handlers)},
        )
    return outputs


def _sssm_probabilities(model: torch.nn.Module, signal_uv: np.ndarray, device: torch.device) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(signal_uv, 300)
    output = np.empty((len(windows), 2), dtype=np.float32)
    wanted = (LABEL_LONG.index("Slow wave"), LABEL_LONG.index("Vertex Sharp"))
    batch_size = 16384 if device.type == "cuda" else 2048
    model.eval()
    with torch.inference_mode():
        for left in range(0, len(windows), batch_size):
            batch = np.ascontiguousarray(windows[left:left + batch_size, None, :])
            logits, _ = model(torch.from_numpy(batch).to(device))
            probability = torch.softmax(logits, dim=1)[:, wanted]
            output[left:left + len(batch)] = probability.cpu().numpy()
    if not np.isfinite(output).all():
        raise FloatingPointError("SSSM emitted non-finite probabilities")
    return output


def _sssm_events(probability: np.ndarray, class_index: int, handler: str, record: str,
                 input_samples: int, native_rate: float, source: Path) -> list[dict[str, Any]]:
    positive = np.flatnonzero(probability[:, class_index] > 0.5)
    runs = np.split(positive, np.flatnonzero(np.diff(positive) != 1) + 1) if positive.size else []
    events = []
    for index, run in enumerate(runs):
        start, end = int(run[0] + 149), int(run[-1] + 151)
        if start <= 149:
            start = 0
        if end >= input_samples - 151:
            end = input_samples
        onset, offset = start / 100.0, end / 100.0
        events.append({
            "event_id": f"{record}:{handler}:sssm:{index:07d}", "event_type": EVENT_TYPE[handler],
            "record_id": record, "onset": onset, "offset": offset, "channel": "EEG C4-M1",
            "detector_version": "SSSM-0.1.0-frozen",
            "parameters": {"threshold": 0.5, "window_samples": 300, "stride_samples": 1,
                           "duration_filter": None, "montage_status": "MONTAGE_TRANSFER_UNVALIDATED"},
            "quality_status": "PARTIALLY_SUPPORTED", "evidence_status": "CANDIDATE_PROXY",
            "measurements": {"mean_model_probability": float(probability[run, class_index].mean())},
            "source_model_sha256": SSSM_SHA256,
            "source_signal_pointer": {"record_id": record, "edf_path": str(source.resolve()),
                                      "channel": "EEG C4-M1", "sampling_rate_hz": native_rate,
                                      "start_sample": int(round(onset * native_rate)),
                                      "end_sample": int(round(offset * native_rate))},
            "adapter_version": "phase2-sssm-canonical-v1",
        })
    return events


def _run_sssm(record: str, row: dict[str, Any], loaded: Any, device: torch.device,
              raw_root: Path) -> dict[str, dict[str, Any]]:
    source = Path(row["edf_path"])
    values, native_rate, _ = read_edf_channel_uv(source, "EEG C4-M1")
    sos = butter(4, (0.7, 45.0), btype="bandpass", fs=native_rate, output="sos")
    signal = resample_poly(sosfiltfilt(sos, values.astype(np.float64, copy=False)),
                           100, int(round(native_rate))).astype(np.float32)
    probability = _sssm_probabilities(loaded.model, signal, device)
    raw_root.mkdir(parents=True, exist_ok=True)
    raw_path = raw_root / f"{record}_sssm_targets.npz"
    np.savez_compressed(raw_path, probability=probability.astype(np.float16),
                        labels=np.asarray(["Slow wave", "Vertex Sharp"]), stride_samples=np.asarray([1]))
    duration = float(row["duration_s"])
    output = {}
    for class_index, handler in enumerate(("E01", "E03")):
        events = _sssm_events(probability, class_index, handler, record, len(signal), native_rate, source)
        output[handler] = _source_payload(
            handler, record, events, duration, status="AVAILABLE" if events else "NO_CANDIDATE",
            model="SSSM 0.1.0", channels=["EEG C4-M1"],
            provenance={"checkpoint": str(SSSM_CHECKPOINT.resolve()), "checkpoint_sha256": SSSM_SHA256,
                        "raw_output": str(raw_path.resolve()), "shared_forward_handlers": ["E01", "E03"]},
        )
    return output


def _run_e07(record: str, row: dict[str, Any], device: torch.device, raw_root: Path) -> dict[str, Any]:
    source = Path(row["edf_path"])
    left, rate_left, _ = read_edf_channel_uv(source, "EOG E1-M2")
    right, rate_right, _ = read_edf_channel_uv(source, "EOG E2-M2")
    if rate_left != rate_right:
        raise ValueError("E07 bilateral EOG rates differ")
    handler = E07SEMNeuralHandler({
        "checkpoint": str(E07_CHECKPOINT), "architecture": "inherited_tcn", "threshold": 0.9,
        "eventization": {"minimum_seconds": 1.0, "maximum_seconds": 5.0, "merge_gap_seconds": 0.3},
        "window_seconds": 20, "inference_stride_seconds": 10, "batch_size": 64,
        "device": str(device), "expected_channel_roles": ["left_eog", "right_eog"],
        "expected_reference": "M2", "expected_polarity": "as_recorded",
    })
    result = handler.detect(np.stack([left, right]), {
        "record_id": record, "channels": ["EOG E1-M2", "EOG E2-M2"],
        "channel_roles": ["left_eog", "right_eog"], "reference": "M2",
        "polarity": "as_recorded", "sampling_rate_hz": rate_left, "signal_unit": "uV",
    })
    probability = np.asarray(result.get("probability", []), dtype=np.float32)
    raw_root.mkdir(parents=True, exist_ok=True)
    raw_path = raw_root / f"{record}_e07.npz"
    np.savez_compressed(raw_path, probability=probability.astype(np.float16),
                        valid_mask=np.asarray(result.get("valid_mask", []), dtype=np.uint8))
    events = result["events"]
    for event in events:
        event["parameters"]["montage_status"] = "MONTAGE_TRANSFER_UNVALIDATED"
        event["source_signal_pointer"]["edf_path"] = str(source.resolve())
    return _source_payload(
        "E07", record, events, float(row["duration_s"]), status=result["status"],
        model="Phase1D inherited_tcn", channels=["EOG E1-M2", "EOG E2-M2"],
        provenance={"checkpoint": str(E07_CHECKPOINT.resolve()), "checkpoint_sha256": E07_SHA256,
                    "raw_output": str(raw_path.resolve())},
    )


def generate_source_banks(run_root: Path) -> dict[str, Any]:
    """Generate and seal all signal-only neural source banks before label access."""
    started = time.perf_counter()
    allowlist = _load_allowlist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = _family_models(device)
    if sha256(SSSM_CHECKPOINT) != SSSM_SHA256 or sha256(E07_CHECKPOINT) != E07_SHA256:
        raise RuntimeError("SSSM or E07 checkpoint identity mismatch")
    sssm = load_frozen_model(SSSM_CHECKPOINT, str(device))
    source_root = run_root / "source_banks"
    raw_root = run_root / "neural_inference" / "raw_outputs"
    manifest_rows = []
    for record in RECORDS:
        row = allowlist[record]
        names = {channel["name"] for channel in edf_layout(row["edf_path"])["channels"]}
        required = {"EEG F4-M1", "EEG C4-M1", "EEG O2-M1", "EMG chin", "EOG E1-M2", "EOG E2-M2"}
        if not required <= names:
            raise ValueError(f"{record}: missing channels {sorted(required - names)}")
        outputs: dict[str, dict[str, Any]] = {}
        outputs.update(_run_sssm(record, row, sssm, device, raw_root))
        for family_name, model in models.items():
            outputs.update(_run_family(record, row, family_name, model, device, raw_root))
        outputs["E07"] = _run_e07(record, row, device, raw_root)
        for handler in PRECOMPUTED:
            outputs[handler] = _load_precomputed(handler, record, float(row["duration_s"]))
        if set(outputs) != set(HANDLERS):
            raise AssertionError(f"{record}: missing source handlers {set(HANDLERS) - set(outputs)}")
        for handler in HANDLERS:
            path = source_root / handler / f"{record}.json.gz"
            write_gzip_json(path, outputs[handler])
            manifest_rows.append({
                "record_id": record, "handler_id": handler, "status": outputs[handler]["status"],
                "event_count": outputs[handler]["event_count"], "model_sha256": outputs[handler]["model_sha256"],
                "source_bank": str(path.resolve()), "source_bank_sha256": sha256(path),
                "valid_signal_seconds": outputs[handler]["valid_signal_seconds"],
                "montage_status": outputs[handler]["montage_status"],
            })
        combined = run_root / "neural_event_banks" / record / "all_handlers.json.gz"
        write_gzip_json(combined, {handler: outputs[handler] for handler in HANDLERS})
    seal_items = [(row["record_id"], row["handler_id"], row["source_bank_sha256"]) for row in manifest_rows]
    seal = {
        "status": "SEALED", "records": len(RECORDS), "handlers": len(HANDLERS),
        "source_bank_count": len(manifest_rows), "source_bank_key_hash": canonical_hash(seal_items),
        "stage_reference_read": False,
        "label_blind_worker_inputs": ["raw_psg", "frozen_checkpoints", "preprocessing", "signal_metadata", "qc", "eventization"],
        "label_blind_worker_forbidden_inputs": ["stage_reference", "hypnogram", "stage_accuracy", "confusion_matrix"],
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(run_root / "neural_inference" / "model_inference_manifest.json", {"rows": manifest_rows, **seal})
    write_json(run_root / "neural_inference" / "prediction_seal.json", seal)
    return seal
