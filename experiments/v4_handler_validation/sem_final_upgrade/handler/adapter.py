from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt

from ..common import eventize, parameter_hash, sha256
from ..evaluation.challenge import infer_full_record
from ..training.checkpoint import strict_load
from .event_contract import validate_events


class InputContractError(ValueError):
    pass


class E07SEMNeuralHandler:
    detector_version = "e07-sem-tcn-phase1d-v1"
    adapter_version = "e07-sem-adapter-phase1d-v1"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.device = torch.device(
            config.get("device")
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model, self.payload = strict_load(
            config["checkpoint"], config["architecture"], self.device
        )
        self.checkpoint_sha256 = sha256(config["checkpoint"])
        self.parameter_hash = parameter_hash(self.model)

    def validate_input(self, signal_metadata: dict[str, Any]) -> None:
        channels = signal_metadata.get("channels")
        if not isinstance(channels, (list, tuple)) or len(channels) != 2:
            raise InputContractError("MISSING_CHANNEL: bilateral EOG is required")
        roles = signal_metadata.get("channel_roles")
        expected_roles = self.config.get(
            "expected_channel_roles", ["left_eog", "right_eog"]
        )
        if list(roles or []) != list(expected_roles):
            raise InputContractError(
                "INVALID_SIGNAL: channel order must be left_eog then right_eog"
            )
        expected_reference = self.config.get("expected_reference")
        if expected_reference and signal_metadata.get("reference") != expected_reference:
            raise InputContractError("INVALID_SIGNAL: EOG reference contract mismatch")
        expected_polarity = self.config.get("expected_polarity", "as_recorded")
        if signal_metadata.get("polarity") != expected_polarity:
            raise InputContractError("INVALID_SIGNAL: EOG polarity contract mismatch")
        rate = float(signal_metadata.get("sampling_rate_hz", 0))
        if rate <= 30:
            raise InputContractError(
                "INVALID_SIGNAL: EOG sampling rate must exceed 30 Hz for 15 Hz filtering"
            )
        unit = str(signal_metadata.get("signal_unit", "")).lower().replace("µ", "u")
        if unit not in {"uv", "mv", "v"}:
            raise InputContractError("INVALID_SIGNAL: unsupported EOG unit")

    def preprocess(
        self, signal: np.ndarray, signal_metadata: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        self.validate_input(signal_metadata)
        values = np.asarray(signal, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != 2:
            raise InputContractError("MISSING_CHANNEL: expected a 2 x time EOG array")
        if values.shape[1] < int(float(signal_metadata["sampling_rate_hz"]) * 20):
            raise InputContractError("INVALID_SIGNAL: less than 20 seconds of EOG")
        if not np.isfinite(values).all():
            raise InputContractError("INVALID_SIGNAL: non-finite EOG samples")
        unit = str(signal_metadata["signal_unit"]).lower().replace("µ", "u")
        factor = {"uv": 1.0, "mv": 1_000.0, "v": 1_000_000.0}[unit]
        values *= factor
        source_rate = float(signal_metadata["sampling_rate_hz"])
        width = int(round(30 * source_rate))
        epoch_count = values.shape[1] // width
        if epoch_count == 0:
            raise InputContractError("INVALID_SIGNAL: no complete QC epoch")
        chunks = values[:, : epoch_count * width].reshape(2, epoch_count, width)
        finite = np.isfinite(chunks).all(axis=(0, 2))
        ptp = np.ptp(chunks, axis=2).min(axis=0)
        maximum = np.abs(chunks).max(axis=(0, 2))
        repeated = (np.diff(chunks, axis=2) == 0).mean(axis=2).max(axis=0)
        good_epochs = finite & (ptp >= 0.5) & (maximum <= 2000) & (repeated <= 0.2)
        prepared = []
        for channel in values:
            sos = butter(4, (0.3, 15.0), btype="bandpass", fs=source_rate, output="sos")
            filtered = sosfiltfilt(sos, channel[: epoch_count * width])
            if abs(source_rate - 100.0) > 1e-6:
                filtered = resample_poly(filtered, 100, int(round(source_rate)))
            prepared.append(np.clip(filtered / 200.0, -10.0, 10.0).astype(np.float32))
        length = min(map(len, prepared))
        neural = np.stack([value[:length] for value in prepared])
        duration = neural.shape[-1] // 100
        neural = neural[:, : duration * 100]
        signal_qc = np.repeat(good_epochs.astype(np.uint8), 300)[: duration * 10]
        return neural, signal_qc

    def detect(
        self,
        signal: np.ndarray,
        signal_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            neural, signal_qc = self.preprocess(signal, signal_metadata)
        except InputContractError as exc:
            return {
                "status": str(exc).split(":", 1)[0],
                "error": str(exc),
                "events": [],
                "probability": [],
                "valid_mask": [],
            }
        try:
            probability = infer_full_record(
                self.model,
                neural,
                self.device,
                int(self.config["window_seconds"]),
                int(self.config["inference_stride_seconds"]),
                int(self.config["batch_size"]),
            )
            intervals = eventize(
                probability,
                signal_qc,
                float(self.config["threshold"]),
                10,
                float(self.config["eventization"]["minimum_seconds"]),
                float(self.config["eventization"]["maximum_seconds"]),
                float(self.config["eventization"]["merge_gap_seconds"]),
            )
            events = self.postprocess(
                intervals, probability, signal_metadata, neural.shape[-1] / 100
            )
        except Exception as exc:
            return {
                "status": "MODEL_FAILURE",
                "error": f"MODEL_FAILURE: {type(exc).__name__}: {exc}",
                "events": [],
                "probability": [],
                "valid_mask": signal_qc,
            }
        return {
            "status": "AVAILABLE" if events else "NO_CANDIDATE",
            "events": events,
            "probability": probability,
            "valid_mask": signal_qc,
            "probability_semantics": "sigmoid_model_score_not_clinically_calibrated",
            "temporal_resolution_seconds": 0.1,
        }

    def postprocess(
        self,
        intervals: list[tuple[float, float]],
        probability: np.ndarray,
        metadata: dict[str, Any],
        duration_seconds: float,
    ) -> list[dict[str, Any]]:
        record_id = str(metadata["record_id"])
        source_rate = float(metadata["sampling_rate_hz"])
        channels = list(metadata["channels"])
        rows = []
        for index, (onset, offset) in enumerate(intervals):
            left = int(math.floor(onset * 10))
            right = max(left + 1, int(math.ceil(offset * 10)))
            confidence = float(np.mean(probability[left:right]))
            rows.append(
                {
                    "event_id": f"{record_id}:E07:{index:07d}",
                    "event_type": "sem",
                    "record_id": record_id,
                    "onset": onset,
                    "offset": offset,
                    "channel": "+".join(channels),
                    "detector_version": self.detector_version,
                    "parameters": {
                        "threshold": float(self.config["threshold"]),
                        **self.config["eventization"],
                    },
                    "quality_status": "AVAILABLE",
                    "evidence_status": "CANDIDATE_PROXY",
                    "measurements": {
                        "mean_model_score": confidence,
                        "score_semantics": "sigmoid_model_score_not_clinically_calibrated",
                    },
                    "source_model_sha256": self.checkpoint_sha256,
                    "source_signal_pointer": {
                        "record_id": record_id,
                        "channels": channels,
                        "sampling_rate_hz": source_rate,
                        "start_sample": int(round(onset * source_rate)),
                        "end_sample": int(round(offset * source_rate)),
                        "signal_unit": metadata["signal_unit"],
                    },
                    "adapter_version": self.adapter_version,
                }
            )
        validate_events(rows, duration_seconds)
        return rows

    def validate_output(self, events: list[dict[str, Any]], duration_seconds: float) -> None:
        validate_events(events, duration_seconds)

    def export_events(self, events: list[dict[str, Any]], output_path: str | Path) -> None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(events, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")


def load_model(config: dict[str, Any]) -> E07SEMNeuralHandler:
    return E07SEMNeuralHandler(config)
