from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import sha256
from .data import (
    FAMILY_ROLES,
    HANDLER_ROLES,
    load_family_signals,
    resolve_role_channels,
)
from .edf import read_header
from .inference import build_event_rows, infer_full_record
from .models import FAMILY_SPECS, build_model
from .teachers import EVENT_TYPES, HANDLER_RATES


CONTINUOUS_HANDLERS = {"E05", "E08", "E09", "E10", "E11", "E12"}


class NeuralHandlerAdapter:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.handler_id = config["handler_id"]
        self.family = config["family"]
        self.checkpoint_path = Path(config["checkpoint"])
        self.model = None
        self.checkpoint_sha256 = ""

    def load_model(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        if config:
            self.config.update(config)
        expected = self.config["checkpoint_sha256"]
        actual = sha256(self.checkpoint_path)
        if actual != expected:
            raise ValueError("Checkpoint SHA256 mismatch")
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        model = build_model(self.family)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        self.model = model
        self.checkpoint_sha256 = actual
        return {
            "checkpoint": str(self.checkpoint_path.resolve()),
            "checkpoint_sha256": actual,
            "strict_load": True,
            "eval_mode": not model.training,
        }

    def validate_input(self, signal_metadata: dict[str, Any]) -> dict[str, Any]:
        path = Path(signal_metadata["file_path"])
        header = read_header(path)
        roles = resolve_role_channels(header)
        required = FAMILY_ROLES[self.family]
        missing = [role for role in required if not roles.get(role)]
        by_name = {signal.name: signal for signal in header.signals}
        handler_channels = [
            roles[role] for role in HANDLER_ROLES[self.handler_id] if roles.get(role)
        ]
        handler_rates = {
            channel: float(by_name[channel].sampling_rate) for channel in handler_channels
        }
        unique_handler_rates = sorted(set(handler_rates.values()))
        return {
            "valid": not missing,
            "missing_roles": missing,
            "role_channels": roles,
            "duration_seconds": header.duration_seconds,
            "handler_source_channels": handler_channels,
            "handler_source_sampling_rates_hz": handler_rates,
            "handler_source_sampling_rate_hz": (
                unique_handler_rates[0] if len(unique_handler_rates) == 1 else None
            ),
        }

    def preprocess(self, signal_metadata: dict[str, Any]) -> np.ndarray:
        contract = self.validate_input(signal_metadata)
        if not contract["valid"]:
            raise ValueError(f"Input contract failed: {contract['missing_roles']}")
        row = dict(signal_metadata)
        row["role_channels"] = contract["role_channels"]
        row.setdefault("recording_id", Path(row["file_path"]).stem)
        return load_family_signals([row], self.family)[row["recording_id"]]

    def detect(self, signal: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.load_model()
        device = torch.device(self.config.get("device", "cpu"))
        self.model.to(device)
        before = sha256(self.checkpoint_path)
        probability = infer_full_record(
            self.model,
            signal,
            self.family,
            int(self.config["window_seconds"]),
            device,
            batch_size=int(self.config.get("batch_size", 16)),
        )
        after = sha256(self.checkpoint_path)
        if before != after:
            raise AssertionError("Frozen checkpoint changed during inference")
        return probability

    def postprocess(
        self, prediction: np.ndarray, signal_metadata: dict[str, Any]
    ) -> list[dict[str, Any]]:
        spec = FAMILY_SPECS[self.family]
        head_index = spec.output_heads.index(self.handler_id)
        contract = self.validate_input(signal_metadata)
        if not contract["valid"]:
            raise ValueError(f"Input contract failed: {contract['missing_roles']}")
        roles = HANDLER_ROLES[self.handler_id]
        role_channels = signal_metadata.get("role_channels") or contract["role_channels"]
        source_channel = "+".join(role_channels[role] for role in roles)
        source_sampling_rate = signal_metadata.get("source_sampling_rate_hz")
        if source_sampling_rate is None:
            source_sampling_rate = contract["handler_source_sampling_rate_hz"]
        if source_sampling_rate is None:
            raise ValueError("Handler source channels do not share one sampling rate")
        return build_event_rows(
            signal_metadata["recording_id"],
            self.handler_id,
            EVENT_TYPES[self.handler_id],
            prediction[head_index],
            float(self.config["threshold"]),
            HANDLER_RATES[self.handler_id],
            self.config["eventization"],
            self.checkpoint_sha256 or sha256(self.checkpoint_path),
            source_channel,
            float(source_sampling_rate),
        )

    def validate_output(
        self, events: list[dict[str, Any]], duration_seconds: float
    ) -> dict[str, Any]:
        identifiers = set()
        for event in events:
            if event["event_id"] in identifiers:
                raise ValueError("Duplicate event ID")
            identifiers.add(event["event_id"])
            if event["event_type"] != EVENT_TYPES[self.handler_id]:
                raise ValueError("Event type mismatch")
            if not 0 <= event["onset"] < event["offset"] <= duration_seconds:
                raise ValueError("Event interval outside recording")
            if not np.isfinite(event["confidence"]):
                raise ValueError("Non-finite event confidence")
        return {"valid": True, "event_count": len(events)}

    def export_events(
        self,
        events: list[dict[str, Any]],
        prediction: np.ndarray,
        output_path: str | Path,
    ) -> dict[str, str]:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(events, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        probability_path = destination.with_suffix(".npz")
        value = np.asarray(prediction)
        if value.ndim == 2:
            head_index = FAMILY_SPECS[self.family].output_heads.index(self.handler_id)
            activity_score = value[head_index]
        elif value.ndim == 1:
            activity_score = value
        else:
            raise ValueError("Prediction must be a head vector or family-by-time matrix")
        rate_hz = HANDLER_RATES[self.handler_id]
        activity_intervals = np.asarray(
            [[event["onset"], event["offset"]] for event in events], dtype=np.float64
        ).reshape(-1, 2)
        np.savez_compressed(
            probability_path,
            probability=value,
            time_axis=np.arange(len(activity_score), dtype=np.float64) / rate_hz,
            activity_score=activity_score.astype(np.float32, copy=False),
            activity_intervals=activity_intervals,
            valid_mask=np.isfinite(activity_score).astype(np.uint8),
            temporal_resolution_seconds=np.asarray([1 / rate_hz], dtype=np.float64),
            signal_unit=np.asarray(["uV"]),
            source_model_sha256=np.asarray(
                [self.checkpoint_sha256 or sha256(self.checkpoint_path)]
            ),
            output_kind=np.asarray(
                [
                    "CONTINUOUS_ACTIVITY_INTERVALS"
                    if self.handler_id in CONTINUOUS_HANDLERS
                    else "DISCRETE_EVENT_INTERVALS"
                ]
            ),
        )
        return {
            "events": str(destination.resolve()),
            "events_sha256": sha256(destination),
            "probability": str(probability_path.resolve()),
            "probability_sha256": sha256(probability_path),
        }
