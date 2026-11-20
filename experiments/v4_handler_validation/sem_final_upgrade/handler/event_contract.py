from __future__ import annotations

import math
from typing import Any


REQUIRED_FIELDS = {
    "event_id",
    "event_type",
    "record_id",
    "onset",
    "offset",
    "channel",
    "detector_version",
    "parameters",
    "quality_status",
    "evidence_status",
    "measurements",
    "source_model_sha256",
    "source_signal_pointer",
    "adapter_version",
}


def validate_events(events: list[dict[str, Any]], duration_seconds: float) -> None:
    identifiers = set()
    for event in events:
        missing = REQUIRED_FIELDS - set(event)
        if missing:
            raise ValueError(f"SEM event is missing fields: {sorted(missing)}")
        if event["event_type"] != "sem":
            raise ValueError("E07 adapter emitted a non-SEM event")
        onset = float(event["onset"])
        offset = float(event["offset"])
        if not (math.isfinite(onset) and math.isfinite(offset)):
            raise ValueError("Non-finite SEM boundary")
        if not 0 <= onset < offset <= duration_seconds:
            raise ValueError("SEM boundary outside source signal")
        if not 1.0 <= offset - onset <= 5.0:
            raise ValueError("SEM event violates frozen 1-5 second contract")
        if event["event_id"] in identifiers:
            raise ValueError("Duplicate SEM event identifier")
        identifiers.add(event["event_id"])
