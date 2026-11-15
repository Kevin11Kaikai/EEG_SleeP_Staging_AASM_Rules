"""Public contracts for detector proposals entering the frozen event bank.

This module validates proposal *shape and provenance*.  It does not run a
detector, infer a clinical negative, or alter the frozen AASM rules.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable


REPLACEMENT_TYPES = {"A": "arousal", "K": "kcomplex", "S": "spindle"}
REPLACEMENT_POLICIES = {"REPLACE", "KEEP_FROZEN", "UNAVAILABLE"}
AVAILABILITY_STATES = {
    "AVAILABLE",
    "PARTIALLY_SUPPORTED",
    "NO_CANDIDATE",
    "INVALID_SIGNAL",
    "UNAVAILABLE",
    "DETECTOR_FAILURE",
}
QUALITY_STATES = AVAILABILITY_STATES
EVIDENCE_STATES = {"TRUE", "FALSE", "UNKNOWN", "CANDIDATE_PROXY"}
REQUIRED_FIELDS = {
    "event_id",
    "event_type",
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


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def validate_event(event: dict[str, Any], expected_type: str | None = None) -> None:
    """Raise ``ValueError`` when a replacement event violates the public contract."""
    missing = sorted(REQUIRED_FIELDS.difference(event))
    if missing:
        raise ValueError(f"event is missing required fields: {missing}")
    if not isinstance(event["event_id"], str) or not event["event_id"].strip():
        raise ValueError("event_id must be a non-empty stable string")
    if expected_type is not None and event["event_type"] != expected_type:
        raise ValueError(
            f"expected event_type={expected_type!r}, got {event['event_type']!r}"
        )
    onset, offset = event["onset"], event["offset"]
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value)
        for value in (onset, offset)
    ):
        raise ValueError("onset and offset must be finite numbers")
    if onset < 0 or offset <= onset:
        raise ValueError("event interval must satisfy 0 <= onset < offset")
    if not isinstance(event["channel"], str) or not event["channel"].strip():
        raise ValueError("channel must be a non-empty string")
    if (
        not isinstance(event["detector_version"], str)
        or not event["detector_version"].strip()
    ):
        raise ValueError("detector_version must be a non-empty string")
    if (
        not isinstance(event["adapter_version"], str)
        or not event["adapter_version"].strip()
    ):
        raise ValueError("adapter_version must be a non-empty string")
    if not isinstance(event["parameters"], dict) or not isinstance(
        event["measurements"], dict
    ):
        raise ValueError("parameters and measurements must be mappings")
    if event["quality_status"] not in QUALITY_STATES:
        raise ValueError(f"invalid quality_status: {event['quality_status']!r}")
    if event["evidence_status"] not in EVIDENCE_STATES:
        raise ValueError(f"invalid evidence_status: {event['evidence_status']!r}")
    if not _is_sha256(event["source_model_sha256"]):
        raise ValueError("source_model_sha256 must be a lowercase SHA256")
    pointer = event["source_signal_pointer"]
    if not isinstance(pointer, dict) or not {
        "record_id",
        "start_sample",
        "end_sample",
    } <= set(pointer):
        raise ValueError(
            "source_signal_pointer must contain record_id/start_sample/end_sample"
        )
    if not isinstance(pointer["record_id"], str) or not pointer["record_id"].strip():
        raise ValueError("source signal record_id must be non-empty")
    if not all(isinstance(pointer[key], int) for key in ("start_sample", "end_sample")):
        raise ValueError("sample pointers must be integers")
    if pointer["start_sample"] < 0 or pointer["end_sample"] <= pointer["start_sample"]:
        raise ValueError("sample pointer must satisfy 0 <= start_sample < end_sample")


def validate_replacement_set(
    events: Iterable[dict[str, Any]], expected_type: str
) -> list[dict[str, Any]]:
    """Validate and return a materialized replacement event list."""
    rows = list(events)
    ids: set[str] = set()
    for event in rows:
        validate_event(event, expected_type)
        if event["event_id"] in ids:
            raise ValueError(f"duplicate replacement event_id: {event['event_id']}")
        ids.add(event["event_id"])
    return rows


def validate_policies(policies: dict[str, str]) -> None:
    if set(policies) != set(REPLACEMENT_TYPES):
        raise ValueError("policies must explicitly cover A, K, and S")
    invalid = {
        key: value
        for key, value in policies.items()
        if value not in REPLACEMENT_POLICIES
    }
    if invalid:
        raise ValueError(f"invalid replacement policies: {invalid}")


def validate_availability(availability: dict[str, str]) -> None:
    if set(availability) != set(REPLACEMENT_TYPES):
        raise ValueError("availability must explicitly cover A, K, and S")
    invalid = {
        key: value
        for key, value in availability.items()
        if value not in AVAILABILITY_STATES
    }
    if invalid:
        raise ValueError(f"invalid availability states: {invalid}")
