from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.v3_detector_integration import (
    overlay_event_bank,
    rebuild_k_arousal,
    validate_associations,
    validate_event,
)


MODEL_SHA = "1" * 64


def event(event_id: str, kind: str, onset: float, offset: float) -> dict:
    return {
        "event_id": event_id,
        "event_type": kind,
        "onset": onset,
        "offset": offset,
        "channel": "synthetic",
        "detector_version": "synthetic-test-v1",
        "parameters": {"fixture": True},
        "quality_status": "AVAILABLE",
        "evidence_status": "CANDIDATE_PROXY",
        "measurements": {},
        "source_model_sha256": MODEL_SHA,
        "source_signal_pointer": {
            "record_id": "SYNTHETIC",
            "start_sample": int(onset * 100),
            "end_sample": int(offset * 100),
        },
        "adapter_version": "fixture-v1",
    }


def bank(events: list[dict]) -> dict:
    return {
        "info": {"record_id": "SYNTHETIC", "duration": 120},
        "events": {
            "original": [event("orig-alpha", "alpha", 0, 1)],
            "enhanced": events,
        },
        "availability": {
            "original": {},
            "enhanced": {
                "arousal": "AVAILABLE",
                "kcomplex": "AVAILABLE",
                "spindle": "AVAILABLE",
            },
        },
        "goods": {key: [True] * 4 for key in "fcolrm"},
        "diagnostics": {},
    }


def test_association_half_open_boundaries_and_idempotence() -> None:
    value = bank(
        [
            event("k", "kcomplex", 10, 12),
            event("eq-plus-one", "arousal", 13, 14),
            event("past-plus-one", "arousal", 13.000001, 14),
            event("end-eq-onset", "arousal", 9, 10),
            event("end-after-onset", "arousal", 9, 10.000001),
            event("cross-epoch", "arousal", 11.5, 30.5),
        ]
    )
    first, manifest = rebuild_k_arousal(value)
    second, _ = rebuild_k_arousal(first)
    linked = next(row for row in first["events"]["enhanced"] if row["event_id"] == "k")[
        "measurements"
    ]["associated_arousal_ids"]
    assert linked == ["end-after-onset", "cross-epoch", "eq-plus-one"]
    assert first == second
    assert manifest["changed_kcomplex_count"] == 1
    assert validate_associations(first) == []


def test_overlay_is_replace_not_union_and_preserves_input() -> None:
    original = bank(
        [
            event("old-a", "arousal", 1, 2),
            event("old-k", "kcomplex", 3, 4),
            event("old-s", "spindle", 5, 6),
            event("rem", "rem", 7, 8),
        ]
    )
    frozen = deepcopy(original)
    integrated, manifest, _ = overlay_event_bank(
        original,
        {"A": [event("new-a", "arousal", 2, 3)]},
        {"A": "REPLACE", "K": "KEEP_FROZEN", "S": "KEEP_FROZEN"},
        {"A": "PARTIALLY_SUPPORTED", "K": "AVAILABLE", "S": "AVAILABLE"},
    )
    ids = {row["event_id"] for row in integrated["events"]["enhanced"]}
    assert "new-a" in ids and "old-a" not in ids
    assert {"old-k", "old-s", "rem"} <= ids
    assert original == frozen
    assert manifest["input_bank_unchanged"]


def test_empty_replacement_stays_no_candidate_not_clinical_negative() -> None:
    original = bank([event("old-a", "arousal", 1, 2)])
    integrated, manifest, _ = overlay_event_bank(
        original,
        {"A": []},
        {"A": "REPLACE", "K": "KEEP_FROZEN", "S": "KEEP_FROZEN"},
        {"A": "NO_CANDIDATE", "K": "AVAILABLE", "S": "AVAILABLE"},
    )
    assert integrated["availability"]["enhanced"]["arousal"] == "NO_CANDIDATE"
    assert manifest["counts_after"]["arousal"] == 0


def test_replacement_requires_complete_provenance() -> None:
    bad = event("bad", "spindle", 1, 2)
    del bad["source_signal_pointer"]
    with pytest.raises(ValueError, match="source_signal_pointer"):
        validate_event(bad, "spindle")


def test_cross_type_replacement_is_rejected() -> None:
    original = bank([event("old-a", "arousal", 1, 2)])
    with pytest.raises(ValueError, match="expected event_type"):
        overlay_event_bank(
            original,
            {"A": [event("wrong", "spindle", 2, 3)]},
            {"A": "REPLACE", "K": "KEEP_FROZEN", "S": "KEEP_FROZEN"},
            {"A": "AVAILABLE", "K": "AVAILABLE", "S": "AVAILABLE"},
        )


def test_duplicate_ids_across_kept_and_inserted_are_rejected() -> None:
    original = bank([event("shared", "rem", 1, 2)])
    with pytest.raises(RuntimeError, match="duplicate event IDs"):
        overlay_event_bank(
            original,
            {"A": [event("shared", "arousal", 2, 3)]},
            {"A": "REPLACE", "K": "KEEP_FROZEN", "S": "KEEP_FROZEN"},
            {"A": "AVAILABLE", "K": "AVAILABLE", "S": "AVAILABLE"},
        )
