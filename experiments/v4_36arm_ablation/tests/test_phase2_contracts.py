from __future__ import annotations

from copy import deepcopy

from experiments.v4_36arm_ablation.config import ARMS, EVENT_TYPE, HANDLERS, RECORDS
from experiments.v4_36arm_ablation.packaging import DEPENDENCIES, SINGLE_DECISIONS
from experiments.v4_36arm_ablation.shadow_overlay import overlay_handlers


def test_exact_arm_matrix() -> None:
    assert len(ARMS) == 36
    assert len(set(ARMS)) == 36
    assert len([arm for arm in ARMS if arm.startswith("S_")]) == 15
    assert len([arm for arm in ARMS if arm.startswith("G_")]) == 4
    assert len([arm for arm in ARMS if arm.startswith("L_")]) == 15
    assert len(ARMS["F_ALL15"]) == 15
    assert len(RECORDS) == 10


def test_event_types_are_unique() -> None:
    assert set(EVENT_TYPE) == set(HANDLERS)
    assert len(set(EVENT_TYPE.values())) == 15
    assert EVENT_TYPE["E03"] == "vertex"


def test_delivery_mappings_cover_every_handler() -> None:
    assert set(DEPENDENCIES) == set(HANDLERS)
    assert set(SINGLE_DECISIONS) == set(HANDLERS)
    assert all(len(DEPENDENCIES[handler]) == 4 for handler in HANDLERS)


def test_fail_closed_overlay_removes_frozen_candidate() -> None:
    bank = {
        "events": {
            "original": [],
            "enhanced": [{"event_id": "old", "event_type": "alpha", "onset": 0.0,
                          "offset": 1.0, "channel": "o", "measurements": {}}],
        },
        "availability": {"enhanced": {event_type: "AVAILABLE" for event_type in EVENT_TYPE.values()}},
        "info": {"duration": 30.0},
    }
    sources = {
        "E05": {"status": "UNAVAILABLE", "events": [], "overlay_availability": "UNAVAILABLE",
                "valid_signal_seconds": 0.0, "model": "blocked", "_source_bank_sha256": "x"},
    }
    integrated, manifest = overlay_handlers(deepcopy(bank), sources, ("E05",))
    assert not [event for event in integrated["events"]["enhanced"] if event["event_type"] == "alpha"]
    assert manifest["partial_effectiveness"] is True
    assert integrated["availability"]["enhanced"]["alpha"] == "UNAVAILABLE"
