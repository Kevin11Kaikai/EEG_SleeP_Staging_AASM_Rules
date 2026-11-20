from experiments.v4_mechanistic_attribution.core import (
    changed_evidence_keys,
    changed_temporal_keys,
    first_rule_difference,
)


def test_event_id_rewrite_is_not_evidence_change() -> None:
    left = {"events": {"slow_wave": {"seconds": 6.0, "detector_status": "AVAILABLE", "event_ids": ["a"]}}}
    right = {"events": {"slow_wave": {"seconds": 6.0, "detector_status": "AVAILABLE", "event_ids": ["b"]}}}
    assert changed_evidence_keys(left, right) == []


def test_predicate_value_change_is_found() -> None:
    left = {"trace": [{"rule_id": "N3.1", "value": "TRUE"}]}
    right = {"trace": [{"rule_id": "N3.1", "value": "FALSE"}]}
    assert first_rule_difference(left, right) == "N3.1:TRUE->N3.1:FALSE"


def test_temporal_field_names_are_reported() -> None:
    left = {"temporal": {"before": {"stage": "N2", "phase": "OTHER"}}}
    right = {"temporal": {"before": {"stage": "N1", "phase": "OTHER"}}}
    assert changed_temporal_keys(left, right) == ["stage"]
