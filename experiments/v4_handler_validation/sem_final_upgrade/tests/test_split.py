from __future__ import annotations

from experiments.v4_handler_validation.sem_final_upgrade.data.split_builder import build_split_manifest


def test_group_isolation_and_original_test_lock() -> None:
    records = [
        {"record_id": "a", "group_id": "A", "split_role": "train"},
        {"record_id": "b", "group_id": "B", "split_role": "validation"},
        {"record_id": "c", "group_id": "C", "split_role": "positive_challenge"},
        {"record_id": "kc_excerpt10", "group_id": "G10", "split_role": "original_test"},
        {"record_id": "kc_excerpt8", "group_id": "G05", "split_role": "original_test"},
        {"record_id": "rem_excerpt9", "group_id": "G05", "split_role": "original_test"},
    ]
    manifest = build_split_manifest(records)
    assert manifest["group_overlaps"]["train__positive_challenge"] == []
    assert manifest["original_phase1c_test_unchanged"] == [
        "kc_excerpt10",
        "kc_excerpt8",
        "rem_excerpt9",
    ]
