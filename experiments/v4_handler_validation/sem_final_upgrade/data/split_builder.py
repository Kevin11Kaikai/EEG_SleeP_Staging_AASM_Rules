from __future__ import annotations

from typing import Any


def build_split_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    roles = ("train", "validation", "positive_challenge", "original_test")
    by_role = {
        role: sorted(row["record_id"] for row in records if row["split_role"] == role)
        for role in roles
    }
    groups = {
        role: sorted({row["group_id"] for row in records if row["split_role"] == role})
        for role in roles
    }
    overlaps = {}
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            overlap = sorted(set(groups[left]) & set(groups[right]))
            overlaps[f"{left}__{right}"] = overlap
            if overlap:
                raise AssertionError(f"Group leakage: {left} vs {right}: {overlap}")
    return {
        "schema_version": 1,
        "frozen_before_training": True,
        "split_policy": "RECORD_GROUP_DISJOINT_WITH_UNCHANGED_PHASE1C_ORIGINAL_TEST",
        "positive_challenge_name": "WEAK_TEACHER_POSITIVE_CHALLENGE",
        "positive_challenge_prevalence": "POSITIVE_ENRICHED; NOT NATURAL PREVALENCE",
        "expert_test": False,
        "records": by_role,
        "groups": groups,
        "group_overlaps": overlaps,
        "original_phase1c_test_unchanged": [
            "kc_excerpt10",
            "kc_excerpt8",
            "rem_excerpt9",
        ],
        "test_labels_available_to_training_worker": False,
        "stage_or_hypnogram_used": False,
    }
