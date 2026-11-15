"""Copy-on-write overlay of selected detector proposals onto a frozen event bank."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

from .association import rebuild_k_arousal, validate_associations
from .contracts import (
    REPLACEMENT_TYPES,
    validate_availability,
    validate_policies,
    validate_replacement_set,
)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _type_hash(events: list[dict[str, Any]], event_type: str) -> str:
    return canonical_hash(
        [event for event in events if event["event_type"] == event_type]
    )


def overlay_event_bank(
    frozen_bank: dict[str, Any],
    replacements: dict[str, list[dict[str, Any]]],
    policies: dict[str, str],
    availability: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build an integrated bank while preserving the input and all non-replaced events.

    ``REPLACE`` is replacement, never union.  A missing candidate list is a valid
    empty proposal set only when the caller explicitly records an availability
    state such as ``NO_CANDIDATE``; it is never promoted to a clinical negative.
    """
    validate_policies(policies)
    validate_availability(availability)
    before_hash = canonical_hash(frozen_bank)
    integrated = deepcopy(frozen_bank)
    current = integrated["events"]["enhanced"]
    replacement_types = {
        REPLACEMENT_TYPES[key]
        for key, policy in policies.items()
        if policy == "REPLACE"
    }
    kept = [event for event in current if event["event_type"] not in replacement_types]
    inserted: list[dict[str, Any]] = []
    for key, event_type in REPLACEMENT_TYPES.items():
        policy = policies[key]
        if policy == "REPLACE":
            rows = validate_replacement_set(replacements.get(key, []), event_type)
            inserted.extend(deepcopy(rows))
            integrated["availability"]["enhanced"][event_type] = availability[key]
        elif policy == "UNAVAILABLE":
            integrated["availability"]["enhanced"][event_type] = "UNAVAILABLE"

    integrated["events"]["enhanced"] = sorted(
        kept + inserted,
        key=lambda event: (
            event["onset"],
            event["offset"],
            event["event_type"],
            event["channel"],
            event["event_id"],
        ),
    )
    integrated, association_manifest = rebuild_k_arousal(integrated)
    ids = [event["event_id"] for event in integrated["events"]["enhanced"]]
    if len(ids) != len(set(ids)):
        raise RuntimeError("overlay produced duplicate event IDs")
    if canonical_hash(frozen_bank) != before_hash:
        raise RuntimeError("input event bank was mutated")
    if canonical_hash(frozen_bank["events"]["original"]) != canonical_hash(
        integrated["events"]["original"]
    ):
        raise RuntimeError("original event bank changed")
    for key, event_type in REPLACEMENT_TYPES.items():
        if policies[key] != "KEEP_FROZEN":
            continue
        old_hash = _type_hash(frozen_bank["events"]["enhanced"], event_type)
        new_hash = _type_hash(integrated["events"]["enhanced"], event_type)
        if old_hash != new_hash and not (
            event_type == "kcomplex" and policies["A"] == "REPLACE"
        ):
            raise RuntimeError(f"frozen {event_type} changed unexpectedly")
    association_errors = validate_associations(integrated)
    if association_errors:
        raise RuntimeError(f"invalid rebuilt associations: {association_errors}")
    manifest = {
        "policies": dict(policies),
        "availability": dict(availability),
        "frozen_bank_hash_before": before_hash,
        "frozen_bank_hash_after": canonical_hash(frozen_bank),
        "integrated_bank_hash": canonical_hash(integrated),
        "counts_before": {
            kind: sum(event["event_type"] == kind for event in current)
            for kind in REPLACEMENT_TYPES.values()
        },
        "counts_after": {
            kind: sum(
                event["event_type"] == kind
                for event in integrated["events"]["enhanced"]
            )
            for kind in REPLACEMENT_TYPES.values()
        },
        "inserted_ids": [event["event_id"] for event in inserted],
        "input_bank_unchanged": canonical_hash(frozen_bank) == before_hash,
        "association_errors": association_errors,
    }
    return integrated, manifest, association_manifest
