"""Rebuild K-complex/arousal links using the frozen half-open timing contract."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


ASSOCIATION_SEMANTICS = "a.onset <= k.offset + 1.0 and a.offset > k.onset"


def _is_associated(kcomplex: dict[str, Any], arousal: dict[str, Any]) -> bool:
    return (
        arousal["onset"] <= kcomplex["offset"] + 1.0
        and arousal["offset"] > kcomplex["onset"]
    )


def rebuild_k_arousal(bank: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a copy with all derived K/arousal associations deterministically rebuilt."""
    rebuilt = deepcopy(bank)
    events = rebuilt["events"]["enhanced"]
    arousals = sorted(
        (event for event in events if event["event_type"] == "arousal"),
        key=lambda event: (event["onset"], event["offset"], event["event_id"]),
    )
    changes: list[dict[str, Any]] = []
    for event in events:
        if event["event_type"] != "kcomplex":
            continue
        before = list(event.get("measurements", {}).get("associated_arousal_ids", []))
        after = [
            arousal["event_id"]
            for arousal in arousals
            if _is_associated(event, arousal)
        ]
        event.setdefault("measurements", {})["associated_arousal_ids"] = after
        changes.append(
            {
                "kcomplex_id": event["event_id"],
                "old_arousal_ids": before,
                "new_arousal_ids": after,
                "changed": before != after,
            }
        )
    return rebuilt, {
        "semantics": ASSOCIATION_SEMANTICS,
        "arousal_count": len(arousals),
        "kcomplex_count": len(changes),
        "changed_kcomplex_count": sum(row["changed"] for row in changes),
        "rows": changes,
    }


def validate_associations(bank: dict[str, Any]) -> list[str]:
    """Return referential/timing errors without mutating the bank."""
    events = bank["events"]["enhanced"]
    by_id = {event["event_id"]: event for event in events}
    errors: list[str] = []
    for event in events:
        if event["event_type"] != "kcomplex":
            continue
        for event_id in event.get("measurements", {}).get("associated_arousal_ids", []):
            linked = by_id.get(event_id)
            if linked is None:
                errors.append(f"missing:{event['event_id']}->{event_id}")
            elif linked["event_type"] != "arousal":
                errors.append(f"wrong_type:{event['event_id']}->{event_id}")
            elif not _is_associated(event, linked):
                errors.append(f"nonoverlap:{event['event_id']}->{event_id}")
    return errors
