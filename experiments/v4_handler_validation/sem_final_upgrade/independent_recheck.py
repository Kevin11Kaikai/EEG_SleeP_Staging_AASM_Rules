from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import sha256, write_json


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.pad(np.asarray(mask, dtype=np.int8), (1, 1)))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _independent_eventize(
    probability: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    minimum_seconds: float,
    maximum_seconds: float,
    gap_seconds: float,
) -> list[tuple[float, float]]:
    fragments = _runs((probability >= threshold) & valid.astype(bool))
    merged: list[tuple[int, int]] = []
    max_gap = int(round(gap_seconds * 10))
    for left, right in fragments:
        if (
            merged
            and left - merged[-1][1] <= max_gap
            and valid[merged[-1][1] : left].all()
        ):
            merged[-1] = (merged[-1][0], right)
        else:
            merged.append((left, right))
    return [
        (left / 10.0, right / 10.0)
        for left, right in merged
        if minimum_seconds <= (right - left) / 10.0 <= maximum_seconds
    ]


def _iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return overlap / union if union else 0.0


def _independent_counts(
    truth: list[tuple[float, float]], predicted: list[tuple[float, float]]
) -> tuple[int, int, int]:
    candidates = []
    for i, left in enumerate(truth):
        for j, right in enumerate(predicted):
            score = _iou(left, right)
            if score >= 0.2:
                candidates.append((score, i, j))
    used_left = set()
    used_right = set()
    for _, i, j in sorted(candidates, reverse=True):
        if i not in used_left and j not in used_right:
            used_left.add(i)
            used_right.add(j)
    tp = len(used_left)
    return tp, len(predicted) - tp, len(truth) - tp


def run_independent_recheck(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    with (root / "selection.json").open("r", encoding="utf-8") as stream:
        selection = json.load(stream)
    with (root / "data" / "split_manifest.json").open("r", encoding="utf-8") as stream:
        split = json.load(stream)
    eventization = selection["eventization"]
    threshold = float(selection["threshold"])
    totals = {"tp": 0, "fp": 0, "fn": 0}
    per_record = []
    for record_id in split["records"]["positive_challenge"]:
        with np.load(root / "data" / "label_cache" / f"{record_id}.npz") as labels:
            target = labels["target"].astype(bool)
            scoring = labels["scoring_mask"].astype(bool)
        with np.load(root / "evaluation" / "probabilities" / f"{record_id}.npz") as values:
            probability = values["probability"]
        truth = [(a / 10.0, b / 10.0) for a, b in _runs(target & scoring)]
        predicted = _independent_eventize(
            probability,
            scoring,
            threshold,
            float(eventization["minimum_seconds"]),
            float(eventization["maximum_seconds"]),
            float(eventization["merge_gap_seconds"]),
        )
        tp, fp, fn = _independent_counts(truth, predicted)
        for key, value in zip(("tp", "fp", "fn"), (tp, fp, fn)):
            totals[key] += value
        per_record.append(
            {
                "record_id": record_id,
                "reference_events": len(truth),
                "predicted_events": len(predicted),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    with (root / "evaluation" / "positive_challenge_metrics.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    aggregate = next(row for row in rows if row["record_id"] == "__AGGREGATE__")
    metric_match = all(int(aggregate[key]) == totals[key] for key in totals)
    with (root / "adapter" / "challenge_events.json").open(
        "r", encoding="utf-8"
    ) as stream:
        adapter_events = json.load(stream)
    adapter_ids = [row["event_id"] for row in adapter_events]
    adapter_events_legal = bool(
        len(adapter_ids) == len(set(adapter_ids))
        and all(
            row["event_type"] == "sem"
            and np.isfinite(float(row["onset"]))
            and np.isfinite(float(row["offset"]))
            and 0 <= float(row["onset"]) < float(row["offset"])
            and 1.0 <= float(row["offset"]) - float(row["onset"]) <= 5.0
            for row in adapter_events
        )
    )
    adapter_count_matches = len(adapter_events) == int(aggregate["predicted_events"])
    full_qc_domain = True
    for record_id in split["records"]["validation"] + split["records"]["positive_challenge"] + split["records"]["original_test"]:
        with np.load(root / "data" / "label_cache" / f"{record_id}.npz") as labels:
            full_qc_domain = full_qc_domain and np.array_equal(
                labels["scoring_mask"], labels["signal_qc"]
            )
    with (root / "heritage_manifest.json").open("r", encoding="utf-8") as stream:
        heritage = json.load(stream)
    protected_matches = []
    for row in heritage["protected_files"]:
        path = Path(row["path"])
        current = sha256(path) if path.exists() else None
        protected_matches.append(
            {
                "path": str(path),
                "before": row["sha256"],
                "after": current,
                "unchanged": current == row["sha256"],
            }
        )
    checkpoint = Path(selection["checkpoint"])
    checkpoint_match = sha256(checkpoint) == selection["checkpoint_sha256"]
    fixed_expected = {
        "kc_excerpt10": "ea16f6013b0dfe16db6b788cf61464d0912c023f32d0633bc18cd5cc6c6fadbd",
        "kc_excerpt8": "ffd9ad452f1117453739bb06864549e5168ceec44d932179b801f6af67af3a25",
        "rem_excerpt9": "92a2c6e4b1223d6d321ad47f670b7bb8552212f502a42ab6c58056e579a9453e",
    }
    with (root / "data" / "source_inventory.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        sources = {row["record_id"]: row for row in csv.DictReader(stream)}
    fixed_test_match = all(
        sources[record_id]["source_edf_sha256"] == expected
        for record_id, expected in fixed_expected.items()
    )
    challenge_groups = set(split["groups"]["positive_challenge"])
    development_groups = set(split["groups"]["train"]) | set(
        split["groups"]["validation"]
    )
    result = {
        "schema_version": 1,
        "independent_algorithm": True,
        "challenge_counts_recomputed": totals,
        "challenge_per_record_recomputed": per_record,
        "main_metrics_match": metric_match,
        "adapter_event_count_matches_full_qc_evaluation": adapter_count_matches,
        "adapter_events_legal": adapter_events_legal,
        "scoring_domain_equals_signal_qc": bool(full_qc_domain),
        "checkpoint_sha256_match": checkpoint_match,
        "fixed_phase1c_test_ids_and_hashes_match": fixed_test_match,
        "challenge_group_isolation": not bool(challenge_groups & development_groups),
        "protected_files": protected_matches,
        "protected_files_unchanged": all(row["unchanged"] for row in protected_matches),
        "no_hmc_path_in_source_inventory": all(
            "hmc" not in row["file_path"].lower() for row in sources.values()
        ),
        "stage_labels_used": False,
        "expert_metrics_claimed": False,
        "pass": bool(
            metric_match
            and adapter_count_matches
            and adapter_events_legal
            and full_qc_domain
            and checkpoint_match
            and fixed_test_match
            and not (challenge_groups & development_groups)
            and all(row["unchanged"] for row in protected_matches)
            and all("hmc" not in row["file_path"].lower() for row in sources.values())
        ),
    }
    write_json(root / "audits" / "independent_recheck.json", result)
    return result
