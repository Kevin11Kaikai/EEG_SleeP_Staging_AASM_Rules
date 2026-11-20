from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    ALIGNED_REFERENCE,
    FULL_MASK,
    RECORDS,
    SOURCE_CACHE,
    SOURCE_HYBRID_RUN,
    STAGE_TO_CODE,
    deterministic_mask_order,
)


B000_CONFUSION = np.asarray([
    [68, 243, 707, 102, 32, 24],
    [2, 246, 664, 16, 65, 19],
    [0, 415, 3076, 83, 17, 6],
    [0, 29, 1124, 381, 0, 0],
    [0, 222, 495, 3, 858, 5],
    [0, 0, 0, 0, 0, 0],
], dtype=np.int64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(matrix: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.int64)
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    f1 = []
    for index in range(5):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum()) - tp
        fn = int(matrix[index, :].sum()) - tp
        f1.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return {
        "n": total,
        "correct": correct,
        "accuracy": correct / total,
        "macro_f1": float(np.mean(f1)),
        "coverage": 1.0 - int(matrix[:, 5].sum()) / total,
        "confusion": matrix.tolist(),
    }


def run_independent_recheck(root: Path, cache_path: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks.append({"name": name, "pass": bool(condition), "evidence": evidence})

    heritage = json.loads((root / "heritage_manifest.json").read_text(encoding="utf-8"))
    current_source_hash = _sha256(SOURCE_CACHE)
    check(
        "source_cache_unchanged",
        heritage["source_cache_sha256_before"] == current_source_hash,
        {"before": heritage["source_cache_sha256_before"], "after": current_source_hash},
    )
    seed_cache = Path(heritage.get("seed_cache", SOURCE_CACHE))
    current_seed_hash = _sha256(seed_cache)
    expected_seed_hash = heritage.get(
        "seed_cache_sha256_before", heritage["source_cache_sha256_before"]
    )
    check(
        "seed_cache_unchanged",
        expected_seed_hash == current_seed_hash,
        {
            "path": str(seed_cache),
            "before": expected_seed_hash,
            "after": current_seed_hash,
        },
    )

    connection = sqlite3.connect(
        f"file:{cache_path.resolve().as_posix()}?mode=ro&immutable=1", uri=True
    )
    complete_masks = {
        int(row[0]) for row in connection.execute("SELECT mask FROM arms WHERE status='PASS'")
    }
    rows = connection.execute(
        """SELECT mask,record_id,stage_codes,epoch_count,stage_sha256,status
           FROM arm_records ORDER BY mask,record_id"""
    ).fetchall()
    compact: dict[tuple[int, str], np.ndarray] = {}
    integrity_failures = []
    counts: dict[int, int] = {}
    for mask, record, blob, epoch_count, expected_hash, status in rows:
        if status != "PASS":
            continue
        values = np.frombuffer(blob, dtype=np.uint8).copy()
        actual_hash = hashlib.sha256(values.tobytes()).hexdigest()
        if (
            record not in RECORDS
            or len(values) != int(epoch_count)
            or actual_hash != expected_hash
            or np.any(values > 5)
        ):
            integrity_failures.append([int(mask), record])
        compact[(int(mask), record)] = values
        counts[int(mask)] = counts.get(int(mask), 0) + 1
    bad_counts = [mask for mask in complete_masks if counts.get(mask) != len(RECORDS)]
    check(
        "compact_cache_integrity",
        not integrity_failures and not bad_counts,
        {"row_failures": integrity_failures[:20], "bad_mask_counts": bad_counts[:20]},
    )

    progress = json.loads((root / "replay" / "progress.json").read_text(encoding="utf-8"))
    order = deterministic_mask_order()
    structural = set(order[: int(progress["next_order_index"])])
    structural_hash = hashlib.sha256(
        ",".join(map(str, sorted(structural))).encode("utf-8")
    ).hexdigest()
    check(
        "deterministic_structural_prefix",
        structural_hash == progress["structural_prefix_sha256"]
        and structural <= complete_masks,
        {
            "next_order_index": progress["next_order_index"],
            "structural_count": len(structural),
            "missing_complete": len(structural - complete_masks),
        },
    )
    check(
        "exhaustive_claim_matches_cache",
        bool(progress["exhaustive_complete"])
        == (complete_masks == set(range(FULL_MASK + 1))),
        {"complete_masks": len(complete_masks), "claimed": progress["exhaustive_complete"]},
    )

    reference_frame = pd.read_csv(ALIGNED_REFERENCE)
    truth: dict[str, np.ndarray] = {}
    primary: dict[str, np.ndarray] = {}
    for record in RECORDS:
        frame = reference_frame.loc[reference_frame.record_id == record].sort_values("epoch_index")
        truth[record] = np.asarray(
            [STAGE_TO_CODE.get(value, 255) for value in frame.true_stage], dtype=np.uint8
        )
        primary[record] = frame.in_primary.to_numpy(dtype=bool)

    def matrix(mask: int, records: list[str] | tuple[str, ...]) -> np.ndarray:
        output = np.zeros((6, 6), dtype=np.int64)
        for record in records:
            valid = primary[record]
            actual = truth[record][valid]
            predicted = compact[(mask, record)][valid]
            for left, right in zip(actual, predicted):
                output[int(left), int(right)] += 1
        return output

    b000 = matrix(0, RECORDS)
    b000_metric = _metrics(b000)
    check("b000_confusion_exact", np.array_equal(b000, B000_CONFUSION), b000.tolist())
    check(
        "b000_metrics_exact",
        b000_metric["n"] == 8902
        and b000_metric["correct"] == 4629
        and abs(b000_metric["accuracy"] - 0.5199955066277241) < 1e-15
        and abs(b000_metric["macro_f1"] - 0.40081488140174376) < 1e-15,
        b000_metric,
    )

    frozen = json.loads((root / "selection" / "outer_winners_frozen.json").read_text(encoding="utf-8"))
    nested = json.loads((root / "selection" / "nested_results.json").read_text(encoding="utf-8"))
    heldout_records = [row["heldout_record"] for row in frozen["winners"]]
    selection_isolation_failures = []
    candidate_universe_failures = []
    for row in frozen["winners"]:
        heldout = row["heldout_record"]
        expected_selection = set(RECORDS) - {heldout}
        if set(row["selection_records"]) != expected_selection:
            selection_isolation_failures.append(row["fold_id"])
        if not bool(progress["exhaustive_complete"]):
            old_path = (
                SOURCE_HYBRID_RUN / "search" / "outer" / f"{row['fold_id']}.json"
            )
            safe_warm = {
                int(mask)
                for mask in json.loads(old_path.read_text(encoding="utf-8"))["evaluated_masks"]
            }
            if int(row["winner_mask"]) not in (structural | safe_warm):
                candidate_universe_failures.append(row["fold_id"])
    check(
        "outer_fold_record_isolation",
        not selection_isolation_failures
        and sorted(heldout_records) == sorted(RECORDS)
        and len(set(heldout_records)) == len(RECORDS),
        {
            "heldout_records": heldout_records,
            "selection_failures": selection_isolation_failures,
        },
    )
    check(
        "partial_candidate_universe_is_fold_safe",
        not candidate_universe_failures,
        candidate_universe_failures,
    )
    check(
        "all_winners_frozen_before_heldout",
        len(frozen["winners"]) == 10
        and frozen["frozen_at"] <= nested["heldout_scoring_started_at"],
        {"frozen_at": frozen["frozen_at"], "score_started": nested["heldout_scoring_started_at"]},
    )
    candidate_matrix = np.zeros((6, 6), dtype=np.int64)
    baseline_matrix = np.zeros((6, 6), dtype=np.int64)
    corrected = introduced = 0
    for winner in frozen["winners"]:
        record = winner["heldout_record"]
        candidate_mask = int(winner["winner_mask"])
        candidate_matrix += matrix(candidate_mask, (record,))
        baseline_matrix += matrix(0, (record,))
        valid = primary[record]
        actual = truth[record][valid]
        left = compact[(0, record)][valid] == actual
        right = compact[(candidate_mask, record)][valid] == actual
        corrected += int(np.sum(~left & right))
        introduced += int(np.sum(left & ~right))
    recomputed_candidate = _metrics(candidate_matrix)
    recomputed_baseline = _metrics(baseline_matrix)
    metric_match = all(
        abs(float(recomputed_candidate[key]) - float(nested["candidate"][key])) < 1e-12
        and abs(float(recomputed_baseline[key]) - float(nested["baseline"][key])) < 1e-12
        for key in ("accuracy", "macro_f1", "coverage")
    )
    check(
        "nested_metrics_independently_recomputed",
        metric_match,
        {"candidate": recomputed_candidate, "baseline": recomputed_baseline},
    )
    closure = {
        "corrected": corrected,
        "introduced": introduced,
        "net_correct": corrected - introduced,
        "candidate_minus_baseline": (
            recomputed_candidate["correct"] - recomputed_baseline["correct"]
        ),
    }
    reported_paired = nested["paired_counts"]
    paired_keys = ("corrected", "introduced", "net_correct")
    paired_matches_report = all(
        int(closure[key]) == int(reported_paired[key]) for key in paired_keys
    )
    check(
        "paired_count_closure",
        closure["net_correct"] == closure["candidate_minus_baseline"]
        and paired_matches_report,
        {"recomputed": closure, "reported": reported_paired},
    )

    finalist_manifest = json.loads(
        (root / "finalists" / "manifest.json").read_text(encoding="utf-8")
    )
    finalist_failures = []
    for item in finalist_manifest:
        mask = int(item["mask"])
        record = item["record_id"]
        arm_id = f"M_{mask:04X}"
        paths = {
            "event_bank": root / "finalists" / "full_trace" / arm_id / "event_banks" / f"{record}.json.gz",
            "evidence": root / "finalists" / "full_trace" / arm_id / "evidence" / f"{record}.json.gz",
            "predictions": root / "finalists" / "full_trace" / arm_id / "predictions" / f"{record}.json.gz",
            "traces": root / "finalists" / "full_trace" / arm_id / "traces" / f"{record}.json.gz",
            "overlay": root / "finalists" / "full_trace" / arm_id / "overlay" / f"{record}.json",
        }
        hashes_match = all(
            path.exists() and _sha256(path) == item["artifact_hashes"][name]
            for name, path in paths.items()
        )
        prediction_hash = None
        if paths["predictions"].exists():
            with gzip.open(paths["predictions"], "rt", encoding="utf-8") as handle:
                predictions = json.load(handle)
            stages = np.asarray(
                [STAGE_TO_CODE[row["stage"]] for row in predictions], dtype=np.uint8
            )
            prediction_hash = hashlib.sha256(stages.tobytes()).hexdigest()
        if (
            not hashes_match
            or prediction_hash != item["stage_sha256"]
            or item["stage_sha256"]
            != hashlib.sha256(compact[(mask, record)].tobytes()).hexdigest()
        ):
            finalist_failures.append([mask, record])
    check("finalist_trace_integrity", not finalist_failures, finalist_failures[:20])

    connection.close()
    passed = all(item["pass"] for item in checks)
    return {
        "status": "PASS" if passed else "FAIL",
        "independent_implementation": True,
        "summary": {"passed": sum(item["pass"] for item in checks), "total": len(checks)},
        "checks": checks,
    }
