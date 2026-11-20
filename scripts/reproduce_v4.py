"""Recheck the public V4 aggregate release without private PSG artifacts.

This script verifies the frozen B000 row, reconstructs the M_6408 metrics from
its released confusion matrix, and checks the released C196 paired counts.  It
does not claim to rerun detectors or stage raw recordings.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aasm_v4.contracts import STAGES, UNDETERMINED  # noqa: E402
from aasm_v4.evaluation.metrics import compute_fixed_mask_metrics  # noqa: E402


EXPECTED_B000 = {
    "n": 8902,
    "accuracy": 0.5199955066277241,
    "macro_f1": 0.40081488140174376,
    "coverage": 0.9939339474275444,
}
EXPECTED_M6408 = {
    "n": 8902,
    "accuracy": 0.555717816221074,
    "macro_f1": 0.44272863589516387,
    "coverage": 0.9946079532689284,
}
EXPECTED_C196_PAIRED = {
    "changed": 90,
    "corrected": 72,
    "introduced": 14,
    "net_correct": 58,
}
EXPECTED_RULE_DIAGNOSTICS = {
    "C000": {"n": 8902, "correct": 4947},
    "C196": {"n": 8902, "correct": 5044},
}


def _assert_close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"{name}: expected {expected}, found {actual}")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _reconstruct_from_confusion(
    confusion: list[list[int]],
) -> tuple[list[str], list[str]]:
    prediction_labels = (*STAGES, UNDETERMINED)
    reference: list[str] = []
    prediction: list[str] = []
    for truth_index, truth in enumerate(STAGES):
        row = confusion[truth_index]
        if len(row) != len(prediction_labels):
            raise ValueError(
                "released confusion matrix must have six prediction columns"
            )
        for guess, count in zip(prediction_labels, row):
            reference.extend([truth] * int(count))
            prediction.extend([guess] * int(count))
    return reference, prediction


def _recheck_rule_diagnostics(root: Path) -> dict[str, Any]:
    diagnostics_root = root / "rule_diagnostics"
    with (diagnostics_root / "final_rule_diagnostics.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        diagnostics = list(csv.DictReader(stream))
    with (diagnostics_root / "rule_stage_counts.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        stage_counts = list(csv.DictReader(stream))

    result: dict[str, Any] = {}
    for profile, expected in EXPECTED_RULE_DIAGNOSTICS.items():
        selected = [
            row
            for row in diagnostics
            if row["rule_profile"] == profile and row["partition"] == "full"
        ]
        n = sum(int(row["count"]) for row in selected)
        correct = sum(int(row["correct"]) for row in selected)
        if (n, correct) != (expected["n"], expected["correct"]):
            raise AssertionError(f"{profile} aggregate rule totals changed")
        wake = [
            row
            for row in stage_counts
            if row["rule_profile"] == profile
            and row["partition"] == "full"
            and row["predicted_stage"] == "W"
        ]
        wake_count = sum(int(row["count"]) for row in wake)
        wake_correct = sum(
            int(row["count"]) for row in wake if row["reference_stage"] == "W"
        )
        direct = sum(
            int(row["count"])
            for row in wake
            if row["final_rule_id"] in {"W.1", "W.2", "W.3"}
        )
        fallback = sum(
            int(row["count"])
            for row in wake
            if row["final_rule_id"].startswith("FALLBACK")
        )
        if (wake_count, wake_correct, direct, fallback) != (82, 80, 73, 9):
            raise AssertionError(f"{profile} Wake path aggregation changed")
        result[profile] = {
            "n": n,
            "correct": correct,
            "wake_outputs": wake_count,
            "wake_correct": wake_correct,
            "direct_wake_rule_outputs": direct,
            "fallback_wake_outputs": fallback,
        }
    return result


def reproduce() -> dict[str, Any]:
    phase2 = ROOT / "results" / "release" / "v4"
    metrics_path = phase2 / "phase2_36arm" / "all_36_arm_metrics.csv"
    with metrics_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    baseline = next(
        row for row in rows if row["scope"] == "primary" and row["arm_id"] == "B000"
    )
    if int(baseline["n"]) != EXPECTED_B000["n"]:
        raise AssertionError("B000 primary denominator changed")
    for metric in ("accuracy", "macro_f1", "coverage"):
        _assert_close(float(baseline[metric]), EXPECTED_B000[metric], f"B000 {metric}")

    champion_path = phase2 / "exhaustive_search" / "full_development_champions.json"
    champion = _load_json(champion_path)
    selected = champion["balanced_champion"]
    if selected["arm_id"] != "M_6408" or selected["handlers"] != "E04;E11;SP;AR":
        raise AssertionError("released exact-search champion changed")
    reference, prediction = _reconstruct_from_confusion(
        champion["balanced_champion_metrics"]["confusion"]
    )
    recomputed = compute_fixed_mask_metrics(reference, prediction)
    for metric in ("n", "accuracy", "macro_f1", "coverage"):
        actual = float(recomputed[metric])
        expected = float(EXPECTED_M6408[metric])
        _assert_close(actual, expected, f"M_6408 {metric}")

    paired_path = phase2 / "rule_discovery" / "confirmation_paired.json"
    paired = _load_json(paired_path)
    for metric, expected in EXPECTED_C196_PAIRED.items():
        if int(paired[metric]) != expected:
            raise AssertionError(f"C196 paired {metric} changed")

    rule_diagnostics = _recheck_rule_diagnostics(phase2)

    return {
        "status": "PUBLIC_AGGREGATE_RECHECK_PASS",
        "scope": "aggregate release only; no raw-signal replay",
        "b000": EXPECTED_B000,
        "m6408": {
            **EXPECTED_M6408,
            "handlers": ["E04", "E11", "SP", "AR"],
            "recomputed_from_confusion": True,
        },
        "c196_paired": EXPECTED_C196_PAIRED,
        "rule_diagnostics": rule_diagnostics,
    }


def main() -> int:
    print(json.dumps(reproduce(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
