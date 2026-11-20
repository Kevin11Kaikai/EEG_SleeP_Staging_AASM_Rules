from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "results" / "release" / "v4" / "rule_diagnostics"
EXPECTED_EPOCHS = {"full": 8902, "discovery": 4540, "confirmation": 4362}
EXPECTED_CORRECT = {
    "C000": {"full": 4947, "discovery": 2359, "confirmation": 2588},
    "C196": {"full": 5044, "discovery": 2398, "confirmation": 2646},
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (RELEASE / name).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_required_public_files_exist() -> None:
    expected = {
        "README.md",
        "current_rule_diagnostics.csv",
        "evidence_rule_audit.json",
        "final_rule_diagnostics.csv",
        "rule_stage_counts.csv",
        "provenance.json",
    }
    assert expected <= {path.name for path in RELEASE.iterdir() if path.is_file()}


def test_final_rule_diagnostics_close_for_every_stratum() -> None:
    rows = read_csv("final_rule_diagnostics.csv")
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        assert row["detector_profile"] == "M6408"
        assert row["window"] == "primary"
        count = int(row["count"])
        correct = int(row["correct"])
        wrong = int(row["wrong"])
        assert correct + wrong == count
        if count == 0:
            assert row["success_rate"] == ""
            assert row["status"] == "NO_FIRINGS"
        else:
            assert abs(float(row["success_rate"]) - correct / count) < 1e-12
            assert row["status"] == "OBSERVED"
        grouped[(row["rule_profile"], row["partition"])]["count"] += count
        grouped[(row["rule_profile"], row["partition"])]["correct"] += correct

    for profile in ("C000", "C196"):
        for partition, expected in EXPECTED_EPOCHS.items():
            assert grouped[(profile, partition)]["count"] == expected
            assert (
                grouped[(profile, partition)]["correct"]
                == EXPECTED_CORRECT[profile][partition]
            )


def test_wake_rules_and_paths_match_frozen_cache() -> None:
    diagnostics = read_csv("final_rule_diagnostics.csv")
    selected = {
        (row["rule_profile"], row["rule_id"]): row
        for row in diagnostics
        if row["partition"] == "full" and row["rule_id"] in {"W.1", "W.2", "W.3"}
    }
    for profile in ("C000", "C196"):
        assert (
            int(selected[(profile, "W.1")]["count"]),
            int(selected[(profile, "W.1")]["correct"]),
        ) == (67, 65)
        assert selected[(profile, "W.2")]["count"] == "0"
        assert selected[(profile, "W.2")]["success_rate"] == ""
        assert selected[(profile, "W.2")]["status"] == "NO_FIRINGS"
        assert (
            int(selected[(profile, "W.3")]["count"]),
            int(selected[(profile, "W.3")]["correct"]),
        ) == (6, 6)

    stages = read_csv("rule_stage_counts.csv")
    for profile in ("C000", "C196"):
        wake = [
            row
            for row in stages
            if row["rule_profile"] == profile
            and row["partition"] == "full"
            and row["predicted_stage"] == "W"
        ]
        assert sum(int(row["count"]) for row in wake) == 82
        assert (
            sum(int(row["count"]) for row in wake if row["reference_stage"] == "W")
            == 80
        )
        assert (
            sum(
                int(row["count"])
                for row in wake
                if row["final_rule_id"].startswith("FALLBACK")
            )
            == 9
        )
        assert (
            sum(
                int(row["count"])
                for row in wake
                if row["final_rule_id"].startswith(("MBM", "MOVEMENT"))
            )
            == 0
        )


def test_rule_stage_counts_rebuild_confusion() -> None:
    rows = read_csv("rule_stage_counts.csv")
    totals: Counter[tuple[str, str]] = Counter()
    diagonal: Counter[tuple[str, str]] = Counter()
    for row in rows:
        key = (row["rule_profile"], row["partition"])
        count = int(row["count"])
        totals[key] += count
        if row["predicted_stage"] == row["reference_stage"] != "UNDETERMINED":
            diagonal[key] += count
    for profile in ("C000", "C196"):
        for partition, expected in EXPECTED_EPOCHS.items():
            assert totals[(profile, partition)] == expected
            assert (
                diagonal[(profile, partition)] == EXPECTED_CORRECT[profile][partition]
            )


def test_evidence_and_provenance_boundaries() -> None:
    evidence = json.loads(
        (RELEASE / "evidence_rule_audit.json").read_text(encoding="utf-8")
    )
    assert evidence["counts"]["primary_epochs"] == 8902
    assert evidence["counts"]["blink_observed"] == 1179
    assert evidence["counts"]["blink_over_15"] == 0

    provenance_text = (RELEASE / "provenance.json").read_text(encoding="utf-8")
    provenance = json.loads(provenance_text)
    assert provenance["sealed_cache"]["record_level_payload_published"] is False
    assert provenance["partitions"]["full"]["record_count"] == 10
    assert provenance["partitions"]["full"]["primary_epoch_count"] == 8902
    assert "D:\\EEG_Conformer" not in provenance_text
    assert "C:\\Users" not in provenance_text
    for suffix in range(1, 11):
        assert f"SN{suffix:03d}" not in provenance_text
