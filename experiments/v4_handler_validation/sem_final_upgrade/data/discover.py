from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from experiments.v4_handler_validation.phase1c_handlers.edf import read_header


def _phase1c_dreams(config: dict[str, Any]) -> list[dict[str, Any]]:
    source = Path(config["phase1c_result"]) / "data" / "dataset_inventory.csv"
    group_to_role = {}
    for role, key in (
        ("train", "dreams_train_groups"),
        ("validation", "dreams_validation_groups"),
        ("original_test", "dreams_original_test_groups"),
    ):
        for group in config["split"][key]:
            group_to_role[group] = role
    rows = []
    with source.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["dataset_id"] != "DREAMS_KC_REM_PSG_EXCERPTS":
                continue
            roles = json.loads(row["role_channels"])
            rows.append(
                {
                    "record_id": row["recording_id"],
                    "dataset_id": row["dataset_id"],
                    "file_path": row["file_path"],
                    "group_id": row["subject_group_id"],
                    "subject_identity": row["subject_identity"],
                    "split_role": group_to_role[row["subject_group_id"]],
                    "left_channel": roles["l"],
                    "right_channel": roles["r"],
                    "license": row["license"],
                    "label_source": "FROZEN_V3_WEAK_TEACHER",
                    "source_task": row["source_task"],
                }
            )
    return rows


def _psg_ipa(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(config["psg_ipa_root"]) / "EEG_arousals" / "PSG"
    role_lookup = {}
    for role, key in (
        ("train", "psg_ipa_train"),
        ("validation", "psg_ipa_validation"),
        ("positive_challenge", "psg_ipa_positive_challenge"),
    ):
        for record in config["split"][key]:
            role_lookup[record] = role
    rows = []
    for stem, role in sorted(role_lookup.items()):
        source = root / f"{stem}.edf"
        if not source.exists():
            raise FileNotFoundError(f"Missing PSG-IPA signal: {source}")
        header = read_header(source)
        names = {signal.name for signal in header.signals}
        required = {"EOG E1-M2", "EOG E2-M2"}
        if not required.issubset(names):
            raise ValueError(f"PSG-IPA bilateral EOG contract failed for {source.name}")
        rows.append(
            {
                "record_id": "psgipa_" + stem,
                "dataset_id": "PSG_IPA_EEG_AROUSALS_1_0_0",
                "file_path": str(source.resolve()),
                "group_id": "psgipa_" + stem.split("_")[0],
                "subject_identity": stem.split("_")[0],
                "split_role": role,
                "left_channel": "EOG E1-M2",
                "right_channel": "EOG E2-M2",
                "license": "CC-BY-4.0",
                "label_source": "FROZEN_V3_WEAK_TEACHER",
                "source_task": "arousal_psg_signal_only",
            }
        )
    return rows


def discover_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _phase1c_dreams(config) + _psg_ipa(config)
    ids = [row["record_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise AssertionError("Duplicate E07 record identifiers")
    return sorted(rows, key=lambda row: row["record_id"])
