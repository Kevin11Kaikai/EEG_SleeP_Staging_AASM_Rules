from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import read_json, sha256, write_json
from .models import FAMILY_SPECS, build_model


HANDLER_RATES = {
    "E05": 1,
    "E06": 10,
    "E07": 10,
    "E08": 1,
    "E09": 2,
    "E10": 2,
    "E11": 2,
    "E12": 1,
    "E13": 1,
}

HANDLER_NAMES = {
    "E05": "alpha",
    "E06": "blink",
    "E07": "sem",
    "E08": "lamf",
    "E09": "low_emg",
    "E10": "normal_emg",
    "E11": "high_emg",
    "E12": "slowing",
    "E13": "mbm",
}


def _independent_counts(target: np.ndarray, prediction: np.ndarray, valid: np.ndarray) -> dict:
    target = target.astype(np.int8)
    prediction = prediction.astype(np.int8)
    valid = valid.astype(bool)
    tp = int(np.sum((target == 1) & (prediction == 1) & valid))
    fp = int(np.sum((target == 0) & (prediction == 1) & valid))
    fn = int(np.sum((target == 1) & (prediction == 0) & valid))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _independent_intervals(mask: np.ndarray, rate_hz: int) -> list[tuple[float, float]]:
    value = np.asarray(mask, dtype=np.int8)
    padded = np.pad(value, (1, 1))
    transition = np.diff(padded)
    starts = np.flatnonzero(transition == 1)
    stops = np.flatnonzero(transition == -1)
    return [(float(left / rate_hz), float(right / rate_hz)) for left, right in zip(starts, stops)]


def _independent_interval_match(
    reference: list[tuple[float, float]], predicted: list[tuple[float, float]]
) -> dict[str, int]:
    candidates: list[tuple[float, int, int]] = []
    first_possible = 0
    for truth_index, truth in enumerate(reference):
        while first_possible < len(predicted) and predicted[first_possible][1] <= truth[0]:
            first_possible += 1
        guess_index = first_possible
        while guess_index < len(predicted) and predicted[guess_index][0] < truth[1]:
            guess = predicted[guess_index]
            overlap = max(0.0, min(truth[1], guess[1]) - max(truth[0], guess[0]))
            union = max(truth[1], guess[1]) - min(truth[0], guess[0])
            score = overlap / union if union > 0 else 0.0
            if score >= 0.2:
                candidates.append((score, truth_index, guess_index))
            guess_index += 1
    matched_truth: set[int] = set()
    matched_guess: set[int] = set()
    for _, truth_index, guess_index in sorted(candidates, reverse=True):
        if truth_index not in matched_truth and guess_index not in matched_guess:
            matched_truth.add(truth_index)
            matched_guess.add(guess_index)
    true_positive = len(matched_truth)
    return {
        "tp": true_positive,
        "fp": len(predicted) - true_positive,
        "fn": len(reference) - true_positive,
    }


def run_independent_recheck(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    split = read_json(root / "data" / "split_manifest.json")
    with (root / "data" / "dataset_inventory.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        inventory = list(csv.DictReader(stream))
    split_names = ("train", "validation", "test")
    group_sets = [
        {row["subject_group_id"] for row in inventory if row["split"] == split_name}
        for split_name in split_names
    ]
    group_disjoint = all(
        not group_sets[i] & group_sets[j]
        for i in range(len(group_sets))
        for j in range(i + 1, len(group_sets))
    )
    split_record_manifest_match = all(
        sorted(row["recording_id"] for row in inventory if row["split"] == split_name)
        == sorted(split["records"][split_name])
        for split_name in split_names
    )
    checkpoint_rows = []
    for family in FAMILY_SPECS:
        manifest = read_json(root / "models" / family / "checkpoint_manifest.json")
        checkpoint = Path(manifest["checkpoint"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = build_model(family)
        incompatible = model.load_state_dict(payload["state_dict"], strict=True)
        checkpoint_rows.append(
            {
                "family": family,
                "hash_match": sha256(checkpoint) == manifest["checkpoint_sha256"],
                "strict_load": not incompatible.missing_keys and not incompatible.unexpected_keys,
                "optimizer_steps_nontrivial": manifest["actual_optimizer_steps"] >= 10,
                "weight_delta_positive": manifest["weight_delta_l2"] > 0,
                "no_nonfinite_gradients": manifest["nonfinite_gradient_steps_skipped"] == 0,
                "test_labels_not_used_for_selection": not manifest[
                    "test_labels_used_for_training_or_selection"
                ],
            }
        )
    metrics_path = root / "benchmarks" / "weak_teacher_agreement.csv"
    with metrics_path.open("r", encoding="utf-8", newline="") as stream:
        metric_rows = list(csv.DictReader(stream))
    recomputed = []
    for row in metric_rows:
        if row["scope"] != "pooled_test":
            continue
        handler = row["handler_id"]
        threshold = float(row["threshold"])
        targets = []
        predictions = []
        validity = []
        family = row["family"]
        prediction_dir = root / "models" / family / "test_predictions"
        event_totals = {"tp": 0, "fp": 0, "fn": 0}
        for path in sorted(prediction_dir.glob("*.npz")):
            with np.load(path, allow_pickle=False) as payload:
                heads = [str(value) for value in payload["heads"]]
                index = heads.index(handler)
                targets.append(payload["target"][index])
                predictions.append(payload["probability"][index] >= threshold)
                validity.append(payload["valid"][index])
                reference_intervals = _independent_intervals(
                    (payload["target"][index] > 0) & (payload["valid"][index] > 0),
                    HANDLER_RATES[handler],
                )
            event_path = (
                root
                / "handlers"
                / f"{handler}_{HANDLER_NAMES[handler]}"
                / "real_inference_results"
                / f"{path.stem}.events.json"
            )
            events = json.loads(event_path.read_text(encoding="utf-8"))
            predicted_intervals = [
                (float(event["onset"]), float(event["offset"])) for event in events
            ]
            event_counts = _independent_interval_match(
                reference_intervals, predicted_intervals
            )
            for key in event_totals:
                event_totals[key] += event_counts[key]
        independent = _independent_counts(
            np.concatenate(targets), np.concatenate(predictions), np.concatenate(validity)
        )
        recomputed.append(
            {
                "handler_id": handler,
                "reported_f1": float(row["temporal_f1"]),
                "recomputed_f1": independent["f1"],
                "match": abs(independent["f1"] - float(row["temporal_f1"])) < 1e-12,
                "reported_event_tp": int(row["event_tp"]),
                "reported_event_fp": int(row["event_fp"]),
                "reported_event_fn": int(row["event_fn"]),
                "recomputed_event_tp": event_totals["tp"],
                "recomputed_event_fp": event_totals["fp"],
                "recomputed_event_fn": event_totals["fn"],
                "event_match": event_totals
                == {
                    "tp": int(row["event_tp"]),
                    "fp": int(row["event_fp"]),
                    "fn": int(row["event_fn"]),
                },
                **independent,
            }
        )
    event_contract_rows = []
    duration_by_record = {
        row["recording_id"]: float(row["duration_seconds"]) for row in inventory
    }
    for handler_dir in sorted((root / "handlers").glob("E*_*")):
        handler = handler_dir.name.split("_", 1)[0]
        expected_event_type = read_json(handler_dir / "output_contract.json")["event_type"]
        valid = True
        count = 0
        for path in sorted((handler_dir / "real_inference_results").glob("*.events.json")):
            events = json.loads(path.read_text(encoding="utf-8"))
            record_id = path.name.removesuffix(".events.json")
            duration = duration_by_record[record_id]
            count += len(events)
            identifiers = set()
            for event in events:
                valid &= event["event_id"] not in identifiers
                identifiers.add(event["event_id"])
                valid &= 0 <= event["onset"] < event["offset"] <= duration
                valid &= event["event_type"] == expected_event_type
                valid &= np.isfinite(event["confidence"])
                valid &= event["source_model_sha256"] == read_json(
                    handler_dir / "checkpoint_reference.json"
                )["checkpoint_sha256"]
        event_contract_rows.append(
            {"handler_id": handler, "event_count": count, "contract_valid": bool(valid)}
        )
    forbidden_input_access = any(
        token in row["file_path"].lower()
        for row in inventory
        for token in ("hmc", "stage_reference", "outside_record_allowlist")
    )
    heritage = read_json(root / "heritage_manifest.json")
    frozen_hash_match = all(
        Path(row["path"]).exists() and sha256(row["path"]) == row["sha256_before"]
        for row in heritage["protected_files"]
    )
    result = {
        "schema_version": 1,
        "independent_implementation": True,
        "group_disjoint": group_disjoint,
        "group_disjoint_checked_from_actual_inventory": True,
        "split_record_manifest_match": split_record_manifest_match,
        "checkpoint_integrity": checkpoint_rows,
        "all_checkpoints_valid": all(
            all(
                value
                for key, value in row.items()
                if key != "family"
            )
            for row in checkpoint_rows
        ),
        "metric_recomputation": recomputed,
        "all_metric_recomputations_match": len(recomputed) == 9
        and all(row["match"] and row["event_match"] for row in recomputed),
        "event_contracts": event_contract_rows,
        "all_event_contracts_valid": len(event_contract_rows) == 9
        and all(row["contract_valid"] for row in event_contract_rows),
        "forbidden_input_access": forbidden_input_access,
        "stage_or_hypnogram_used": any(
            "hypnogram" in row["file_path"].lower() for row in inventory
        ),
        "frozen_heritage_hash_match": frozen_hash_match,
    }
    result["pass"] = all(
        [
            result["group_disjoint"],
            result["split_record_manifest_match"],
            result["all_checkpoints_valid"],
            result["all_metric_recomputations_match"],
            result["all_event_contracts_valid"],
            not result["forbidden_input_access"],
            not result["stage_or_hypnogram_used"],
            result["frozen_heritage_hash_match"],
        ]
    )
    write_json(root / "audits" / "independent_recheck.json", result)
    return result
