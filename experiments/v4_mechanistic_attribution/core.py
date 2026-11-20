from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


STAGES = ("W", "N1", "N2", "N3", "R")
ALL_PREDICTIONS = STAGES + ("UNDETERMINED",)
HANDLERS = (
    "E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08",
    "E09", "E10", "E11", "E12", "E13", "SP", "AR",
)
EVENT_TYPE = {
    "E01": "slow_wave", "E02": "kcomplex", "E03": "vertex",
    "E04": "rem", "E05": "alpha", "E06": "blink", "E07": "sem",
    "E08": "lamf", "E09": "low_emg", "E10": "normal_emg",
    "E11": "high_emg", "E12": "slowing", "E13": "mbm",
    "SP": "spindle", "AR": "arousal",
}
EVIDENCE_KEYS = {
    "E01": ("slow_wave",),
    "E02": ("kcomplex", "n2_kcomplex", "first_half_n2"),
    "E03": ("vertex",),
    "E04": ("rem", "rem_low_overlap", "irregular_rem_normal"),
    "E05": ("alpha",), "E06": ("blink",),
    "E07": ("sem", "arousal_then_sem"), "E08": ("lamf",),
    "E09": ("low_emg", "rem_low_overlap"),
    "E10": ("normal_emg", "irregular_rem_normal"),
    "E11": ("high_emg",), "E12": ("slowing",), "E13": ("mbm",),
    "SP": ("spindle", "n2_spindle"),
    "AR": ("arousal", "arousal_then_sem", "n2_kcomplex"),
}
RULE_PREFIXES = {
    "E01": ("N3.1",),
    "E02": ("N2.1", "N2.2", "N2.4", "R.1", "R.2", "R.3"),
    "E03": ("N1.2",), "E04": ("W.3", "R.1", "R.2", "R.3"),
    "E05": ("W.1", "N1.1", "MBM.1"), "E06": ("W.2",),
    "E07": ("N1.2", "N2.4", "R.2", "R.3"),
    "E08": ("N1.1", "N2.2", "N2.4", "R.1", "R.2", "R.3"),
    "E09": ("R.1", "R.2"), "E10": ("W.3",), "E11": ("R.3",),
    "E12": ("N1.2",), "E13": ("MBM.1", "MBM.2", "N2.4"),
    "SP": ("N2.1", "N2.2", "R.1", "R.2", "R.3"),
    "AR": ("N2.1", "N2.2", "N2.4", "R.2", "R.3"),
}
GROUPS = {
    "G_N2": ("E02", "SP", "AR"),
    "G_OCULAR": ("E04", "E06", "E07"),
    "G_EEG": ("E01", "E03", "E05", "E08", "E12"),
    "G_EMG_MOV": ("E09", "E10", "E11", "E13"),
}
_record_text = os.environ.get("AASM_V4_RECORDS", "")
RECORDS = tuple(item.strip() for item in _record_text.split(",") if item.strip())
if not RECORDS:
    RECORDS = tuple(f"development_record_{index:02d}" for index in range(1, 11))


def read_config(path: Path) -> dict[str, Any]:
    raw = os.path.expandvars(path.read_text(encoding="utf-8"))
    if "${AASM_V4_" in raw:
        raise RuntimeError(
            "private replay paths are not configured; set the AASM_V4_* environment variables"
        )
    return json.loads(raw)


def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def load_reference(config: dict[str, Any]) -> pd.DataFrame:
    path = Path(config["reference_path"])
    actual = sha256(path)
    if actual != config["expected_reference_sha256"]:
        raise RuntimeError(f"Reference hash mismatch: {actual}")
    frame = pd.read_csv(path)
    frame = frame[frame["true_stage"].isin(STAGES)].copy()
    frame["epoch_index"] = frame["epoch_index"].astype(int)
    frame["record_id"] = frame["record_id"].astype(str)
    return frame


def load_arm_payload(source_root: Path, arm_id: str, kind: str) -> dict[str, list[dict[str, Any]]]:
    directory = source_root / kind / arm_id
    # Explicit allowlist: never enumerate records outside the frozen development partition.
    payload: dict[str, list[dict[str, Any]]] = {}
    for record_id in RECORDS:
        path = directory / f"{record_id}.json.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload[record_id] = read_json(path)
    return payload


def arm_prediction_frame(source_root: Path, arm_id: str, reference: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record_id, payload in load_arm_payload(source_root, arm_id, "predictions").items():
        for item in payload:
            rows.append({
                "record_id": record_id,
                "epoch_index": int(item["epoch_index"]),
                "prediction": item["stage"],
                "rule_id": item.get("rule_id", ""),
                "abstention_reason": item.get("abstention_reason", ""),
                "decision_source": item.get("decision_source", ""),
            })
    pred = pd.DataFrame(rows)
    merged = reference.merge(pred, on=["record_id", "epoch_index"], how="left", validate="one_to_one")
    if merged["prediction"].isna().any():
        missing = merged.loc[merged["prediction"].isna(), ["record_id", "epoch_index"]]
        raise RuntimeError(f"Missing predictions for {arm_id}: {missing.head().to_dict('records')}")
    return merged


def metric_from_arrays(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, Any]:
    yt = np.asarray(list(y_true), dtype=object)
    yp = np.asarray(list(y_pred), dtype=object)
    n = int(len(yt))
    accepted = yp != "UNDETERMINED"
    correct = yt == yp
    per_stage: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for stage in STAGES:
        tp = int(np.sum((yt == stage) & (yp == stage)))
        fp = int(np.sum((yt != stage) & (yp == stage)))
        fn = int(np.sum((yt == stage) & (yp != stage)))
        support = int(np.sum(yt == stage))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_stage[stage] = {
            "tp": tp, "fp": fp, "fn": fn, "support": support,
            "precision": precision, "recall": recall, "f1": f1,
        }
    correct_n = int(correct.sum())
    accepted_n = int(accepted.sum())
    matrix = np.zeros((6, 6), dtype=np.int64)
    lookup = {stage: index for index, stage in enumerate(ALL_PREDICTIONS)}
    for expected, actual in zip(yt, yp):
        matrix[lookup[str(expected)], lookup.get(str(actual), 5)] += 1
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    expected_accuracy = float(predicted @ support) / n**2 if n else 0.0
    accuracy = correct_n / n if n else 0.0
    return {
        "n": n, "correct": correct_n,
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "kappa": (accuracy - expected_accuracy) / (1 - expected_accuracy)
        if n and expected_accuracy < 1 else None,
        "coverage": accepted_n / n if n else 0.0,
        "accepted": accepted_n,
        "conditional_accuracy": int(np.sum(correct & accepted)) / accepted_n if accepted_n else 0.0,
        "per_stage": per_stage,
    }


def metric_for_arm(frame: pd.DataFrame, mask_column: str = "in_primary") -> dict[str, Any]:
    subset = frame[frame[mask_column].astype(bool)]
    return metric_from_arrays(subset["true_stage"], subset["prediction"])


def flatten_metric(metric: dict[str, Any]) -> dict[str, Any]:
    keys = ("n", "correct", "accuracy", "macro_f1", "kappa", "coverage", "accepted", "conditional_accuracy")
    row = {key: metric[key] for key in keys}
    for stage in STAGES:
        for name in ("precision", "recall", "f1", "support", "tp", "fp", "fn"):
            row[f"{stage}_{name}"] = metric["per_stage"][stage][name]
    return row


def confusion_rows(arm_id: str, scope: str, frame: pd.DataFrame, mask_column: str) -> list[dict[str, Any]]:
    subset = frame[frame[mask_column].astype(bool)]
    rows: list[dict[str, Any]] = []
    for true_stage in STAGES:
        for predicted in ALL_PREDICTIONS:
            rows.append({
                "arm_id": arm_id, "scope": scope, "true_stage": true_stage,
                "predicted_stage": predicted,
                "count": int(np.sum((subset.true_stage == true_stage) & (subset.prediction == predicted))),
            })
    return rows


def comparison_counts(base: pd.DataFrame, comp: pd.DataFrame, mask_column: str = "in_primary") -> dict[str, int]:
    left = base[base[mask_column].astype(bool)][["record_id", "epoch_index", "true_stage", "prediction"]]
    right = comp[comp[mask_column].astype(bool)][["record_id", "epoch_index", "prediction"]]
    merged = left.merge(right, on=["record_id", "epoch_index"], suffixes=("_base", "_comp"), validate="one_to_one")
    base_correct = merged.true_stage == merged.prediction_base
    comp_correct = merged.true_stage == merged.prediction_comp
    return {
        "corrected": int((~base_correct & comp_correct).sum()),
        "introduced": int((base_correct & ~comp_correct).sum()),
        "net_correct": int(comp_correct.sum() - base_correct.sum()),
        "both_correct": int((base_correct & comp_correct).sum()),
        "both_incorrect": int((~base_correct & ~comp_correct).sum()),
        "stage_changed": int((merged.prediction_base != merged.prediction_comp).sum()),
    }


def event_stats(source_root: Path, arm_id: str, event_type: str) -> dict[str, float | int]:
    count = 0
    duration_sum = union_duration = recording_duration = 0.0
    associated = 0
    for bank in load_arm_payload(source_root, arm_id, "event_banks").values():
        recording_duration += float(bank.get("info", {}).get("duration", 0.0))
        intervals: list[tuple[float, float]] = []
        for event in bank.get("events", {}).get("enhanced", []):
            if event.get("event_type") != event_type:
                continue
            onset, offset = float(event["onset"]), float(event["offset"])
            count += 1
            duration_sum += max(0.0, offset - onset)
            intervals.append((onset, offset))
            measurements = event.get("measurements", {}) or {}
            if measurements.get("associated_arousal_ids") or measurements.get("associated_arousal_id"):
                associated += 1
        intervals.sort()
        merged: list[list[float]] = []
        for onset, offset in intervals:
            if not merged or onset > merged[-1][1]:
                merged.append([onset, offset])
            else:
                merged[-1][1] = max(merged[-1][1], offset)
        union_duration += sum(offset - onset for onset, offset in merged)
    hours = recording_duration / 3600.0
    return {
        "event_count": count, "event_duration_sum_s": duration_sum,
        "event_union_duration_s": union_duration,
        "event_density_per_hour": count / hours if hours else 0.0,
        "recording_hours": hours, "associated_event_count": associated,
    }


def evidence_stats(source_root: Path, arm_id: str, keys: Iterable[str]) -> dict[str, Any]:
    positive_epochs = known_epochs = unknown_epochs = 0
    total_seconds = 0.0
    key_positive: Counter[str] = Counter()
    for payload in load_arm_payload(source_root, arm_id, "evidence").values():
        for epoch in payload:
            any_positive = False
            statuses: list[str] = []
            for key in keys:
                evidence = epoch.get("events", {}).get(key, {})
                try:
                    seconds = float(evidence.get("seconds") or 0.0)
                except (TypeError, ValueError):
                    seconds = 0.0
                total_seconds += seconds
                if seconds > 0:
                    any_positive = True
                    key_positive[key] += 1
                statuses.append(str(evidence.get("detector_status", "UNAVAILABLE")))
            positive_epochs += int(any_positive)
            if statuses and all(status in {"AVAILABLE", "NO_CANDIDATE"} for status in statuses):
                known_epochs += 1
            else:
                unknown_epochs += 1
    return {
        "evidence_positive_epochs": positive_epochs,
        "evidence_total_seconds": total_seconds,
        "evidence_known_epochs": known_epochs,
        "evidence_unknown_epochs": unknown_epochs,
        "evidence_key_positive_epochs": dict(key_positive),
    }


def rule_firing_counts(source_root: Path, arm_id: str, prefixes: Iterable[str]) -> Counter[str]:
    result: Counter[str] = Counter()
    for payload in load_arm_payload(source_root, arm_id, "predictions").values():
        for row in payload:
            rule_id = str(row.get("rule_id", ""))
            if any(rule_id.startswith(prefix) for prefix in prefixes):
                result[rule_id] += 1
    return result


def first_rule_difference(left: dict[str, Any], right: dict[str, Any]) -> str:
    l_items = [(str(x.get("rule_id")), str(x.get("value"))) for x in left.get("trace", [])]
    r_items = [(str(x.get("rule_id")), str(x.get("value"))) for x in right.get("trace", [])]
    for index in range(max(len(l_items), len(r_items))):
        l_value = l_items[index] if index < len(l_items) else ("<MISSING>", "<MISSING>")
        r_value = r_items[index] if index < len(r_items) else ("<MISSING>", "<MISSING>")
        if l_value != r_value:
            return f"{l_value[0]}:{l_value[1]}->{r_value[0]}:{r_value[1]}"
    if left.get("rule_id") != right.get("rule_id"):
        return f"FINAL:{left.get('rule_id')}->{right.get('rule_id')}"
    return ""


def evidence_signature(epoch: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    result: dict[str, tuple[Any, ...]] = {}
    for key, value in epoch.get("events", {}).items():
        result[key] = (
            round(float(value.get("seconds") or 0.0), 6),
            str(value.get("detector_status", "")),
            bool(value.get("negative_validated", False)),
        )
    return result


def changed_evidence_keys(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    l_sig, r_sig = evidence_signature(left), evidence_signature(right)
    return sorted(key for key in set(l_sig) | set(r_sig) if l_sig.get(key) != r_sig.get(key))


def temporal_before_signature(row: dict[str, Any]) -> str:
    before = row.get("temporal", {}).get("before", {})
    return json.dumps(before, sort_keys=True, ensure_ascii=False)


def changed_temporal_keys(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    l_before = left.get("temporal", {}).get("before", {})
    r_before = right.get("temporal", {}).get("before", {})
    return sorted(
        key for key in set(l_before) | set(r_before)
        if l_before.get(key) != r_before.get(key)
    )


def bootstrap_summary(values: np.ndarray, draws: int, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return {
        "record_equal_mean": float(values.mean()),
        "median_per_night": float(np.median(values)),
        "positive_night_fraction": float(np.mean(values > 0)),
        "worst_night_delta": float(values.min()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }


def sign_class(single: float, full: float, tolerance: float = 1e-12) -> str:
    s = 0 if abs(single) <= tolerance else (1 if single > 0 else -1)
    f = 0 if abs(full) <= tolerance else (1 if full > 0 else -1)
    mapping = {
        (1, 1): "POSITIVE_IN_BOTH", (-1, -1): "NEGATIVE_IN_BOTH",
        (1, -1): "SINGLE_POSITIVE_FULL_NEGATIVE",
        (-1, 1): "SINGLE_NEGATIVE_FULL_POSITIVE",
    }
    return mapping.get((s, f), "NEAR_ZERO_OR_MIXED")
