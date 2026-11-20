from __future__ import annotations

from collections import Counter, defaultdict
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ALIGNED_REFERENCE, ARMS, HANDLERS, PUBLIC_ROOT, RECORDS
from .io_utils import canonical_hash, read_gzip_json, read_json, sha256, write_csv, write_json


STAGES = ("W", "N1", "N2", "N3", "R", "UNDETERMINED")


def _evaluator():
    source = PUBLIC_ROOT / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("phase2_frozen_evaluator", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


def _verify_seal(run_root: Path) -> dict[str, Any]:
    seal = read_json(run_root / "arms" / "prediction_seal.json")
    items = []
    for arm in ARMS:
        for record in RECORDS:
            path = run_root / "predictions" / arm / f"{record}.json.gz"
            items.append((arm, record, sha256(path)))
    if seal.get("status") != "SEALED" or seal.get("prediction_key_hash") != canonical_hash(items):
        raise RuntimeError("prediction seal mismatch; stage labels remain locked")
    return seal


def _comparator(arm: str) -> str:
    return "F_ALL15" if arm.startswith("L_") else "B000"


def _metric_flat(metric: dict[str, Any]) -> dict[str, Any]:
    output = {key: metric[key] for key in (
        "n", "correct", "accuracy", "macro_f1", "kappa", "coverage",
        "abstention_rate", "conditional_accuracy", "accepted_risk",
    )}
    for stage in STAGES[:5]:
        for key in ("precision", "recall", "f1", "support"):
            output[f"{stage}_{key}"] = metric["per_stage"][stage][key]
    return output


def _bootstrap(
    per_record_cm: dict[str, np.ndarray],
    comparator_cm: dict[str, np.ndarray],
    evaluator: Any,
    draws: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    records = list(RECORDS)
    rng = np.random.default_rng(seed)
    deltas = {key: [] for key in ("accuracy", "macro_f1", "W_f1", "N1_f1", "N2_f1", "N3_f1", "R_f1")}
    for _ in range(draws):
        sampled = rng.choice(records, size=len(records), replace=True)
        cm = sum((per_record_cm[record] for record in sampled), np.zeros((6, 6), dtype=np.int64))
        base = sum((comparator_cm[record] for record in sampled), np.zeros((6, 6), dtype=np.int64))
        m, b = evaluator.metrics(cm), evaluator.metrics(base)
        deltas["accuracy"].append(m["accuracy"] - b["accuracy"])
        deltas["macro_f1"].append(m["macro_f1"] - b["macro_f1"])
        for stage in STAGES[:5]:
            deltas[f"{stage}_f1"].append(m["per_stage"][stage]["f1"] - b["per_stage"][stage]["f1"])
    output = {"draws": draws, "seed": seed, "resampling_unit": "recording/night"}
    for key, values in deltas.items():
        array = np.asarray(values)
        output[key] = {
            "mean": float(array.mean()), "ci_low": float(np.quantile(array, 0.025)),
            "ci_high": float(np.quantile(array, 0.975)), "positive_fraction": float(np.mean(array > 0)),
        }
    return output


def evaluate(run_root: Path) -> dict[str, Any]:
    seal = _verify_seal(run_root)
    seal["label_access_started"] = True
    seal["label_access_path"] = str(ALIGNED_REFERENCE.resolve())
    write_json(run_root / "arms" / "prediction_seal.json", seal)
    aligned = pd.read_csv(ALIGNED_REFERENCE)
    if set(aligned.record_id.unique()) != set(RECORDS):
        raise RuntimeError("reference record set differs from allow-list")
    evaluator, evaluator_source = _evaluator()
    predictions: dict[str, dict[str, list[dict[str, Any]]]] = {arm: {} for arm in ARMS}
    matrices: dict[str, dict[str, dict[str, np.ndarray]]] = {
        scope: {arm: {} for arm in ARMS} for scope in ("primary", "full")
    }
    per_record_rows, paired_rows, transition_rows, rule_rows, evidence_rows = [], [], [], [], []
    for record in RECORDS:
        reference = aligned.loc[aligned.record_id == record].sort_values("epoch_index")
        if reference.epoch_index.tolist() != list(range(len(reference))):
            raise RuntimeError(f"reference index mismatch: {record}")
        y = reference.true_stage.to_numpy()
        for arm in ARMS:
            pred = read_gzip_json(run_root / "predictions" / arm / f"{record}.json.gz")
            if len(pred) != len(reference):
                raise RuntimeError(f"prediction/reference mismatch: {arm}/{record}")
            predictions[arm][record] = pred
            p = np.asarray([row["stage"] for row in pred])
            for scope, column in (("primary", "in_primary"), ("full", "in_full_record")):
                mask = reference[column].to_numpy(dtype=bool)
                cm = evaluator.confusion(y[mask], p[mask])
                matrices[scope][arm][record] = cm
                per_record_rows.append({"record_id": record, "scope": scope, "arm_id": arm,
                                        **_metric_flat(evaluator.metrics(cm))})
            evidence = read_gzip_json(run_root / "evidence" / arm / f"{record}.json.gz")
            for key in sorted({name for row in evidence for name in row["events"]}):
                entries = [row["events"].get(key, {}) for row in evidence]
                seconds = [entry.get("seconds") for entry in entries]
                evidence_rows.append({
                    "record_id": record, "arm_id": arm, "evidence": key, "epochs": len(entries),
                    "known_epochs": sum(value is not None for value in seconds),
                    "positive_epochs": sum(value is not None and value > 0 for value in seconds),
                    "unknown_epochs": sum(value is None for value in seconds),
                    "total_seconds": sum(float(value or 0) for value in seconds),
                })
        for arm in ARMS:
            comparator = _comparator(arm)
            alt = predictions[arm][record]
            base = predictions[comparator][record]
            for scope, column in (("primary", "in_primary"), ("full", "in_full_record")):
                mask = reference[column].to_numpy(dtype=bool)
                for true_stage in ("ALL",) + STAGES[:5]:
                    selected = mask if true_stage == "ALL" else mask & (y == true_stage)
                    indices = np.flatnonzero(selected)
                    b_correct = np.asarray([base[i]["stage"] == y[i] for i in indices])
                    a_correct = np.asarray([alt[i]["stage"] == y[i] for i in indices])
                    corrected = int(np.sum(~b_correct & a_correct))
                    introduced = int(np.sum(b_correct & ~a_correct))
                    paired_rows.append({
                        "record_id": record, "scope": scope, "arm_id": arm,
                        "comparator": comparator, "true_stage": true_stage, "n": len(indices),
                        "corrected": corrected, "introduced": introduced,
                        "net_corrected": corrected - introduced,
                        "both_correct": int(np.sum(b_correct & a_correct)),
                        "both_incorrect": int(np.sum(~b_correct & ~a_correct)),
                        "rule_changed": sum(base[i]["rule_id"] != alt[i]["rule_id"] for i in indices),
                        "evidence_ids_changed": sum(sorted(base[i]["evidence_event_ids"]) != sorted(alt[i]["evidence_event_ids"]) for i in indices),
                    })
                for i in np.flatnonzero(mask):
                    transition_rows.append({
                        "record_id": record, "scope": scope, "arm_id": arm,
                        "comparator": comparator, "true_stage": y[i],
                        "from_stage": base[i]["stage"], "to_stage": alt[i]["stage"],
                        "from_rule": base[i]["rule_id"], "to_rule": alt[i]["rule_id"],
                    })
            for row in alt:
                rule_rows.append({"record_id": record, "arm_id": arm, "rule_id": row["rule_id"],
                                  "stage": row["stage"]})
    metrics_rows, per_stage_rows, metric_tree, confusion_tree, bootstrap_tree = [], [], {}, {}, {}
    for scope in ("primary", "full"):
        metric_tree[scope], confusion_tree[scope], bootstrap_tree[scope] = {}, {}, {}
        for arm in ARMS:
            cm = sum(matrices[scope][arm].values(), np.zeros((6, 6), dtype=np.int64))
            metric = evaluator.metrics(cm)
            metric_tree[scope][arm] = metric
            confusion_tree[scope][arm] = cm.tolist()
            comparator = _comparator(arm)
            base_cm = sum(matrices[scope][comparator].values(), np.zeros((6, 6), dtype=np.int64))
            base_metric = evaluator.metrics(base_cm)
            row = {"scope": scope, "arm_id": arm, "comparator": comparator, **_metric_flat(metric),
                   "delta_accuracy": metric["accuracy"] - base_metric["accuracy"],
                   "delta_macro_f1": metric["macro_f1"] - base_metric["macro_f1"]}
            for stage in STAGES[:5]:
                row[f"delta_{stage}_f1"] = metric["per_stage"][stage]["f1"] - base_metric["per_stage"][stage]["f1"]
                per_stage_rows.append({"scope": scope, "arm_id": arm, "comparator": comparator,
                                       "stage": stage, **metric["per_stage"][stage],
                                       "delta_f1": row[f"delta_{stage}_f1"]})
            metrics_rows.append(row)
            if arm != comparator:
                bootstrap_tree[scope][arm] = _bootstrap(matrices[scope][arm], matrices[scope][comparator], evaluator)
        expected_n = 8902 if scope == "primary" else 9211
        if {metric_tree[scope][arm]["n"] for arm in ARMS} != {expected_n}:
            raise RuntimeError(f"{scope} fixed mask count changed")
    # Aggregate transition and rule tables compactly.
    transition_counts = Counter((row["scope"], row["arm_id"], row["comparator"], row["true_stage"],
                                 row["from_stage"], row["to_stage"], row["from_rule"], row["to_rule"])
                                for row in transition_rows)
    transition_agg = [{"scope": k[0], "arm_id": k[1], "comparator": k[2], "true_stage": k[3],
                       "from_stage": k[4], "to_stage": k[5], "from_rule": k[6], "to_rule": k[7],
                       "count": value} for k, value in sorted(transition_counts.items())]
    rule_counts = Counter((row["arm_id"], row["rule_id"], row["stage"]) for row in rule_rows)
    rule_agg = [{"arm_id": k[0], "rule_id": k[1], "stage": k[2], "count": value}
                for k, value in sorted(rule_counts.items())]
    evaluation = run_root / "evaluation"
    write_csv(evaluation / "all_36_arm_metrics.csv", metrics_rows)
    write_csv(evaluation / "per_stage_metrics.csv", per_stage_rows)
    write_csv(evaluation / "per_night_metrics.csv", per_record_rows)
    write_csv(evaluation / "paired_error_analysis.csv", paired_rows)
    write_csv(evaluation / "stage_rule_transitions.csv", transition_agg)
    write_csv(evaluation / "rule_firing_summary.csv", rule_agg)
    write_csv(evaluation / "evidence_usage.csv", evidence_rows)
    write_json(evaluation / "metrics.json", metric_tree)
    write_json(evaluation / "confusion_matrices.json", confusion_tree)
    write_json(evaluation / "recording_bootstrap.json", bootstrap_tree)
    write_json(evaluation / "evaluator_manifest.json", {
        "evaluator_source": str(evaluator_source.resolve()), "evaluator_sha256": sha256(evaluator_source),
        "reference": str(ALIGNED_REFERENCE.resolve()), "reference_sha256": sha256(ALIGNED_REFERENCE),
        "prediction_seal": str((run_root / "arms" / "prediction_seal.json").resolve()),
        "primary_epochs": 8902, "full_valid_epochs": 9211,
        "label_access_after_neural_and_arm_prediction_seals": True,
    })
    return {"metrics": metric_tree, "matrices": matrices, "paired_rows": paired_rows,
            "per_record_rows": per_record_rows, "bootstrap": bootstrap_tree}


def build_effect_tables(run_root: Path, evaluation: dict[str, Any]) -> dict[str, Any]:
    metrics = evaluation["metrics"]["primary"]
    base = metrics["B000"]
    full = metrics["F_ALL15"]
    single_rows, loo_rows = [], []
    for handler in HANDLERS:
        single = metrics[f"S_{handler}"]
        leave = metrics[f"L_{handler}"]
        single_rows.append({
            "handler_id": handler,
            "single_delta_accuracy": single["accuracy"] - base["accuracy"],
            "single_delta_macro_f1": single["macro_f1"] - base["macro_f1"],
            **{f"single_delta_{stage}_f1": single["per_stage"][stage]["f1"] - base["per_stage"][stage]["f1"] for stage in STAGES[:5]},
        })
        loo_rows.append({
            "handler_id": handler,
            "full_minus_leaveout_accuracy": full["accuracy"] - leave["accuracy"],
            "full_minus_leaveout_macro_f1": full["macro_f1"] - leave["macro_f1"],
            **{f"full_minus_leaveout_{stage}_f1": full["per_stage"][stage]["f1"] - leave["per_stage"][stage]["f1"] for stage in STAGES[:5]},
        })
    interaction_rows = []
    for group, members in {key: value for key, value in ARMS.items() if key.startswith("G_")}.items():
        metric = metrics[group]
        summed_accuracy = sum(metrics[f"S_{member}"]["accuracy"] - base["accuracy"] for member in members)
        summed_macro = sum(metrics[f"S_{member}"]["macro_f1"] - base["macro_f1"] for member in members)
        interaction_rows.append({
            "group": group, "members": ";".join(members),
            "observed_delta_accuracy": metric["accuracy"] - base["accuracy"],
            "sum_single_delta_accuracy": summed_accuracy,
            "interaction_accuracy": metric["accuracy"] - base["accuracy"] - summed_accuracy,
            "observed_delta_macro_f1": metric["macro_f1"] - base["macro_f1"],
            "sum_single_delta_macro_f1": summed_macro,
            "interaction_macro_f1": metric["macro_f1"] - base["macro_f1"] - summed_macro,
        })
    write_csv(run_root / "analysis" / "single_handler_effects.csv", single_rows)
    write_csv(run_root / "analysis" / "leave_one_out_effects.csv", loo_rows)
    write_csv(run_root / "analysis" / "interaction_analysis.csv", interaction_rows)
    return {"single": single_rows, "leave_one_out": loo_rows, "interactions": interaction_rows}
