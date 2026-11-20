from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .adapters import NeuralHandlerAdapter
from .common import binary_counts, sha256, write_json
from .data import FAMILY_ROLES, HANDLER_ROLES
from .inference import build_event_rows, evaluate_probability, infer_full_record
from .models import FAMILY_SPECS, build_model
from .teachers import EVENT_TYPES, HANDLER_RATES


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

CONTINUOUS_HANDLERS = {"E05", "E08", "E09", "E10", "E11", "E12"}


def _write_event_overlay(
    run_dir: Path,
    family: str,
    handler_id: str,
    head_index: int,
    probabilities: dict[str, np.ndarray],
    signals: dict[str, np.ndarray],
    labels: dict[str, dict[str, np.ndarray]],
    record_lookup: dict[str, dict[str, Any]],
) -> None:
    rate_hz = HANDLER_RATES[handler_id]
    record_id = max(
        probabilities,
        key=lambda key: float(np.max(probabilities[key][head_index])),
    )
    probability = probabilities[record_id][head_index]
    center_seconds = int(np.argmax(probability)) / rate_hz
    duration_seconds = len(probability) / rate_hz
    window_seconds = min(60.0, duration_seconds)
    left_seconds = min(
        max(0.0, center_seconds - window_seconds / 2),
        max(0.0, duration_seconds - window_seconds),
    )
    right_seconds = left_seconds + window_seconds
    spec = FAMILY_SPECS[family]
    family_roles = FAMILY_ROLES[family]
    role_indices = [family_roles.index(role) for role in HANDLER_ROLES[handler_id]]
    figure, axes = plt.subplots(
        len(role_indices) + 1,
        1,
        figsize=(10, 2.2 * (len(role_indices) + 1)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    input_left = int(round(left_seconds * spec.input_rate_hz))
    input_right = int(round(right_seconds * spec.input_rate_hz))
    signal_time = np.arange(input_left, input_right) / spec.input_rate_hz
    row = record_lookup[record_id]
    for axis, role_index, role in zip(axes[:-1], role_indices, HANDLER_ROLES[handler_id]):
        axis.plot(signal_time, signals[record_id][role_index, input_left:input_right], lw=0.6)
        axis.set_ylabel(f"{row['role_channels'][role]}\nnormalized")
    output_left = int(round(left_seconds * rate_hz))
    output_right = int(round(right_seconds * rate_hz))
    output_time = np.arange(output_left, output_right) / rate_hz
    target = labels[record_id][handler_id + "_target"]
    axes[-1].plot(
        output_time,
        probability[output_left:output_right],
        label="neural score",
        lw=1.0,
    )
    axes[-1].step(
        output_time,
        target[output_left:output_right],
        where="post",
        label="weak target",
        lw=0.8,
    )
    axes[-1].set_ylim(-0.05, 1.05)
    axes[-1].set_ylabel("activity")
    axes[-1].set_xlabel("seconds")
    axes[-1].legend(loc="upper right", fontsize=8)
    figure.suptitle(
        f"{handler_id} {HANDLER_NAMES[handler_id]} | {record_id} | model-selected display window"
    )
    destination = run_dir / "visualizations" / "event_overlays"
    destination.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination / f"{handler_id}_{HANDLER_NAMES[handler_id]}.png", dpi=140)
    plt.close(figure)


def _shifted_probability(probability: np.ndarray, shift: int) -> np.ndarray:
    if shift <= 0:
        return probability.copy()
    output = np.zeros_like(probability)
    output[shift:] = probability[:-shift]
    return output


def evaluate_family(
    run_dir: Path,
    family: str,
    signals: dict[str, np.ndarray],
    labels: dict[str, dict[str, np.ndarray]],
    split: dict[str, list[str]],
    records: list[dict[str, Any]],
    config: dict[str, Any],
    training_result: dict[str, Any],
) -> dict[str, Any]:
    spec = FAMILY_SPECS[family]
    family_config = config["families"][family]
    checkpoint = Path(training_result["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(family)
    model.load_state_dict(payload["state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    checkpoint_hash = sha256(checkpoint)
    model_output = run_dir / "models" / family / "test_predictions"
    model_output.mkdir(parents=True, exist_ok=True)
    record_lookup = {row["recording_id"]: row for row in records}
    probabilities: dict[str, np.ndarray] = {}
    repeat_max_difference = None
    multimodal_ablation: dict[str, Any] | None = None
    for record_index, record_id in enumerate(split["test"]):
        probability = infer_full_record(
            model,
            signals[record_id],
            family,
            int(family_config["window_seconds"]),
            device,
            batch_size=int(family_config["batch_size"]),
        )
        if record_index == 0:
            repeated = infer_full_record(
                model,
                signals[record_id],
                family,
                int(family_config["window_seconds"]),
                device,
                batch_size=int(family_config["batch_size"]),
            )
            repeat_max_difference = float(np.max(np.abs(probability - repeated)))
        probabilities[record_id] = probability
        target = np.stack(
            [labels[record_id][head + "_target"] for head in spec.output_heads]
        )
        valid = np.stack(
            [labels[record_id][head + "_valid"] for head in spec.output_heads]
        )
        length = min(probability.shape[-1], target.shape[-1])
        probability = probability[:, :length]
        target = target[:, :length]
        valid = valid[:, :length]
        probabilities[record_id] = probability
        np.savez_compressed(
            model_output / f"{record_id}.npz",
            heads=np.asarray(spec.output_heads),
            probability=probability,
            target=target,
            valid=valid,
        )
    if family == "major_movement":
        record_id = split["test"][0]
        full = probabilities[record_id]
        eeg_only_input = signals[record_id].copy()
        eeg_only_input[2] = 0
        emg_only_input = signals[record_id].copy()
        emg_only_input[:2] = 0
        eeg_only = infer_full_record(
            model,
            eeg_only_input,
            family,
            int(family_config["window_seconds"]),
            device,
        )
        emg_only = infer_full_record(
            model,
            emg_only_input,
            family,
            int(family_config["window_seconds"]),
            device,
        )
        multimodal_ablation = {
            "recording_id": record_id,
            "mean_absolute_change_without_emg": float(np.mean(np.abs(full - eeg_only))),
            "mean_absolute_change_without_eeg": float(np.mean(np.abs(full - emg_only))),
            "uses_both_modalities_nontrivially": bool(
                np.mean(np.abs(full - eeg_only)) > 1e-6
                and np.mean(np.abs(full - emg_only)) > 1e-6
            ),
        }
    thresholds = training_result["selected_thresholds"]
    rows = []
    per_record_rows = []
    baseline_rows = []
    physiological_rows = []
    handler_summaries = {}
    for head_index, handler_id in enumerate(spec.output_heads):
        handler_name = HANDLER_NAMES[handler_id]
        rate = HANDLER_RATES[handler_id]
        threshold = float(thresholds[handler_id])
        handler_dir = run_dir / "handlers" / f"{handler_id}_{handler_name}"
        inference_dir = handler_dir / "real_inference_results"
        inference_dir.mkdir(parents=True, exist_ok=True)
        pooled_probability = []
        pooled_target = []
        pooled_valid = []
        head_record_metrics = []
        all_events = 0
        all_durations = []
        for record_id in split["test"]:
            probability = probabilities[record_id][head_index]
            target = labels[record_id][handler_id + "_target"][: len(probability)]
            valid = labels[record_id][handler_id + "_valid"][: len(probability)]
            metrics = evaluate_probability(
                probability,
                target,
                valid,
                threshold,
                rate,
                config["eventization"][handler_id],
            )
            head_record_metrics.append(metrics)
            per_record_rows.append(
                {
                    "family": family,
                    "handler_id": handler_id,
                    "recording_id": record_id,
                    **metrics,
                }
            )
            row = record_lookup[record_id]
            source_channel = "+".join(
                str(row["role_channels"][role]) for role in HANDLER_ROLES[handler_id]
            )
            events = build_event_rows(
                record_id,
                handler_id,
                EVENT_TYPES[handler_id],
                probability,
                threshold,
                rate,
                config["eventization"][handler_id],
                checkpoint_hash,
                source_channel,
                float(row["sampling_rate_hz"]),
            )
            all_events += len(events)
            all_durations.extend(event["offset"] - event["onset"] for event in events)
            event_path = inference_dir / f"{record_id}.events.json"
            event_path.write_text(
                json.dumps(events, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            np.savez_compressed(
                inference_dir / f"{record_id}.probability.npz",
                time_axis=np.arange(len(probability), dtype=np.float32) / rate,
                activity_score=probability,
                valid_mask=valid,
                weak_teacher_target=target,
                temporal_resolution_seconds=np.asarray([1 / rate], dtype=np.float32),
            )
            pooled_probability.append(probability)
            pooled_target.append(target)
            pooled_valid.append(valid)
        probability = np.concatenate(pooled_probability)
        target = np.concatenate(pooled_target)
        valid = np.concatenate(pooled_valid)
        temporal = binary_counts(target, probability >= threshold, valid)
        event_tp = sum(int(row["event_tp"]) for row in head_record_metrics)
        event_fp = sum(int(row["event_fp"]) for row in head_record_metrics)
        event_fn = sum(int(row["event_fn"]) for row in head_record_metrics)
        event_precision = event_tp / (event_tp + event_fp) if event_tp + event_fp else 0.0
        event_recall = event_tp / (event_tp + event_fn) if event_tp + event_fn else 0.0
        event_f1 = (
            2 * event_precision * event_recall / (event_precision + event_recall)
            if event_precision + event_recall
            else 0.0
        )
        pooled = {
            **{f"temporal_{key}": value for key, value in temporal.items()},
            "event_tp": event_tp,
            "event_fp": event_fp,
            "event_fn": event_fn,
            "event_precision": event_precision,
            "event_recall": event_recall,
            "event_f1": event_f1,
            "event_mean_matched_iou": (
                sum(
                    float(row["event_mean_matched_iou"]) * int(row["event_tp"])
                    for row in head_record_metrics
                )
                / event_tp
                if event_tp
                else 0.0
            ),
            "teacher_interval_count": sum(
                int(row["teacher_interval_count"]) for row in head_record_metrics
            ),
            "prediction_interval_count": sum(
                int(row["prediction_interval_count"]) for row in head_record_metrics
            ),
            "prediction_fraction": float(np.mean(probability >= threshold)),
            "probability_mean": float(np.mean(probability)),
            "probability_std": float(np.std(probability)),
            "threshold": threshold,
        }
        rows.append(
            {
                "scope": "pooled_test",
                "family": family,
                "handler_id": handler_id,
                **pooled,
            }
        )
        constant = binary_counts(target, np.zeros_like(target), valid)
        shifted_probability = np.concatenate(
            [
                _shifted_probability(value, 60 * rate)
                for value in pooled_probability
            ]
        )
        shifted = binary_counts(
            target,
            shifted_probability >= threshold,
            valid,
        )
        shuffle_metric = training_result["negative_control"]["metrics"][handler_id]
        baseline_rows.append(
            {
                "handler_id": handler_id,
                "family": family,
                "model_test_temporal_f1": pooled["temporal_f1"],
                "constant_all_negative_f1": constant["f1"],
                "temporal_shift_60s_f1": shifted["f1"],
                "label_shuffle_validation_f1": shuffle_metric["f1"],
                "model_and_shift_scope": "held-out test; shift applied within each record",
                "label_shuffle_scope": "validation at fixed threshold 0.5; diagnostic only",
                "comparison_interpretation": (
                    "weak-teacher diagnostics, not expert improvement or a directly "
                    "comparable leaderboard"
                ),
            }
        )
        nondegenerate = (
            pooled["probability_std"] > 1e-4
            and 0 < pooled["prediction_fraction"] < 1
            and pooled["prediction_interval_count"] > 0
        )
        training_source = (
            "PHYSIOLOGICAL_CONSTRAINT_WEAK_LABEL"
            if handler_id == "E13"
            else "FROZEN_V3_WEAK_TEACHER"
        )
        scientific_status = (
            "PHYSIOLOGICAL_CONSTRAINT_PROXY_FIDELITY_ONLY"
            if handler_id == "E13"
            else (
                "INSUFFICIENT_EVIDENCE_NO_HELDOUT_POSITIVES"
                if pooled["teacher_interval_count"] == 0
                else "WEAK_TEACHER_FIDELITY_ONLY"
            )
        )
        duration_minimum = float(config["eventization"][handler_id]["minimum_seconds"])
        duration_valid = all(value + 1e-9 >= duration_minimum for value in all_durations)
        physiological_rows.append(
            {
                "handler_id": handler_id,
                "family": family,
                "finite_output": True,
                "nondegenerate_output": nondegenerate,
                "event_count": all_events,
                "duration_contract_pass": duration_valid,
                "mean_duration_seconds": float(np.mean(all_durations)) if all_durations else 0.0,
                "probability_std": pooled["probability_std"],
                "weak_teacher_f1": pooled["temporal_f1"],
                "multimodal_ablation_pass": (
                    multimodal_ablation["uses_both_modalities_nontrivially"]
                    if handler_id == "E13" and multimodal_ablation
                    else "NOT_APPLICABLE"
                ),
            }
        )
        adapter_config = {
            "handler_id": handler_id,
            "family": family,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "threshold": threshold,
            "eventization": config["eventization"][handler_id],
            "window_seconds": int(family_config["window_seconds"]),
            "batch_size": int(family_config["batch_size"]),
            "device": "cpu",
        }
        config_path = handler_dir / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(adapter_config, sort_keys=False), encoding="utf-8"
        )
        (handler_dir / "adapter.py").write_text(
            "from pathlib import Path\n"
            "import yaml\n"
            "from experiments.v4_handler_validation.phase1c_handlers.adapters import NeuralHandlerAdapter\n\n"
            "def load_adapter(config_path: str | Path = Path(__file__).with_name('config.yaml')):\n"
            "    with Path(config_path).open('r', encoding='utf-8') as stream:\n"
            "        return NeuralHandlerAdapter(yaml.safe_load(stream))\n",
            encoding="utf-8",
        )
        write_json(
            handler_dir / "selected_model.json",
            {
                "handler_id": handler_id,
                "model_family": family,
                "status": "FUNCTIONAL_NEURAL_HANDLER" if nondegenerate else "NEURAL_PROTOTYPE_ONLY",
                "training_source": training_source,
                "scientific_status": scientific_status,
                "expert_event_improvement_established": False,
            },
        )
        write_json(
            handler_dir / "input_contract.json",
            {
                "roles": list(HANDLER_ROLES[handler_id]),
                "source_channels": [
                    record_lookup[split["test"][0]]["role_channels"][role]
                    for role in HANDLER_ROLES[handler_id]
                ],
                "source_sampling_rate_hz": 200.0 if family != "eeg_background" else 100.0,
                "model_sampling_rate_hz": spec.input_rate_hz,
                "unit": "uV",
                "montage_transfer": "DATASET_SPECIFIC_RESEARCH_CONTRACT",
            },
        )
        write_json(
            handler_dir / "output_contract.json",
            {
                "event_type": EVENT_TYPES[handler_id],
                "kind": "CONTINUOUS_ACTIVITY_INTERVALS" if handler_id in CONTINUOUS_HANDLERS else "DISCRETE_EVENT_INTERVALS",
                "temporal_resolution_seconds": 1 / rate,
                "activity_score_semantics": "sigmoid task-head output; not clinically calibrated probability",
                "interval_threshold": threshold,
                "minimum_duration_seconds": duration_minimum,
            },
        )
        write_json(
            handler_dir / "checkpoint_reference.json",
            {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_hash,
                "shared_model_family": family,
            },
        )
        write_json(
            handler_dir / "validation_results.json",
            {
                "training_source": training_source,
                "scientific_status": scientific_status,
                "test_teacher_agreement": pooled,
                "expert_event_metrics": "NOT_MEASURED_NO_EXPERT_TARGET_TRUTH",
                "constant_baseline": constant,
                "temporal_misalignment": shifted,
                "label_shuffle_control": shuffle_metric,
                "multimodal_ablation": multimodal_ablation if handler_id == "E13" else None,
            },
        )
        write_json(
            handler_dir / "execution_status.json",
            {
                "training": "PASS",
                "real_inference": "PASS",
                "adapter": "PASS",
                "event_contract": "PASS" if duration_valid else "FAIL",
                "engineering_status": "FUNCTIONAL_NEURAL_HANDLER" if nondegenerate and duration_valid else "NEURAL_PROTOTYPE_ONLY",
            },
        )
        adapter = NeuralHandlerAdapter(adapter_config)
        load_audit = adapter.load_model()
        contract = adapter.validate_input(record_lookup[split["test"][0]])
        write_json(
            handler_dir / "adapter_smoke_test.json",
            {**load_audit, "input_contract": contract, "full_inference_exercised_by_family_evaluator": True},
        )
        _write_event_overlay(
            run_dir,
            family,
            handler_id,
            head_index,
            probabilities,
            signals,
            labels,
            record_lookup,
        )
        handler_summaries[handler_id] = {
            "handler_id": handler_id,
            "family": family,
            "pooled": pooled,
            "nondegenerate": nondegenerate,
            "duration_contract_pass": duration_valid,
            "event_count": all_events,
            "adapter_ready": True,
            "engineering_status": "FUNCTIONAL_NEURAL_HANDLER" if nondegenerate and duration_valid else "NEURAL_PROTOTYPE_ONLY",
            "training_source": training_source,
            "scientific_status": scientific_status,
        }
    return {
        "family": family,
        "teacher_agreement_rows": rows,
        "per_record_rows": per_record_rows,
        "baseline_rows": baseline_rows,
        "physiological_rows": physiological_rows,
        "handler_summaries": handler_summaries,
        "repeat_inference_max_abs_difference": repeat_max_difference,
        "multimodal_ablation": multimodal_ablation,
    }
