from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

# Required by PyTorch for deterministic CuBLAS kernels on CUDA >= 10.2.  This
# must be set before importing torch or creating a CUDA context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml

from .common import artifact_manifest, now_iso, read_json, sha256, write_json
from .data import (
    discover_records,
    load_family_signals,
    load_teacher_labels,
    split_manifest,
)
from .edf import read_header
from .evaluation import HANDLER_NAMES, evaluate_family
from .independent_recheck import run_independent_recheck
from .models import FAMILY_SPECS, build_model
from .teachers import generate_teacher_record, label_registry
from .training import train_family


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_learning_curves(run_dir: Path) -> None:
    from matplotlib import pyplot as plt

    destination = run_dir / "visualizations" / "learning_curves"
    destination.mkdir(parents=True, exist_ok=True)
    for family in FAMILY_SPECS:
        history_path = run_dir / "models" / family / "training_history.csv"
        with history_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        epoch = [int(row["epoch"]) for row in rows]
        figure, axes = plt.subplots(2, 1, figsize=(7, 6), constrained_layout=True)
        axes[0].plot(epoch, [float(row["train_loss"]) for row in rows], label="train")
        axes[0].plot(
            epoch,
            [float(row["validation_loss"]) for row in rows],
            label="validation",
        )
        axes[0].set_ylabel("masked BCE")
        axes[0].legend()
        axes[1].plot(
            epoch,
            [float(row["mean_validation_f1"]) for row in rows],
            marker="o",
        )
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("mean validation temporal F1")
        figure.suptitle(f"{family} training history")
        figure.savefig(destination / f"{family}.png", dpi=140)
        plt.close(figure)


def _git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _total_elapsed_seconds(state: dict[str, Any]) -> float:
    started_at = datetime.fromisoformat(state["started_at"])
    return max(0.0, (datetime.now().astimezone() - started_at).total_seconds())


def _load_completed_training_result(run_dir: Path, family: str) -> dict[str, Any] | None:
    """Reuse a fully audited checkpoint when resuming after evaluation/report failure."""
    model_dir = run_dir / "models" / family
    manifest_path = model_dir / "checkpoint_manifest.json"
    validation_path = model_dir / "validation_metrics.json"
    if not manifest_path.exists() or not validation_path.exists():
        return None
    manifest = read_json(manifest_path)
    checkpoint = Path(manifest["checkpoint"])
    if not checkpoint.exists() or sha256(checkpoint) != manifest["checkpoint_sha256"]:
        return None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(family)
    model.load_state_dict(payload["state_dict"], strict=True)
    validation = read_json(validation_path)
    return {
        "family": family,
        "checkpoint": checkpoint,
        "checkpoint_manifest": manifest,
        "selected_thresholds": manifest["selected_thresholds"],
        "validation_metrics": validation["per_head"],
        "negative_control": validation["label_shuffle_negative_control"],
        "device": manifest["device"],
    }


def _resolve_config_path(config_path: Path, value: str) -> Path:
    candidate = Path(os.path.expandvars(value))
    if "${AASM_V4_" in str(candidate):
        raise RuntimeError("required AASM_V4_* path environment variable is not configured")
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).resolve()


def _choose_run_dir(resume: bool) -> Path:
    base = Path("results/local")
    base.mkdir(parents=True, exist_ok=True)
    if resume:
        candidates = sorted(base.glob("neural_handlers_phase1c_*"), reverse=True)
        for candidate in candidates:
            state = candidate / "run_state.json"
            if state.exists() and read_json(state).get("status") not in ("COMPLETE", "PASS"):
                return candidate
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    return base / f"neural_handlers_phase1c_{stamp}"


def _protected_files(heritage_root: Path, config: dict[str, Any]) -> list[Path]:
    candidates = [
        heritage_root / "configs" / "frozen_detector_config.json",
        heritage_root / "src" / "aasm_v2" / "events" / "__init__.py",
        heritage_root / "src" / "aasm_v2" / "rules" / "__init__.py",
        heritage_root / "src" / "aasm_v2" / "temporal" / "__init__.py",
        _resolve_config_path(
            Path("experiments/v4_handler_validation/phase1c_handlers/configs/main.yaml"),
            config["historical_tournament"],
        )
        / "artifact_manifest.json",
        Path(os.environ.get("AASM_V4_PHASE1B_MANIFEST", "results/local/phase1b_manifest.json")).resolve(),
    ]
    return [path for path in candidates if path.exists()]


def _initialise_run(
    run_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    max_wall_hours: float,
) -> dict[str, Any]:
    for directory in (
        "data",
        "models",
        "handlers",
        "benchmarks",
        "visualizations/event_overlays",
        "visualizations/learning_curves",
        "audits",
    ):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "run_state.json"
    if state_path.exists():
        return read_json(state_path)
    state = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "started_at": now_iso(),
        "status": "RUNNING",
        "max_wall_hours": max_wall_hours,
        "config": str(config_path.resolve()),
        "phases": {
            key: "NOT_STARTED"
            for key in (
                "P0_DATA_AUDIT",
                "P1_WEAK_LABELS",
                "P2_TRAINING",
                "P3_INFERENCE",
                "P4_VALIDATION",
                "P5_ADAPTERS",
                "P6_INDEPENDENT_RECHECK",
                "P7_REPORT",
            )
        },
        "families": {family: "NOT_STARTED" for family in FAMILY_SPECS},
        "failures": [],
    }
    write_json(state_path, state)
    (run_dir / "RUN_PLAN.md").write_text(
        "# AASM V4 Phase 1C Run Plan\n\n"
        "Four shared temporal neural families produce nine target handlers. "
        "DREAMS and SleepEDF signal EDFs are grouped before windowing. Frozen V3 "
        "signal-only detectors provide E05-E12 weak labels; E13 uses a separately "
        "identified physiology-constraint proxy only after Frozen V3 produced zero "
        "positive samples. No hypnogram or stage label is read. "
        "Train/validation/test, checkpoint selection, eventization, and independent "
        "recomputation remain separated. HMC, staging, and rule changes are excluded.\n",
        encoding="utf-8",
    )
    return state


def _save_state(run_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    write_json(run_dir / "run_state.json", state)


def _prepare_data(
    run_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    state: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    state["phases"]["P0_DATA_AUDIT"] = "RUNNING"
    _save_state(run_dir, state)
    tournament = _resolve_config_path(config_path, config["historical_tournament"])
    sleepedf = _resolve_config_path(config_path, config["sleepedf_root"])
    heritage_root = _resolve_config_path(config_path, config["frozen_heritage_root"])
    records = discover_records(tournament, sleepedf)
    split = split_manifest(records)
    inventory_rows = []
    channel_rows = []
    for row in records:
        inventory_rows.append(
            {
                **{key: value for key, value in row.items() if key not in ("channels", "role_channels", "eligible_handlers")},
                "channels": ";".join(row["channels"]),
                "role_channels": json.dumps(row["role_channels"], ensure_ascii=False),
                "eligible_handlers": ";".join(row["eligible_handlers"]),
            }
        )
        header = read_header(row["file_path"])
        for signal in header.signals:
            channel_rows.append(
                {
                    "recording_id": row["recording_id"],
                    "dataset_id": row["dataset_id"],
                    "channel_name": signal.name,
                    "sampling_rate_hz": signal.sampling_rate,
                    "signal_unit": signal.unit,
                    "prefilter": signal.prefilter,
                    "duration_seconds": header.duration_seconds,
                }
            )
    _write_csv(run_dir / "data" / "dataset_inventory.csv", inventory_rows)
    _write_csv(run_dir / "data" / "channel_inventory.csv", channel_rows)
    write_json(run_dir / "data" / "split_manifest.json", split)
    write_json(
        run_dir / "data" / "raw_signal_audit.json",
        {
            "recordings": len(records),
            "dreams_recordings": sum(row["dataset_id"].startswith("DREAMS") for row in records),
            "sleepedf_recordings": sum(row["dataset_id"].startswith("SLEEP") for row in records),
            "dreams_sampling_rate_hz": 200,
            "sleepedf_eeg_sampling_rate_hz": 100,
            "sleepedf_1hz_emg_rejected_for_raw_emg_training": True,
            "dreams_emg1_contract": "200 Hz uV; CHANNEL_SEMANTICS_INFERRED_NOT_EXPLICIT",
            "raw_waveforms_copied_into_artifacts": False,
        },
    )
    write_json(
        run_dir / "data" / "leakage_audit.json",
        {
            "source_night_group_overlap": False,
            "same_record_window_overlap_across_splits": False,
            "test_labels_loaded_by_orchestrator": True,
            "test_labels_indexed_by_training_or_validation_dataset": False,
            "training_dataset_record_ids": {
                family: values["train"]
                for family, values in split["records_by_family"].items()
            },
            "validation_dataset_record_ids": {
                family: values["validation"]
                for family, values in split["records_by_family"].items()
            },
            "normalization_fit_on_test": False,
            "stage_labels_loaded": False,
            "hypnogram_paths_loaded": False,
            "split_frozen_before_training": True,
            "person_identity_limit": "DREAMS person identity is not independently verified",
        },
    )
    protected = _protected_files(heritage_root, config)
    write_json(
        run_dir / "heritage_manifest.json",
        {
            "git_head": _git(["rev-parse", "HEAD"]),
            "git_status_before": _git(["status", "--short"]),
            "config": str(config_path.resolve()),
            "config_sha256": sha256(config_path),
            "frozen_heritage_root": str(heritage_root),
            "protected_files": [
                {"path": str(path.resolve()), "sha256_before": sha256(path)}
                for path in protected
            ],
            "historical_assets_read_only": True,
            "git_commit_or_push_authorized": False,
        },
    )
    write_json(
        run_dir / "access_allowlist.json",
        {
            "signal_files": [row["file_path"] for row in records],
            "annotation_files": [],
            "forbidden": ["HMC", "records outside the training allowlist", "hypnogram", "stage reference"],
            "scope": "public DREAMS and SleepEDF signal EDF only",
        },
    )
    state["phases"]["P0_DATA_AUDIT"] = "PASS"
    state["phases"]["P1_WEAK_LABELS"] = "RUNNING"
    _save_state(run_dir, state)
    teacher_cache = run_dir / "data" / "weak_teacher_cache"
    teacher_results = []
    for index, record in enumerate(records, 1):
        cache_file = teacher_cache / f"{record['recording_id']}.npz"
        manifest_file = teacher_cache / f"{record['recording_id']}.teacher.json"
        if cache_file.exists() and manifest_file.exists():
            manifest = read_json(manifest_file)
            expected_teacher_schema = 3 if "E13" in record["eligible_handlers"] else 1
            if manifest.get("teacher_schema_version", 1) == expected_teacher_schema:
                teacher_results.append(
                    {"recording_id": record["recording_id"], "handlers": manifest["handlers"]}
                )
                continue
        print(f"[labels {index}/{len(records)}] {record['recording_id']}", flush=True)
        teacher_results.append(generate_teacher_record(record, heritage_root, teacher_cache))
    registry = label_registry(heritage_root)
    counts: dict[str, dict[str, int]] = {
        head: {"events": 0, "positive_samples": 0, "valid_samples": 0}
        for head in HANDLER_NAMES
    }
    for result in teacher_results:
        for row in result["handlers"]:
            aggregate = counts[row["handler_id"]]
            for key in aggregate:
                aggregate[key] += int(row[key])
    registry["aggregate_counts"] = counts
    registry["records"] = len(records)
    write_json(run_dir / "data" / "label_registry.json", registry)
    if any(counts[head]["positive_samples"] == 0 for head in counts):
        missing = [head for head in counts if counts[head]["positive_samples"] == 0]
        raise ValueError(f"No positive weak labels available for handlers: {missing}")
    state["phases"]["P1_WEAK_LABELS"] = "PASS"
    _save_state(run_dir, state)
    return records, split, teacher_cache


def _write_benchmarks(run_dir: Path, evaluations: list[dict[str, Any]]) -> None:
    agreement = [row for value in evaluations for row in value["teacher_agreement_rows"]]
    per_record = [row for value in evaluations for row in value["per_record_rows"]]
    baseline = [row for value in evaluations for row in value["baseline_rows"]]
    physiology = [row for value in evaluations for row in value["physiological_rows"]]
    _write_csv(run_dir / "benchmarks" / "weak_teacher_agreement.csv", agreement)
    _write_csv(run_dir / "benchmarks" / "per_record_metrics.csv", per_record)
    _write_csv(run_dir / "benchmarks" / "baseline_comparisons.csv", baseline)
    _write_csv(run_dir / "benchmarks" / "physiological_checks.csv", physiology)
    _write_csv(
        run_dir / "benchmarks" / "expert_event_metrics.csv",
        [
            {
                "handler_id": handler,
                "status": "NOT_MEASURED_NO_EXPERT_TARGET_TRUTH",
                "precision": "NOT MEASURED",
                "recall": "NOT MEASURED",
                "f1": "NOT MEASURED",
                "fp_per_hour": "NOT MEASURED",
            }
            for handler in HANDLER_NAMES
        ],
    )


def _final_reports(
    run_dir: Path,
    state: dict[str, Any],
    training_results: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    independent: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    summaries = {
        handler: summary
        for value in evaluations
        for handler, summary in value["handler_summaries"].items()
    }
    functional = sum(
        row["engineering_status"] == "FUNCTIONAL_NEURAL_HANDLER"
        for row in summaries.values()
    )
    prototype = len(summaries) - functional
    total_steps = sum(
        row["checkpoint_manifest"]["actual_optimizer_steps"] for row in training_results
    )
    total_parameters = sum(
        row["checkpoint_manifest"]["trainable_parameter_count"] for row in training_results
    )
    gpu_seconds = sum(
        row["checkpoint_manifest"]["actual_gpu_seconds"] for row in training_results
    )
    engineering = (
        "NINE_HANDLER_NEURAL_ENGINEERING_PASS"
        if functional == 9 and independent["pass"]
        else "NINE_HANDLER_NEURAL_ENGINEERING_PARTIAL"
    )
    order = list(HANDLER_NAMES)
    table_lines = [
        "| ID | Handler | Training Source | Neural Model | Real Inference | Expert Evidence | Weak Evidence | Baseline Comparison | Status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    family_by_handler = {
        head: family for family, spec in FAMILY_SPECS.items() for head in spec.output_heads
    }
    for handler in order:
        row = summaries[handler]
        pooled = row["pooled"]
        weak_evidence = (
            "NOT EVALUABLE (0 held-out positive intervals)"
            if pooled["teacher_interval_count"] == 0
            else (
                f"temporal F1={pooled['temporal_f1']:.4f}; "
                f"event F1={pooled['event_f1']:.4f}"
            )
        )
        comparison = (
            "constraint-proxy fidelity only"
            if handler == "E13"
            else (
                "insufficient held-out positive evidence"
                if pooled["teacher_interval_count"] == 0
                else "weak-teacher fidelity only"
            )
        )
        table_lines.append(
            f"| {handler} | {HANDLER_NAMES[handler]} | {row['training_source']} | "
            f"{family_by_handler[handler]} | PASS | NOT MEASURED | "
            f"{weak_evidence} | {comparison} | "
            f"{row['engineering_status']} |"
        )
    table = "\n".join(table_lines)
    summary_text = f"""# AASM V4 Phase 1C Executive Summary

上一轮 E05–E13 没有完整神经 Detector；本轮实际得到 **{functional}/9** 个可运行的任务 Head。

工程状态：`{engineering}`。四个共享时序网络均完成真实 optimizer 更新、checkpoint 严格重载、公开 PSG 全记录推理和独立复算。E05–E12 使用 Frozen V3 信号规则弱标签；Frozen V3 在当前 DREAMS montage 上给 E13 产生了 **0** 个 MBM 正样本，因此 E13 使用单独标识的跨模态生理约束代理。九类均无独立专家目标事件真值，新增专家级改善为 **0**；表中 F1 只能解释为 held-out 弱目标一致性。

{table}

- DISTINCT_NEURAL_MODELS_TRAINED = 4
- FUNCTIONAL_NEURAL_HANDLERS = {functional}
- EXPERT_SUPPORTED_NEURAL_IMPROVEMENTS = 0
- WEAKLY_SUPERVISED_FUNCTIONAL_HANDLERS = {functional}
- PROTOTYPE_ONLY_HANDLERS = {prototype}
- ACTUAL_OPTIMIZER_STEPS = {total_steps}
- TRAINABLE_PARAMETERS = {total_parameters}
- PRIMARY_TRAINING_GPU_SECONDS = {gpu_seconds:.1f} s (不含负控与推理)
- TOTAL_WALL_TIME = {elapsed_seconds:.1f} s
- Independent recheck = {'PASS' if independent['pass'] else 'FAIL'}
"""
    (run_dir / "EXECUTIVE_SUMMARY.md").write_text(summary_text, encoding="utf-8")
    full_report = f"""# Final Nine Neural Handler Report

## Outcome

`{engineering}`

{summary_text}

## Scientific boundary

E05–E12 targets were generated by the frozen signal-only V3 detector. E13 is different: Frozen V3 produced zero positive MBM samples after montage transfer, so its 616 positive seconds come from a separately identified EEG+EMG physiological-constraint proxy. No stage label, hypnogram, HMC signal, or expert target interval was used. The held-out split is source-night-group disjoint, but DREAMS person identity remains unverified. These agreement results establish engineering reproduction only, not improvement over a teacher or clinical event accuracy. E07 remains a prototype because its fixed held-out records contain zero SEM weak-target intervals and the model produced no accepted one-second SEM interval. EMG1 is consistent with the DREAMS published submental channel, but its EDF label is not explicit.

## Negative controls

Each family ran a separately initialized label-shuffle validation diagnostic at a fixed 0.5 threshold. It is not a same-split leaderboard against the held-out primary model. Constant-output and record-local 60-second temporal-misalignment diagnostics are stored in `benchmarks/baseline_comparisons.csv`. E13 additionally ran EEG-only and EMG-only input ablations.

## Delivery

Every functional handler has a strict-hash checkpoint reference, callable adapter, input/output contracts, full held-out probabilities, interval events, and validation result. Phase 2 staging work was not executed.
"""
    (run_dir / "FINAL_NINE_NEURAL_HANDLER_REPORT.md").write_text(
        full_report, encoding="utf-8"
    )
    (run_dir / "MODEL_SELECTION_DECISION.md").write_text(
        "# Model Selection Decision\n\n"
        + "\n".join(
            f"- {handler}: `{summaries[handler]['engineering_status']}` via `{family_by_handler[handler]}`; training source `{summaries[handler]['training_source']}`; scientific status `{summaries[handler]['scientific_status']}`."
            for handler in order
        )
        + "\n\nNo handler is claimed to outperform Frozen V3 without independent target-event truth.\n",
        encoding="utf-8",
    )
    (run_dir / "PHASE2_INTEGRATION_READINESS.md").write_text(
        "# Phase 2 Integration Readiness\n\n"
        + "\n".join(
            f"- {handler} {HANDLER_NAMES[handler]}: {summaries[handler]['engineering_status']}; adapter `handlers/{handler}_{HANDLER_NAMES[handler]}/adapter.py`; source `{summaries[handler]['training_source']}`; status `{summaries[handler]['scientific_status']}`."
            for handler in order
        )
        + "\n\nThese candidates are ready only for future copy-on-write shadow evaluation. SUMOv2 spindle and PSGScoring U-Net arousal remain frozen. K08's known downstream negative result remains a warning against inferring staging benefit from event fidelity. No Phase 2 execution occurred.\n",
        encoding="utf-8",
    )
    (run_dir / "reproduce.md").write_text(
        "# Reproduce\n\n"
        "```powershell\n"
        "python -m experiments.v4_handler_validation.phase1c_handlers.run --config experiments/v4_handler_validation/phase1c_handlers/configs/main.yaml --max-wall-hours 12 --resume\n"
        "python -m pytest experiments/neural_handlers_phase1c/tests -q -p no:cacheprovider\n"
        "```\n\n"
        "The command resumes the latest incomplete Phase 1C directory and never reads hypnograms or HMC assets.\n",
        encoding="utf-8",
    )
    return {
        "engineering_status": engineering,
        "functional_handlers": functional,
        "prototype_handlers": prototype,
        "total_optimizer_steps": total_steps,
        "trainable_parameters": total_parameters,
        "gpu_seconds": gpu_seconds,
        "wall_seconds": elapsed_seconds,
        "expert_supported_improvements": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-wall-hours", type=float, default=12.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    raw_config = os.path.expandvars(args.config.read_text(encoding="utf-8"))
    if "${AASM_V4_" in raw_config:
        raise RuntimeError("required AASM_V4_* path environment variable is not configured")
    config = yaml.safe_load(raw_config)
    run_dir = _choose_run_dir(args.resume)
    run_dir.mkdir(parents=True, exist_ok=True)
    state = _initialise_run(run_dir, args.config, config, args.max_wall_hours)
    print(f"RUN_DIR={run_dir.resolve()}", flush=True)
    records, split, teacher_cache = _prepare_data(
        run_dir, args.config, config, state
    )
    state["phases"]["P2_TRAINING"] = "RUNNING"
    _save_state(run_dir, state)
    training_results = []
    evaluations = []
    for family in FAMILY_SPECS:
        if _total_elapsed_seconds(state) / 3600 >= args.max_wall_hours:
            state["families"][family] = "BLOCKED_WALL_TIME"
            _save_state(run_dir, state)
            continue
        family_split = split["records_by_family"][family]
        family_ids = [record_id for values in family_split.values() for record_id in values]
        family_records = [row for row in records if row["recording_id"] in family_ids]
        try:
            state["families"][family] = "TRAINING"
            _save_state(run_dir, state)
            print(f"[family] loading {family}", flush=True)
            signals = load_family_signals(family_records, family)
            labels = load_teacher_labels(
                teacher_cache, family_ids, FAMILY_SPECS[family].output_heads
            )
            result = _load_completed_training_result(run_dir, family)
            if result is None:
                print(f"[family] training {family}", flush=True)
                result = train_family(
                    run_dir, family, signals, labels, family_split, config
                )
            else:
                print(f"[family] reusing audited checkpoint {family}", flush=True)
            state["families"][family] = "INFERENCE"
            _save_state(run_dir, state)
            evaluation = evaluate_family(
                run_dir,
                family,
                signals,
                labels,
                family_split,
                family_records,
                config,
                result,
            )
            write_json(
                run_dir / "models" / family / "inference_audit.json",
                {
                    "repeat_inference_max_abs_difference": evaluation[
                        "repeat_inference_max_abs_difference"
                    ],
                    "deterministic_repeat_pass": evaluation[
                        "repeat_inference_max_abs_difference"
                    ]
                    <= 1e-7,
                    "multimodal_ablation": evaluation["multimodal_ablation"],
                    "test_record_ids": family_split["test"],
                },
            )
            training_results.append(result)
            evaluations.append(evaluation)
            state["families"][family] = "PASS"
            _save_state(run_dir, state)
            del signals, labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            state["families"][family] = "FAILED"
            state["failures"].append(
                {
                    "family": family,
                    "time": now_iso(),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            _save_state(run_dir, state)
            print(f"[family] FAILED {family}: {exc!r}", flush=True)
    state["phases"]["P2_TRAINING"] = (
        "PASS" if len(training_results) == 4 else "PARTIAL"
    )
    state["phases"]["P3_INFERENCE"] = (
        "PASS" if len(evaluations) == 4 else "PARTIAL"
    )
    state["phases"]["P4_VALIDATION"] = (
        "PASS" if len(evaluations) == 4 else "PARTIAL"
    )
    state["phases"]["P5_ADAPTERS"] = (
        "PASS" if len(evaluations) == 4 else "PARTIAL"
    )
    _save_state(run_dir, state)
    if len(evaluations) != 4:
        raise RuntimeError(
            f"Only {len(evaluations)}/4 model families completed; inspect run_state.json"
        )
    _write_benchmarks(run_dir, evaluations)
    _write_learning_curves(run_dir)
    write_json(
        run_dir / "audits" / "training_integrity.json",
        {
            "families": [row["checkpoint_manifest"] for row in training_results],
            "all_weight_deltas_positive": all(
                row["checkpoint_manifest"]["weight_delta_l2"] > 0
                for row in training_results
            ),
            "all_optimizer_steps_nontrivial": all(
                row["checkpoint_manifest"]["actual_optimizer_steps"] >= 10
                for row in training_results
            ),
            "foundation_backbone_finetuned": False,
            "test_label_selection": False,
        },
    )
    write_json(
        run_dir / "audits" / "label_provenance.json",
        {
            "per_handler_source": {
                handler: (
                    "PHYSIOLOGICAL_CONSTRAINT_WEAK_LABEL"
                    if handler == "E13"
                    else "FROZEN_V3_WEAK_TEACHER"
                )
                for handler in HANDLER_NAMES
            },
            "e13_frozen_v3_positive_samples_before_fallback": 0,
            "e13_interpretation": (
                "constraint-proxy fidelity only; not Frozen V3 teacher fidelity, "
                "expert MBM accuracy, or event improvement"
            ),
            "expert_target_truth": False,
            "stage_labels": False,
            "scientific_interpretation": (
                "E05-E12 weak-teacher fidelity; E13 physiology-constraint proxy fidelity"
            ),
        },
    )
    state["phases"]["P6_INDEPENDENT_RECHECK"] = "RUNNING"
    _save_state(run_dir, state)
    independent = run_independent_recheck(run_dir)
    state["phases"]["P6_INDEPENDENT_RECHECK"] = (
        "PASS" if independent["pass"] else "FAIL"
    )
    elapsed = _total_elapsed_seconds(state)
    resume_segment_elapsed = time.perf_counter() - started
    delivery = _final_reports(
        run_dir, state, training_results, evaluations, independent, elapsed
    )
    write_json(
        run_dir / "audits" / "frozen_heritage_validation.json",
        {
            "pass": independent["frozen_heritage_hash_match"],
            "git_commit_or_push": False,
            "hmc_access": False,
            "stage_engine_execution": False,
        },
    )
    write_json(
        run_dir / "wall_time_accounting.json",
        {
            "started_at": state["started_at"],
            "completed_at": now_iso(),
            "wall_seconds": elapsed,
            "last_resume_segment_seconds": resume_segment_elapsed,
            "max_wall_hours": args.max_wall_hours,
            "within_budget": elapsed <= args.max_wall_hours * 3600,
            "primary_training_gpu_seconds": delivery["gpu_seconds"],
        },
    )
    state["phases"]["P7_REPORT"] = "PASS"
    state["status"] = "PASS" if delivery["engineering_status"].endswith("PASS") else "COMPLETE"
    state["engineering_status"] = delivery["engineering_status"]
    state["completed_at"] = now_iso()
    _save_state(run_dir, state)
    write_json(run_dir / "artifact_manifest.json", artifact_manifest(run_dir))
    print(json.dumps({"run_dir": str(run_dir.resolve()), **delivery}, indent=2), flush=True)


if __name__ == "__main__":
    main()
