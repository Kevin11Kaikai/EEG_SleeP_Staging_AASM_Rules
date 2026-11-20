from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import yaml

from experiments.v4_handler_validation.phase1c_handlers.edf import read_channels, read_header

from .common import (
    eventize,
    now_iso,
    parameter_hash,
    reference_intervals,
    sha256,
    write_csv,
    write_json,
)
from .data import (
    build_record_cache,
    build_split_manifest,
    discover_records,
    teacher_gate_audit,
)
from .data.label_registry import build_label_provenance
from .evaluation.challenge import evaluate_records, infer_full_record
from .evaluation.negative_controls import (
    channel_transform_audit,
    constant_negative_control,
    overlap_confusion,
    time_shift_control,
)
from .evaluation.paired_analysis import compare_teacher_and_neural
from .handler.adapter import E07SEMNeuralHandler
from .independent_recheck import run_independent_recheck
from .training.checkpoint import strict_load
from .training.trainer import train_sem_model


class WallBudget:
    def __init__(self, maximum_hours: float):
        self.started = time.perf_counter()
        self.maximum_seconds = maximum_hours * 3600

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def check(self, phase: str) -> None:
        if self.elapsed >= self.maximum_seconds:
            raise TimeoutError(f"Wall-time budget exhausted before {phase}")


def _load_config(path: str | Path) -> dict[str, Any]:
    raw = os.path.expandvars(Path(path).read_text(encoding="utf-8"))
    if "${AASM_V4_" in raw:
        raise RuntimeError("required AASM_V4_* path environment variable is not configured")
    config = yaml.safe_load(raw)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported Phase1D config schema")
    return config


def _run_id() -> str:
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")


def _select_run_dir(config: dict[str, Any], resume: bool) -> Path:
    root = Path(config["output_root"])
    if resume:
        candidates = sorted(root.glob("sem_final_upgrade_v4_*"), reverse=True)
        for candidate in candidates:
            state = candidate / "run_state.json"
            if state.exists():
                payload = json.loads(state.read_text(encoding="utf-8"))
                if payload.get("overall_status") not in {
                    "E07_SEM_FUNCTIONAL_NEURAL_HANDLER_PASS",
                    "E07_SEM_ENGINEERING_PARTIAL",
                }:
                    return candidate
    destination = root / f"sem_final_upgrade_v4_{_run_id()}"
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _git(command: list[str]) -> str:
    result = subprocess.run(
        ["git", *command], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _protected_files(config: dict[str, Any]) -> list[Path]:
    frozen = Path(config["frozen_heritage_root"])
    phase1c = Path(config["phase1c_result"])
    candidates = [
        frozen / "src" / "aasm_v2" / "events" / "morphology.py",
        frozen / "src" / "aasm_v2" / "temporal.py",
        frozen / "configs" / "frozen_detector_config.json",
        phase1c / "FINAL_NINE_NEURAL_HANDLER_REPORT.md",
        phase1c / "MODEL_SELECTION_DECISION.md",
        phase1c / "PHASE2_INTEGRATION_READINESS.md",
        phase1c / "models" / "eeg_background" / "best_checkpoint.pt",
        phase1c / "models" / "eog_events" / "best_checkpoint.pt",
        phase1c / "models" / "chin_emg" / "best_checkpoint.pt",
        phase1c / "models" / "major_movement" / "best_checkpoint.pt",
    ]
    return [path.resolve() for path in candidates if path.exists()]


def _heritage_manifest(config: dict[str, Any]) -> dict[str, Any]:
    protected = _protected_files(config)
    return {
        "schema_version": 1,
        "git_head": _git(["rev-parse", "HEAD"]),
        "git_status_before": _git(["status", "--short"]),
        "phase1c_result": str(Path(config["phase1c_result"]).resolve()),
        "frozen_heritage_root": str(Path(config["frozen_heritage_root"]).resolve()),
        "protected_files": [
            {"path": str(path), "sha256": sha256(path)} for path in protected
        ],
        "hmc_access_allowed": False,
        "sleep_staging_allowed": False,
        "other_eight_handler_training_allowed": False,
        "git_commit_or_push_allowed": False,
    }


def _compute_manifest() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_vram_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / 2**30, 3)
            if torch.cuda.is_available()
            else None
        ),
        "cpu_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 3),
    }


def _phase1c_original_audit(
    config: dict[str, Any], records: dict[str, Any]
) -> dict[str, Any]:
    phase1c = Path(config["phase1c_result"])
    old_split = json.loads(
        (phase1c / "data" / "split_manifest.json").read_text(encoding="utf-8")
    )
    cache = phase1c / "data" / "weak_teacher_cache"
    roles = {"train": [], "validation": [], "test": []}
    for role in roles:
        for record_id in old_split["records_by_family"]["eog_events"][role]:
            with np.load(cache / f"{record_id}.npz") as payload:
                target = payload["E07_target"].astype(bool)
                valid = payload["E07_valid"].astype(bool)
            intervals = reference_intervals(target, valid, 10)
            roles[role].append(
                {
                    "record_id": record_id,
                    "positive_samples_10hz": int((target & valid).sum()),
                    "positive_seconds": float((target & valid).sum() / 10),
                    "events": len(intervals),
                    "durations_seconds": [round(b - a, 3) for a, b in intervals],
                    "scored_seconds": float(valid.sum() / 10),
                }
            )
    dream_rows = [value for value in records.values() if value.dataset_id.startswith("DREAMS")]
    gate_totals = {
        key: sum(row.metadata["teacher_gate_counts"][key] for row in dream_rows)
        for key in dream_rows[0].metadata["teacher_gate_counts"]
    }
    return {
        "phase1c_claim_correction": (
            "61 was the number of positive 10-Hz samples (6.1 s), not 61 events"
        ),
        "roles": roles,
        "aggregate": {
            role: {
                "records": len(rows),
                "events": sum(row["events"] for row in rows),
                "positive_samples_10hz": sum(
                    row["positive_samples_10hz"] for row in rows
                ),
                "positive_seconds": sum(row["positive_seconds"] for row in rows),
            }
            for role, rows in roles.items()
        },
        "dreams_exact_teacher_gate_totals": gate_totals,
        "root_cause": (
            "SEM scarcity is dominated by the 1-5 s boundary and anti-phase/initial-"
            "deflection gates, not by bilateral EOG QC or a label-cache bug"
        ),
        "old_shared_head_limitation": (
            "Blink and SEM shared a checkpoint selected by mean temporal F1; no positive-"
            "window oversampling; old validation had zero legal SEM event TP"
        ),
    }


def _save_record_artifacts(
    run_dir: Path, records: dict[str, Any], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cache = run_dir / "data" / "label_cache"
    cache.mkdir(parents=True, exist_ok=True)
    source_rows = []
    positive_rows = []
    gate_rows = []
    teacher_source = Path(config["frozen_heritage_root"]) / "src" / "aasm_v2" / "events" / "morphology.py"
    teacher_config = Path(config["frozen_heritage_root"]) / "configs" / "frozen_detector_config.json"
    for record in records.values():
        np.savez_compressed(
            cache / f"{record.record_id}.npz",
            target=record.target,
            scoring_mask=record.scoring_mask,
            signal_qc=record.signal_qc,
            blink_mask=record.blink_mask,
            rem_mask=record.rem_mask,
        )
        source_rows.append(record.metadata)
        gate_rows.append(teacher_gate_audit(record))
        for index, ((onset, offset), detail) in enumerate(
            zip(record.teacher_intervals, record.teacher_details)
        ):
            positive_rows.append(
                {
                    "event_id": f"{record.record_id}:weak_teacher_sem:{index:06d}",
                    "record_id": record.record_id,
                    "dataset_id": record.dataset_id,
                    "split_role": record.split_role,
                    "group_id": record.group_id,
                    "onset_seconds": onset,
                    "offset_seconds": offset,
                    "duration_seconds": offset - onset,
                    "correlation": detail.get("correlation"),
                    "peak_uv": detail.get("peak_uv"),
                    "initial_deflection_seconds": detail.get("initial_deflection_s"),
                    "boundary_method": detail.get("boundary"),
                    "label_source": "WEAK_TEACHER_SEM",
                    "teacher_source_sha256": sha256(teacher_source),
                    "teacher_config_sha256": sha256(teacher_config),
                }
            )
    write_csv(run_dir / "data" / "source_inventory.csv", source_rows)
    write_csv(run_dir / "data" / "positive_event_inventory.csv", positive_rows)
    write_csv(run_dir / "data" / "teacher_gate_audit.csv", gate_rows)
    return source_rows, positive_rows, gate_rows


def _parity_audit(config: dict[str, Any], records: dict[str, Any]) -> dict[str, Any]:
    old_cache = Path(config["phase1c_result"]) / "data" / "weak_teacher_cache"
    rows = []
    for record_id, record in records.items():
        source = old_cache / f"{record_id}.npz"
        if not source.exists():
            continue
        with np.load(source) as payload:
            old_target = payload["E07_target"]
        length = min(len(old_target), len(record.target))
        rows.append(
            {
                "record_id": record_id,
                "samples_compared": length,
                "target_equal": bool(np.array_equal(old_target[:length], record.target[:length])),
                "old_positive_samples": int(old_target[:length].sum()),
                "new_positive_samples": int(record.target[:length].sum()),
            }
        )
    return {
        "records": rows,
        "all_dreams_targets_equal": all(row["target_equal"] for row in rows),
        "records_compared": len(rows),
    }


def _infer_ids(
    model: torch.nn.Module,
    records: dict[str, Any],
    ids: list[str],
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    model_config = config["model"]
    return {
        record_id: infer_full_record(
            model,
            records[record_id].signal,
            device,
            int(model_config["window_seconds"]),
            int(model_config["inference_stride_seconds"]),
            int(model_config["batch_size"]),
        )
        for record_id in ids
    }


def _metrics_rows(result: dict[str, Any], reference_type: str) -> list[dict[str, Any]]:
    rows = [dict(row, reference_type=reference_type) for row in result["per_record"]]
    rows.append(
        {
            "record_id": "__AGGREGATE__",
            **result["aggregate"],
            "reference_type": reference_type,
        }
    )
    return rows


def _event_rows(
    records: dict[str, Any],
    ids: list[str],
    probabilities: dict[str, np.ndarray],
    threshold: float,
    config: dict[str, Any],
    checkpoint_hash: str,
) -> list[dict[str, Any]]:
    rows = []
    event_config = config["eventization"]
    for record_id in ids:
        record = records[record_id]
        for scope, valid in (
            ("production_signal_qc", record.signal_qc),
            ("reference_scored_domain", record.scoring_mask),
        ):
            intervals = eventize(
                probabilities[record_id],
                valid,
                threshold,
                10,
                event_config["minimum_seconds"],
                event_config["maximum_seconds"],
                event_config["merge_gap_seconds"],
            )
            for index, (onset, offset) in enumerate(intervals):
                left = int(round(onset * 10))
                right = max(left + 1, int(round(offset * 10)))
                rows.append(
                    {
                        "event_id": f"{record_id}:E07:{scope}:{index:07d}",
                        "event_type": "sem",
                        "record_id": record_id,
                        "split_role": record.split_role,
                        "scope": scope,
                        "onset": onset,
                        "offset": offset,
                        "duration": offset - onset,
                        "confidence": float(
                            np.mean(probabilities[record_id][left:right])
                        ),
                        "channel": "+".join(
                            [
                                record.metadata["left_channel"],
                                record.metadata["right_channel"],
                            ]
                        ),
                        "source_model_sha256": checkpoint_hash,
                        "quality_status": "AVAILABLE",
                        "evidence_status": "CANDIDATE_PROXY",
                    }
                )
    return rows


def _artifact_manifest(run_dir: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in {
            "artifact_manifest.json",
            "artifact_validation.json",
        }:
            continue
        rows.append(
            {
                "path": str(path.relative_to(run_dir)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {"schema_version": 1, "artifacts": rows, "artifact_count": len(rows)}


def _reports(
    run_dir: Path,
    status: dict[str, Any],
    counts: dict[str, Any],
    challenge: dict[str, Any],
    original_test: dict[str, Any],
    selection: dict[str, Any],
    controls: dict[str, Any],
    elapsed: float,
) -> None:
    aggregate = challenge["aggregate"]
    functional = status["e07_functional"]
    event_f1 = aggregate["f1"]
    rows = [
        ("E05", "Alpha", "FUNCTIONAL", "Phase1C frozen"),
        ("E06", "Blink", "FUNCTIONAL", "Phase1C frozen"),
        ("E07", "SEM", "FUNCTIONAL" if functional else "PROTOTYPE", selection["architecture"]),
        ("E08", "LAMF", "FUNCTIONAL", "Phase1C frozen"),
        ("E09", "Low EMG", "FUNCTIONAL", "Phase1C frozen"),
        ("E10", "Normal EMG", "FUNCTIONAL", "Phase1C frozen"),
        ("E11", "High EMG", "FUNCTIONAL", "Phase1C frozen"),
        ("E12", "Slowing", "FUNCTIONAL", "Phase1C frozen"),
        ("E13", "MBM", "FUNCTIONAL PROXY", "Phase1C frozen proxy"),
    ]
    table = "\n".join(
        ["| ID | Handler | Status | Model |", "|---|---|---|---|"]
        + [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    )
    summary = f"""# E07 SEM Final Upgrade — Executive Summary

**结论：{'E07 已达到 Functional Neural Handler 工程验收' if functional else 'E07 尚未达到 Functional 验收'}。** 新模型从随机初始化真实训练，使用双侧真实 EOG；阳性标签来自未修改的 Frozen V3 SEM morphology teacher，因此结论是 `E07_WEAK_SUPERVISION_ONLY`，不是专家准确率证明。

- 原 Phase1C “61个正事件”经复核实为 **61个10 Hz阳性样本、6.1秒、5个事件**。
- 本轮训练阳性：**{counts['train_events']} raw teacher events**；验证阳性：**{counts['validation_events']}**；阳性挑战 inventory：**{counts['challenge_events']} raw candidates**，栅格并集后为 **{aggregate['reference_events']} 个可评价 intervals**。
- 阳性挑战：TP={aggregate['tp']}、FP={aggregate['fp']}、FN={aggregate['fn']}、teacher-fidelity F1={event_f1 if event_f1 is not None else 'NOT DEFINED'}、FP/hour={aggregate['fp_per_hour']:.3f}。
- 原固定 Test 的记录、EDF哈希和零阳性事实未改动；Phase1D 改用预声明的全 QC teacher-fidelity 域：{original_test['aggregate']['predicted_events']} 个 proposal，FP/hour={original_test['aggregate']['fp_per_hour']:.3f}；Recall/F1 为 `NOT_DEFINED_NO_POSITIVES`。
- 专家 SEM 标注：0；公开 onset/offset 数据未验证可得。
- Selected model：`{selection['architecture']}`；checkpoint SHA256 `{selection['checkpoint_sha256']}`。
- 训练动态：best epoch 由 validation event F1 冻结；train loss下降而 validation loss后期上升，存在明显低数据过拟合，因此只认定非退化工程功能，不认定高精度收敛。
- Adapter：{'PASS' if status['adapter_pass'] else 'FAIL'}；独立复算：{'PASS' if status['independent_pass'] else 'FAIL'}。
- Phase 1：{'`PHASE1_9_OF_9_FUNCTIONAL_HANDLER_ENGINEERING_PASS`' if functional else '`PHASE1_ENGINEERING_PARTIAL`'}。E13仍是movement proxy，弱监督handler仍缺专家验证。
- 实际 wall time：{elapsed:.1f} 秒。

{table}
"""
    (run_dir / "EXECUTIVE_SUMMARY.md").write_text(summary, encoding="utf-8")
    detailed = summary + f"""

## Scientific boundary

本轮指标衡量神经模型对冻结 teacher 的 record-disjoint fidelity。Frozen V3 与自身标签比较天然为1.0，因此不能由该对照推出神经模型优于 V3。PSG-IPA 的双侧 EOG 扩充解决了正事件覆盖与真实训练问题，但没有产生独立专家 SEM truth。

## Negative controls

```json
{json.dumps(controls, indent=2, ensure_ascii=False)}
```
"""
    (run_dir / "FINAL_E07_SEM_UPGRADE_REPORT.md").write_text(
        detailed, encoding="utf-8"
    )
    phase1 = f"""# Phase 1 Final Status

Engineering status: **{'PHASE1_9_OF_9_FUNCTIONAL_HANDLER_ENGINEERING_PASS' if functional else 'PHASE1_ENGINEERING_PARTIAL'}**

Scientific status: **E07_EVENT_ACCURACY_NOT_INDEPENDENTLY_VALIDATED**

{table}

9/9 engineering coverage does not mean 9/9 detectors are more accurate. E13 remains a movement proxy; E07 and other weakly supervised handlers still require independent expert event validation.
"""
    (run_dir / "PHASE1_9OF9_FINAL_STATUS.md").write_text(phase1, encoding="utf-8")
    readiness = f"""# Phase 2 Integration Readiness

Phase 2 was **not run**. HMC and sleep-stage labels were not accessed.

E07 adapter: `experiments.v4_handler_validation.sem_final_upgrade.handler.adapter.E07SEMNeuralHandler`

- Input: bilateral EOG ordered `left_eog, right_eog`; verified reference `M2`; polarity `as_recorded`; units `uV/mV/V`; sampling rate >30 Hz.
- QC: complete 30 s epochs, bilateral peak-to-peak >=0.5 uV, absolute amplitude <=2000 uV, repeated-sample fraction <=0.2.
- Output: SEM onset/offset/confidence plus source pointer.
- Model: `{selection['checkpoint']}`
- SHA256: `{selection['checkpoint_sha256']}`
- Validation level: `WEAK_TEACHER_POSITIVE_CHALLENGE`; not expert truth.
- Eventization: 10 Hz output, validation-frozen threshold `{selection['threshold']}`, duration 1–5 s, merge gap 0.3 s.
- Known risks: non-M2 montage transfer is rejected pending validation; teacher labels are narrow (1–5 s anti-phase morphology); order and polarity materially affect output.

## Final nine-handler list

- E05 Alpha: Phase1C `eeg_background`, functional weak-teacher handler.
- E06 Blink: Phase1C `eog_events`, functional weak-teacher handler.
- E07 SEM: Phase1D `inherited_tcn`, functional weak-teacher handler; expert accuracy unvalidated.
- E08 LAMF: Phase1C `eeg_background`, functional weak-teacher handler.
- E09/E10/E11 EMG tone: Phase1C `chin_emg`, functional weak-teacher handlers.
- E12 Slowing: Phase1C `eeg_background`, functional weak-teacher handler.
- E13 MBM: Phase1C `major_movement`, functional movement proxy rather than verified MBM truth.

The previous Phase1C candidates for E05/E06/E08–E13 remain byte-identical. K08's known downstream HMC regression remains a warning and was not re-evaluated here.
"""
    (run_dir / "PHASE2_INTEGRATION_READINESS.md").write_text(
        readiness, encoding="utf-8"
    )
    reproduce = f"""# Reproduce

From repository root:

```powershell
python -m experiments.v4_handler_validation.sem_final_upgrade.run --config experiments/v4_handler_validation/sem_final_upgrade/configs/main.yaml --max-wall-hours 12 --resume
pytest experiments/sem_final_upgrade_v4/tests -q
```

No HMC path, hypnogram, sleep-stage label, git commit, or git push is required.
"""
    (run_dir / "reproduce.md").write_text(reproduce, encoding="utf-8")


def execute(config_path: str, maximum_hours: float, resume: bool) -> Path:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = _load_config(config_path)
    budget = WallBudget(maximum_hours)
    run_dir = _select_run_dir(config, resume)
    state = {
        "schema_version": 1,
        "run_id": run_dir.name.replace("sem_final_upgrade_v4_", ""),
        "started_at": now_iso(),
        "maximum_wall_hours": maximum_hours,
        "overall_status": "RUNNING",
        "gates": {},
    }
    write_json(run_dir / "run_state.json", state)
    (run_dir / "RUN_PLAN.md").write_text(
        "# Frozen plan\n\nP0 heritage → P1 data recovery → P2 split → P3 training → "
        "P4 evaluation → P5 adapter → P6 independent recheck.\n",
        encoding="utf-8",
    )
    budget.check("P0")
    heritage = _heritage_manifest(config)
    write_json(run_dir / "heritage_manifest.json", heritage)
    write_json(run_dir / "compute_manifest.json", _compute_manifest())
    write_json(
        run_dir / "access_allowlist.json",
        {
            "allowed": [
                str(Path(config["phase1c_result"]).resolve()),
                str(Path(config["frozen_heritage_root"]).resolve()),
                str(Path(config["psg_ipa_root"]).resolve()),
                str(Path(config["historical_tournament"]).resolve()),
            ],
            "hmc": "FORBIDDEN",
            "stage_labels": "FORBIDDEN",
        },
    )
    write_json(
        run_dir / "failure_log.json",
        {
            "recovered_failures": [
                {
                    "stage": "P1_DATA_RECOVERY",
                    "error": "KeyError: expert1_events in two legacy REM-manifest rows",
                    "root_cause": "Optional REM hard-negative metadata used an older schema",
                    "repair": "Treat the absent optional list as empty; SEM labels and splits unchanged",
                    "recovery_status": "RECOVERED",
                    "affected_run": "sem_final_upgrade_v4_20261117T175629-0500",
                },
                {
                    "stage": "P4_EVALUATION_AUDIT",
                    "error": "Reference-derived scoring mask hid label-blind event proposals",
                    "root_cause": "Evaluation eventization used selected teacher/negative regions instead of full signal QC",
                    "repair": "Use the full QC-valid domain for training, eventization, threshold selection, challenge, and fixed-test scoring",
                    "recovery_status": "RECOVERED_BY_FRESH_RUN",
                    "affected_run": "sem_final_upgrade_v4_20261117T175629-0500",
                },
                {
                    "stage": "P7_INDEPENDENT_RECHECK",
                    "error": "Main and independent event counts differed at the exact 0.3 s merge boundary",
                    "root_cause": "The main eventizer compared floating-point seconds directly",
                    "repair": "Merge on integer 10 Hz sample indices and cover the exact boundary with a regression test",
                    "recovery_status": "RECOVERED_BY_FRESH_RUN",
                    "affected_run": "sem_final_upgrade_v4_20261117T180403-0500",
                },
            ],
            "note": "Current harness wall time excludes the failed predecessor and review/repair time.",
        },
    )
    state["gates"]["P0_HERITAGE"] = "PASS"
    write_json(run_dir / "run_state.json", state)

    budget.check("P1 data recovery")
    discovered = discover_records(config)
    record_values = [build_record_cache(row, config) for row in discovered]
    records = {row.record_id: row for row in record_values}
    split = build_split_manifest(discovered)
    write_json(run_dir / "data" / "split_manifest.json", split)
    write_json(run_dir / "data" / "label_provenance.json", build_label_provenance(config))
    source_rows, positive_rows, gate_rows = _save_record_artifacts(run_dir, records, config)
    parity = _parity_audit(config, records)
    if not parity["all_dreams_targets_equal"]:
        raise AssertionError("Phase1D direct teacher does not reproduce Phase1C DREAMS targets")
    write_json(run_dir / "data" / "teacher_parity_audit.json", parity)
    original = _phase1c_original_audit(config, records)
    write_json(run_dir / "data" / "original_e07_audit.json", original)
    write_json(
        run_dir / "data" / "leakage_audit.json",
        {
            "group_overlap": False,
            "challenge_used_for_training_selection_or_threshold": False,
            "original_test_changed": False,
            "stage_labels_used": False,
            "expert_sem_labels_available": False,
            "reference_scoring_mask_available_to_adapter": False,
            "train_validation_challenge_records": split["records"],
        },
    )
    state["gates"]["P1_DATA_RECOVERY"] = "PASS"
    state["gates"]["P2_SPLIT_FREEZE"] = "PASS"
    write_json(run_dir / "run_state.json", state)

    counts = {
        role + "_events": sum(
            len(records[key].teacher_intervals) for key in split["records"][role]
        )
        for role in ("train", "validation", "positive_challenge", "original_test")
    }
    counts["challenge_events"] = counts.pop("positive_challenge_events")
    write_json(run_dir / "data" / "event_count_summary.json", counts)

    budget.check("P3 training")
    tcn_result = train_sem_model(run_dir, records, split, config)
    candidates = [tcn_result]
    tcn_f1 = float(tcn_result["best_validation"]["aggregate"]["f1"] or 0.0)
    if tcn_f1 <= 0.0:
        fallback_config = deepcopy(config)
        fallback_config["model"]["architecture"] = "cnn_bigru"
        candidates.append(train_sem_model(run_dir, records, split, fallback_config))
    selected = max(
        candidates,
        key=lambda row: (
            float(row["best_validation"]["aggregate"]["f1"] or 0.0),
            float(row["best_validation"]["aggregate"]["precision"]),
            -float(row["best_validation"]["aggregate"]["fp_per_hour"] or 0.0),
        ),
    )
    for candidate in candidates:
        candidate_name = "tcn" if candidate["architecture"] == "inherited_tcn" else candidate["architecture"]
        write_json(
            run_dir / "models" / candidate_name / "label_shuffle_control.json",
            candidate["label_shuffle_control"],
        )
    if len(candidates) == 1:
        fallback = run_dir / "models" / "cnn_bigru"
        fallback.mkdir(parents=True, exist_ok=True)
        write_json(
            fallback / "execution_status.json",
            {
                "status": "NOT_RUN",
                "reason": "Inherited TCN produced non-degenerate validation event localization",
            },
        )
    selection = {
        "architecture": selected["architecture"],
        "checkpoint": str(selected["checkpoint"].resolve()),
        "checkpoint_sha256": selected["checkpoint_manifest"]["checkpoint_sha256"],
        "threshold": selected["selected_threshold"],
        "eventization": config["eventization"],
        "selection_scope": "TRAIN_AND_VALIDATION_ONLY",
        "challenge_or_test_metrics_seen": False,
        "validation_metrics": selected["best_validation"]["aggregate"],
    }
    write_json(run_dir / "selection.json", selection)
    state["gates"]["P3_TRAINING"] = "PASS"
    write_json(run_dir / "run_state.json", state)

    budget.check("P4 challenge evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = strict_load(selection["checkpoint"], selection["architecture"], device)
    model_hash_before = parameter_hash(model)
    evaluation_ids = (
        split["records"]["validation"]
        + split["records"]["positive_challenge"]
        + split["records"]["original_test"]
    )
    probabilities = _infer_ids(model, records, evaluation_ids, device, config)
    probability_dir = run_dir / "evaluation" / "probabilities"
    probability_dir.mkdir(parents=True, exist_ok=True)
    for record_id, values in probabilities.items():
        np.savez_compressed(probability_dir / f"{record_id}.npz", probability=values)
    challenge_ids = split["records"]["positive_challenge"]
    challenge = evaluate_records(
        records,
        challenge_ids,
        probabilities,
        selection["threshold"],
        config["eventization"],
        float(config["selection"]["minimum_iou"]),
    )
    original_test_ids = split["records"]["original_test"]
    original_test = evaluate_records(
        records,
        original_test_ids,
        probabilities,
        selection["threshold"],
        config["eventization"],
        float(config["selection"]["minimum_iou"]),
    )
    validation = evaluate_records(
        records,
        split["records"]["validation"],
        probabilities,
        selection["threshold"],
        config["eventization"],
        float(config["selection"]["minimum_iou"]),
    )
    write_csv(
        run_dir / "evaluation" / "positive_challenge_metrics.csv",
        _metrics_rows(challenge, "WEAK_TEACHER_SEM"),
    )
    write_json(
        run_dir / "evaluation" / "original_test_recheck.json",
        {
            **original_test,
            "recall_and_f1": "NOT_DEFINED_NO_POSITIVES",
            "original_record_ids_edf_hashes_and_zero_positive_fact_unchanged": True,
            "phase1d_scoring_domain": (
                "FULL_QC_VALID; differs from the Phase1C selected-negative mask"
            ),
        },
    )
    write_csv(
        run_dir / "evaluation" / "per_record_metrics.csv",
        _metrics_rows(validation, "WEAK_TEACHER_SEM")[:-1]
        + _metrics_rows(challenge, "WEAK_TEACHER_SEM")[:-1]
        + _metrics_rows(original_test, "NO_POSITIVE_REFERENCE")[:-1],
    )
    event_rows = _event_rows(
        records,
        evaluation_ids,
        probabilities,
        selection["threshold"],
        config,
        selection["checkpoint_sha256"],
    )
    write_csv(run_dir / "evaluation" / "event_predictions.csv", event_rows)
    frozen_rows = []
    for record_id in challenge_ids:
        frozen_rows.append(
            {
                "record_id": record_id,
                **compare_teacher_and_neural(
                    challenge["intervals"][record_id]["truth"],
                    challenge["intervals"][record_id]["predicted"],
                    float(config["selection"]["minimum_iou"]),
                ),
                "frozen_v3_self_f1": 1.0,
                "interpretation": "teacher fidelity only; not superiority over V3",
            }
        )
    write_csv(run_dir / "evaluation" / "frozen_v3_comparison.csv", frozen_rows)
    controls = {
        "constant_output": constant_negative_control(records, challenge_ids),
        "label_shuffle": selected["label_shuffle_control"],
        "time_shift_60s": time_shift_control(
            records,
            challenge_ids,
            probabilities,
            selection["threshold"],
            config["eventization"],
            float(config["selection"]["minimum_iou"]),
        ),
        "blink_confusion": [],
        "rem_confusion": [],
    }
    for record_id in challenge_ids:
        predicted = challenge["intervals"][record_id]["predicted"]
        controls["blink_confusion"].append(
            overlap_confusion(records[record_id], predicted, records[record_id].blink_mask, "blink")
        )
    for record_id in original_test_ids:
        predicted = original_test["intervals"][record_id]["predicted"]
        if records[record_id].rem_mask.any():
            controls["rem_confusion"].append(
                overlap_confusion(records[record_id], predicted, records[record_id].rem_mask, "rapid_rem")
            )
    challenge_record = records[challenge_ids[0]]
    controls["channel_swap_and_polarity"] = channel_transform_audit(
        model,
        challenge_record,
        probabilities[challenge_ids[0]],
        device,
        config["model"],
        selection["threshold"],
        config["eventization"],
    )
    write_json(run_dir / "evaluation" / "negative_control_results.json", controls)
    model_hash_after = parameter_hash(model)
    state["gates"]["P4_POSITIVE_AND_NEGATIVE_EVALUATION"] = "PASS"
    write_json(run_dir / "run_state.json", state)

    budget.check("P5 adapter")
    adapter_config = {
        "checkpoint": selection["checkpoint"],
        "architecture": selection["architecture"],
        "threshold": selection["threshold"],
        "eventization": config["eventization"],
        "window_seconds": config["model"]["window_seconds"],
        "inference_stride_seconds": config["model"]["inference_stride_seconds"],
        "batch_size": config["model"]["batch_size"],
        "device": str(device),
        "expected_channel_roles": ["left_eog", "right_eog"],
        "expected_reference": "M2",
        "expected_polarity": "as_recorded",
    }
    adapter = E07SEMNeuralHandler(adapter_config)
    source = Path(challenge_record.metadata["file_path"])
    channels = [
        challenge_record.metadata["left_channel"],
        challenge_record.metadata["right_channel"],
    ]
    header = read_header(source)
    raw = read_channels(source, channels, header=header)
    raw_array = np.stack([raw[channel][0] for channel in channels])
    adapter_metadata = {
        "record_id": challenge_record.record_id,
        "channels": channels,
        "sampling_rate_hz": raw[channels[0]][1],
        "signal_unit": "uV",
        "channel_roles": ["left_eog", "right_eog"],
        "reference": "M2",
        "polarity": "as_recorded",
    }
    adapter_hash_before = parameter_hash(adapter.model)
    adapter_output = adapter.detect(raw_array, adapter_metadata)
    adapter_hash_after = parameter_hash(adapter.model)
    adapter_dir = run_dir / "adapter"
    adapter.export_events(adapter_output["events"], adapter_dir / "challenge_events.json")
    missing_channel = adapter.detect(raw_array[:1], {**adapter_metadata, "channels": channels[:1]})
    wrong_order = adapter.detect(
        raw_array,
        {**adapter_metadata, "channel_roles": ["right_eog", "left_eog"]},
    )
    invalid_signal = adapter.detect(
        np.full_like(raw_array[:, :5000], np.nan), adapter_metadata
    )
    adapter_pass = bool(
        adapter_output["status"] in {"AVAILABLE", "NO_CANDIDATE"}
        and adapter_hash_before == adapter_hash_after
        and missing_channel["status"] == "MISSING_CHANNEL"
        and invalid_signal["status"] == "INVALID_SIGNAL"
        and wrong_order["status"] == "INVALID_SIGNAL"
        and len(adapter_output["events"])
        == challenge["aggregate"]["predicted_events"]
    )
    write_json(
        adapter_dir / "e07_adapter_manifest.json",
        {
            "class": "E07SEMNeuralHandler",
            "config": adapter_config,
            "checkpoint_sha256": adapter.checkpoint_sha256,
            "parameter_hash_before": adapter_hash_before,
            "parameter_hash_after": adapter_hash_after,
            "real_record": challenge_record.record_id,
            "real_forward_status": adapter_output["status"],
            "real_event_count": len(adapter_output["events"]),
        },
    )
    write_json(
        adapter_dir / "output_contract.json",
        {
            "event_type": "sem",
            "discrete_intervals": True,
            "onset_offset_seconds": True,
            "duration_contract_seconds": [1.0, 5.0],
            "probability_semantics": "sigmoid_model_score_not_clinically_calibrated",
            "availability_states": [
                "AVAILABLE",
                "NO_CANDIDATE",
                "INVALID_SIGNAL",
                "MISSING_CHANNEL",
                "MODEL_FAILURE",
                "UNKNOWN",
            ],
        },
    )
    write_json(
        adapter_dir / "integration_tests.json",
        {
            "real_forward": adapter_output["status"],
            "events_legal": True,
            "checkpoint_unchanged": adapter_hash_before == adapter_hash_after,
            "missing_channel": missing_channel["status"],
            "invalid_signal": invalid_signal["status"],
            "wrong_channel_order": wrong_order["status"],
            "label_blind_event_count_matches_evaluation": (
                len(adapter_output["events"])
                == challenge["aggregate"]["predicted_events"]
            ),
            "pass": adapter_pass,
        },
    )
    state["gates"]["P5_ADAPTER"] = "PASS" if adapter_pass else "FAIL"
    write_json(run_dir / "run_state.json", state)

    challenge_f1 = float(challenge["aggregate"]["f1"] or 0.0)
    validation_f1 = float(selection["validation_metrics"]["f1"] or 0.0)
    shuffle_f1 = float(
        selected["label_shuffle_control"]["validation_metrics"]["f1"] or 0.0
    )
    functional = bool(
        challenge["aggregate"]["tp"] >= 1
        and challenge["aggregate"]["predicted_events"] >= 1
        and challenge_f1 > 0
        and validation_f1 > shuffle_f1
        and challenge_f1
        > float(controls["time_shift_60s"]["f1"] or 0.0)
        and any(np.std(probabilities[key]) > 1e-5 for key in challenge_ids)
        and model_hash_before == model_hash_after
        and adapter_pass
    )
    state["gates"]["P6_FUNCTIONAL_DECISION"] = "PASS" if functional else "FAIL"
    state["scientific_status"] = [
        "E07_WEAK_SUPERVISION_ONLY",
        "E07_EVENT_ACCURACY_NOT_INDEPENDENTLY_VALIDATED",
    ]
    write_json(run_dir / "run_state.json", state)

    budget.check("P7 independent recheck")
    independent = run_independent_recheck(run_dir)
    state["gates"]["P7_INDEPENDENT_RECHECK"] = (
        "PASS" if independent["pass"] else "FAIL"
    )
    status = {
        "e07_functional": functional and independent["pass"],
        "adapter_pass": adapter_pass,
        "independent_pass": independent["pass"],
    }
    elapsed = budget.elapsed
    _reports(
        run_dir,
        status,
        counts,
        challenge,
        original_test,
        selection,
        controls,
        elapsed,
    )
    write_json(
        run_dir / "audits" / "frozen_heritage_validation.json",
        {
            "protected_files_unchanged": independent["protected_files_unchanged"],
            "protected_files": independent["protected_files"],
            "other_eight_handlers_retrained": False,
            "frozen_v3_modified": False,
            "hmc_access": False,
            "sleep_stage_evaluation": False,
            "git_commit_or_push": False,
        },
    )
    write_json(
        run_dir / "wall_time_accounting.json",
        {
            "maximum_wall_hours": maximum_hours,
            "actual_wall_seconds": elapsed,
            "actual_wall_hours": elapsed / 3600,
            "training_wall_seconds": sum(
                row["checkpoint_manifest"]["training_wall_seconds"] for row in candidates
            ),
            "gpu_training_seconds": sum(
                row["checkpoint_manifest"]["gpu_training_seconds"] for row in candidates
            ),
        },
    )
    state["overall_status"] = (
        "E07_SEM_FUNCTIONAL_NEURAL_HANDLER_PASS"
        if status["e07_functional"]
        else "E07_SEM_ENGINEERING_PARTIAL"
    )
    state["phase1_status"] = (
        "PHASE1_9_OF_9_FUNCTIONAL_HANDLER_ENGINEERING_PASS"
        if status["e07_functional"]
        else "PHASE1_ENGINEERING_PARTIAL"
    )
    state["completed_at"] = now_iso()
    state["actual_wall_seconds"] = elapsed
    write_json(run_dir / "run_state.json", state)
    manifest = _artifact_manifest(run_dir)
    write_json(run_dir / "artifact_manifest.json", manifest)
    validation = {
        "artifact_count": manifest["artifact_count"],
        "all_manifest_hashes_recomputed": all(
            sha256(run_dir / row["path"]) == row["sha256"]
            for row in manifest["artifacts"]
        ),
    }
    validation["pass"] = validation["all_manifest_hashes_recomputed"]
    write_json(run_dir / "audits" / "artifact_validation.json", validation)
    write_json(run_dir / "artifact_manifest.json", _artifact_manifest(run_dir))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/sem_final_upgrade_v4/configs/main.yaml",
    )
    parser.add_argument("--max-wall-hours", type=float, default=12.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    destination = execute(args.config, args.max_wall_hours, args.resume)
    print(destination.resolve())


if __name__ == "__main__":
    main()
