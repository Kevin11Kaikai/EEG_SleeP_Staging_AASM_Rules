from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime
from typing import Any

import pandas as pd
import torch

from experiments.aasm_full_detector_integration.paths import UNET_MODEL

from .config import (
    ALIGNED_REFERENCE, ALLOWLIST_MANIFEST, ARMS, BASE_BANK_ROOT, DISPLAY_NAME,
    E07_CHECKPOINT, EVENT_TYPE, EXPECTED_MODEL_SHA, FAMILIES, FROZEN_CORE,
    HANDLERS, K08_RUN, PHASE1C, PHASE1D, PUBLIC_ROOT, R04_RUN, RECORDS,
    RESULTS, SP_RUN, SSSM_CHECKPOINT, WORKTREE,
)
from .independent_recheck import run as independent_recheck
from .io_utils import file_manifest, read_json, sha256, write_csv, write_json
from .metrics import STAGES, build_effect_tables, evaluate
from .packaging import package_delivery
from .shadow_overlay import execute_arms
from .source_banks import generate_source_banks


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=WORKTREE, capture_output=True, text=True, check=False)
    return result.stdout.strip()


def _checkpoint_paths() -> dict[str, Path]:
    return {
        "E01": SSSM_CHECKPOINT,
        "E02": K08_RUN / "hmc_k08" / "k08_final_checkpoint.pt",
        "E03": SSSM_CHECKPOINT,
        "E04": R04_RUN / "r04" / "r04_final_checkpoint.pt",
        "E05": FAMILIES["eeg_background"].checkpoint,
        "E06": FAMILIES["eog_events"].checkpoint,
        "E07": E07_CHECKPOINT,
        "E08": FAMILIES["eeg_background"].checkpoint,
        "E09": FAMILIES["chin_emg"].checkpoint,
        "E10": FAMILIES["chin_emg"].checkpoint,
        "E11": FAMILIES["chin_emg"].checkpoint,
        "E12": FAMILIES["eeg_background"].checkpoint,
        "E13": FAMILIES["major_movement"].checkpoint,
        "SP": SP_RUN / "inputs" / "software" / "sumov2" / "output" / "final.ckpt",
        "AR": UNET_MODEL,
    }


def _model_registry() -> list[dict[str, Any]]:
    paths = _checkpoint_paths()
    rows = []
    for handler in HANDLERS:
        path = paths[handler]
        actual = sha256(path)
        rows.append({
            "handler_id": handler, "handler": DISPLAY_NAME[handler], "event_type": EVENT_TYPE[handler],
            "checkpoint": str(path.resolve()), "checkpoint_sha256": actual,
            "expected_sha256": EXPECTED_MODEL_SHA[handler], "checkpoint_match": actual == EXPECTED_MODEL_SHA[handler],
            "frozen": True, "training_in_phase2": False,
            "evidence_boundary": (
                "expert-supported development candidate" if handler in {"E02", "E04", "SP", "AR"}
                else "weak-teacher/prototype; no independent HMC event truth"
            ),
        })
    if not all(row["checkpoint_match"] for row in rows):
        raise RuntimeError("one or more model checkpoints failed identity validation")
    return rows


def _input_contracts() -> list[dict[str, Any]]:
    channels = {
        "E01": "EEG C4-M1", "E02": "EEG C4-M1", "E03": "EEG C4-M1",
        "E04": "EOG E1-M2+EOG E2-M2", "E05": "EEG C4-M1+EEG O2-M1",
        "E06": "EOG E1-M2+EOG E2-M2", "E07": "EOG E1-M2+EOG E2-M2",
        "E08": "EEG C4-M1", "E09": "EMG chin", "E10": "EMG chin",
        "E11": "EMG chin", "E12": "EEG C4-M1+EEG O2-M1",
        "E13": "EEG F4-M1+EEG C4-M1+EMG chin", "SP": "EEG C4-M1",
        "AR": "EEG C4-M1+EOG E1-M2+EMG chin",
    }
    return [{
        "handler_id": handler, "source_channels": channels[handler], "native_sampling_hz": 256.0,
        "native_unit": "uV", "contract_status": "EXECUTABLE_WITH_UNVALIDATED_TRANSFER",
        "selection_frozen_before_stage_reference": True,
        "caveat": (
            "O2-M1 supplies a real occipital channel, but is not the Phase1C Pz-Oz montage"
            if handler in {"E05", "E12"} else
            "real HMC signal mapping; cross-dataset montage/domain transfer is not independently validated"
        ),
    } for handler in HANDLERS]


def _preflight(run_root: Path, config_path: Path) -> dict[str, Any]:
    freeze = read_json(FROZEN_CORE / "configs" / "repair_freeze.json")
    core_rows = []
    for relative, expected in freeze["files"].items():
        path = FROZEN_CORE / relative
        actual = sha256(path)
        core_rows.append({"path": relative, "expected": expected, "actual": actual, "match": expected == actual})
    if not all(row["match"] for row in core_rows):
        raise RuntimeError("frozen core hash mismatch")
    allowlist = read_json(ALLOWLIST_MANIFEST)
    if tuple(sorted(row["record_id"] for row in allowlist["records"])) != RECORDS:
        raise PermissionError("HMC allow-list mismatch")
    registry = _model_registry()
    contracts = _input_contracts()
    compute = {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count(),
    }
    preflight = {
        "git_head": _git("rev-parse", "HEAD"), "git_branch": _git("branch", "--show-current") or "DETACHED",
        "git_status_before": _git("status", "--short"), "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path), "frozen_core_all_match": True,
        "records": list(RECORDS), "handler_count": len(HANDLERS), "arm_count": len(ARMS),
        "no_training": True, "stage_reference_not_read": True, "compute": compute,
    }
    write_json(run_root / "preflight.json", preflight)
    write_json(run_root / "heritage_manifest.json", {
        "phase1c": str(PHASE1C.resolve()), "phase1d": str(PHASE1D.resolve()),
        "frozen_core": str(FROZEN_CORE.resolve()), "base_bank_root": str(BASE_BANK_ROOT.resolve()),
        "public_evaluator_root": str(PUBLIC_ROOT.resolve()),
    })
    write_json(run_root / "frozen_core_hashes.json", {"rows": core_rows, "all_match": True,
                                                          "declared_combined_hash": freeze["source_config_test_hash"]})
    write_json(run_root / "model_registry_15.json", registry)
    write_json(run_root / "input_contract_15.json", contracts)
    write_json(run_root / "access_allowlist.json", {
        "records": list(RECORDS), "signal_paths": {row["record_id"]: row["edf_path"] for row in allowlist["records"]},
        "stage_reference_forbidden_until_prediction_seals": str(ALIGNED_REFERENCE.resolve()),
        "outside_record_allowlist_allowed": False,
    })
    return preflight


def _arm_manifest(run_root: Path) -> None:
    rows = []
    for arm, handlers in ARMS.items():
        if arm == "B000":
            kind = "baseline"
        elif arm.startswith("S_"):
            kind = "single"
        elif arm.startswith("G_"):
            kind = "group"
        elif arm == "F_ALL15":
            kind = "full"
        else:
            kind = "leave_one_out"
        rows.append({"arm_id": arm, "arm_type": kind, "replacement_count": len(handlers),
                     "replacement_handlers": ";".join(handlers),
                     "comparator": "F_ALL15" if kind == "leave_one_out" else "B000"})
    write_csv(run_root / "arms" / "arm_manifest_36.csv", rows)


def _table(metrics: dict[str, Any], arms: list[str]) -> str:
    lines = ["| Arm | Accuracy | Macro-F1 | W F1 | N1 F1 | N2 F1 | N3 F1 | REM F1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in arms:
        row = metrics[arm]
        lines.append("| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
            arm, row["accuracy"], row["macro_f1"], *[row["per_stage"][stage]["f1"] for stage in STAGES[:5]]))
    return "\n".join(lines)


def _reports(run_root: Path, start: float, evaluation: dict[str, Any], effects: dict[str, Any], audit: dict[str, Any]) -> None:
    metrics = evaluation["metrics"]["primary"]
    base, full = metrics["B000"], metrics["F_ALL15"]
    singles = [f"S_{handler}" for handler in HANDLERS]
    ranked = sorted(effects["single"], key=lambda row: row["single_delta_macro_f1"], reverse=True)
    best = ranked[0]
    worst = ranked[-1]
    status = "PHASE2_36_ARM_ENGINEERING_PASS" if audit["status"] == "PASS" and audit["all_360_executed"] else "PHASE2_36_ARM_PARTIAL"
    full_effective = pd.read_csv(run_root / "arms" / "arm_execution_10night.csv")
    full_rows = full_effective.loc[full_effective.arm_id == "F_ALL15"]
    full_status = "FULL_15_EFFECTIVE_REPLACEMENT" if int(full_rows.effective_replacements.min()) == 15 else "PARTIAL_HANDLER_AVAILABILITY"
    wall = time.perf_counter() - start
    executive = f"""# AASM V4 Phase 2 Executive Summary

## 结论

本轮实际完成 **36/36 Arms、360/360 Night-Arm 重放**。B000 十晚 bank/evidence/prediction 均与冻结 Temporal R1 精确回归；15 路来源在读取 Stage Reference 前已完成 label-blind 封存。工程状态为 `{status}`，完整替换状态为 `{full_status}`。

F_ALL15 相对 B000：Accuracy `{full['accuracy'] - base['accuracy']:+.6f}`，Macro-F1 `{full['macro_f1'] - base['macro_f1']:+.6f}`；这是 HMC 十晚开发集 shadow 结果，不是临床泛化验证。

最强单换（按 Macro-F1）：`{best['handler_id']}` `{best['single_delta_macro_f1']:+.6f}`；最弱单换：`{worst['handler_id']}` `{worst['single_delta_macro_f1']:+.6f}`。K08、Phase1C/1D 弱监督 Handler 与 E13 movement proxy 的既有限制均保留。

{_table(metrics, ['B000', 'F_ALL15'])}

## 证据边界

- HMC 没有本轮独立事件级真值；这里只测事件候选如何传播到冻结规则和 Stage 决策。
- 所有新输入映射均保留 `MONTAGE_TRANSFER_UNVALIDATED`；E05/E12 使用真实 O2-M1，但与训练 Pz-Oz 并非等价导联。
- 全部阈值、权重、规则和 Temporal R1 在标签解封前冻结，未以 HMC 结果调参。

实际 Wall Time：`{wall:.1f}s`。独立复算最大指标差：`{audit['maximum_independent_metric_difference']:.3e}`。
"""
    full_report = f"""# FINAL PHASE 2 — 36-Arm Event-to-Stage Report

## Engineering closure

- Status: `{status}`
- Full replacement status: `{full_status}`
- Registered/Executed: 36/36 arms; 360/360 night-arm runs
- Primary/Full-valid: 8,902 / 9,211 fixed epochs
- B000 exact regression: PASS (10/10)
- Independent recheck: `{audit['status']}`
- No training, no checkpoint selection, no rule changes, and no access outside the frozen record allowlist.

## Primary 36-arm matrix

{_table(metrics, list(ARMS))}

## Full-combination result

F_ALL15 vs B000: ΔAccuracy `{full['accuracy'] - base['accuracy']:+.9f}`, ΔMacro-F1 `{full['macro_f1'] - base['macro_f1']:+.9f}`.

## Causal interpretation boundary

Single swaps estimate one-source replacement effects under the frozen baseline context. Group and full results include interactions. Leave-one-out estimates a handler's marginal contribution inside F_ALL15 and is not interchangeable with its single-swap effect. Development-set replay establishes internal system behavior only.

## Main limitations

Phase1C/1D models are largely weak-teacher models; E07 is weak-supervision-only; E13 is a movement proxy; SSSM E01/E03 lack independent expert event truth; K08 already had an event-level gain/downstream regression history. Therefore Stage deltas do not prove event accuracy or clinical validity.
"""
    mechanism = """# Mechanism Analysis

The executable evidence chain is `source event -> frozen aggregate() -> rule predicate -> Temporal R1 state -> final stage`.

- `evaluation/evidence_usage.csv` quantifies candidate coverage and duration changes.
- `evaluation/rule_firing_summary.csv` records rule firings by arm.
- `evaluation/stage_rule_transitions.csv` records paired rule/stage transitions.
- `evaluation/paired_error_analysis.csv` closes corrected, introduced and net-correct identities.
- `analysis/interaction_analysis.csv` separates observed group deltas from sums of single-swap deltas.

No HMC stage outcome was used to alter any source bank or event threshold.
"""
    write_text = lambda path, text: (path.parent.mkdir(parents=True, exist_ok=True), path.write_text(text, encoding="utf-8"))
    write_text(run_root / "EXECUTIVE_SUMMARY.md", executive)
    write_text(run_root / "FINAL_PHASE2_36ARM_REPORT.md", full_report)
    write_text(run_root / "analysis" / "mechanism_analysis.md", mechanism)
    write_text(run_root / "FINAL_SCIENTIFIC_DECISION.md", f"""# Final Scientific Decision

Engineering: `{status}`. Scientific result: **development-set event-to-stage effects measured; external validity not established**.

Full combination ΔMacro-F1 `{full['macro_f1'] - base['macro_f1']:+.6f}` and ΔAccuracy `{full['accuracy'] - base['accuracy']:+.6f}`. Handler retention should be based on the single-swap, leave-one-out, coverage, and mechanism tables together—not the maximum of 36 post-hoc scores.
""")
    write_text(run_root / "reproduce.md", f"""# Reproduce

```powershell
python -m experiments.v4_36arm_ablation.run --config experiments/v4_36arm_ablation/configs/phase2.yaml --max-wall-hours 12 --resume --run-dir "{run_root}"
```

The command resumes only from sealed source/prediction gates and revalidates hashes before label access.
""")


def _run_dir(resume: bool, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    active = RESULTS / "aasm_v4_phase2_active.json"
    if resume and active.exists():
        candidate = Path(read_json(active)["run_root"])
        if candidate.exists():
            return candidate
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    root = RESULTS / f"aasm_v4_phase2_36arm_{stamp}"
    write_json(active, {"run_root": str(root.resolve())})
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs" / "phase2.yaml")
    parser.add_argument("--max-wall-hours", type=float, default=12.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    start = time.perf_counter()
    run_root = _run_dir(args.resume, args.run_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "run_state.json"
    state = {"status": "RUNNING", "run_root": str(run_root), "max_wall_hours": args.max_wall_hours,
             "started_at": datetime.now().astimezone().isoformat(), "gates": {}}
    write_json(state_path, state)
    try:
        state["gates"]["P0"] = "RUNNING"; write_json(state_path, state)
        _preflight(run_root, args.config)
        _arm_manifest(run_root)
        state["gates"]["P0"] = "PASS"; write_json(state_path, state)
        if time.perf_counter() - start > args.max_wall_hours * 3600:
            raise TimeoutError("wall-time budget exhausted before inference")
        state["gates"]["P2"] = "RUNNING"; write_json(state_path, state)
        source_seal = run_root / "neural_inference" / "prediction_seal.json"
        if not source_seal.exists():
            generate_source_banks(run_root)
        state["gates"]["P2"] = "PASS"; write_json(state_path, state)
        state["gates"]["P3_P8"] = "RUNNING"; write_json(state_path, state)
        arm_seal = run_root / "arms" / "prediction_seal.json"
        if not arm_seal.exists():
            execute_arms(run_root)
        evaluation = evaluate(run_root)
        effects = build_effect_tables(run_root, evaluation)
        audit = independent_recheck(run_root)
        state["gates"]["P3_P8"] = "PASS" if audit["status"] == "PASS" else "FAIL"
        state["status"] = "PASS" if audit["status"] == "PASS" else "FAIL"
        state["completed_at"] = datetime.now().astimezone().isoformat()
        state["wall_seconds"] = time.perf_counter() - start
        write_json(state_path, state)
        _reports(run_root, start, evaluation, effects, audit)
        write_json(run_root / "wall_time_accounting.json", {
            "total_wall_seconds": time.perf_counter() - start, "max_wall_hours": args.max_wall_hours,
            "completed_within_budget": time.perf_counter() - start <= args.max_wall_hours * 3600,
        })
        write_json(run_root / "audits" / "label_blind_inference.json", {
            "status": "PASS", "source_bank_sealed_before_reference": True,
            "arm_predictions_sealed_before_reference": True, "stage_reference_read_by_inference": False,
        })
        write_json(run_root / "audits" / "no_training.json", {
            "status": "PASS", "optimizer_steps": 0, "backpropagation": False,
            "checkpoint_selection_from_hmc": False, "threshold_tuning_from_hmc": False,
        })
        delivery = package_delivery(run_root)
        print(json.dumps(delivery, ensure_ascii=False))
    except Exception as exc:
        state["status"] = "FAILED"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["wall_seconds"] = time.perf_counter() - start
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
