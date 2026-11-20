from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .analysis import run_analysis
from .core import RECORDS, load_reference, read_config, sha256, write_json
from .figures import generate_figures
from .independent_recheck import run_independent_recheck
from .reports import write_reports


TZ = ZoneInfo("America/New_York")


def now_iso() -> str:
    return datetime.now(TZ).isoformat()


def _git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as error:  # provenance should not block analysis
        return f"UNAVAILABLE: {error}"


def _select_output(config: dict[str, Any], resume: bool) -> Path:
    base = Path(config["output_base"])
    if resume:
        candidates = sorted(base.glob("phase2_mechanistic_attribution_*"), reverse=True)
        for candidate in candidates:
            state_path = candidate / "run_state.json"
            if state_path.is_file():
                return candidate
    run_id = datetime.now(TZ).strftime("%Y%m%dT%H%M%S%z")
    return base / f"phase2_mechanistic_attribution_{run_id}"


def _source_registry(source_root: Path) -> dict[str, Any]:
    critical = [
        "artifact_manifest.json", "arms/arm_manifest_36.csv", "arms/arm_execution_10night.csv",
        "evaluation/all_36_arm_metrics.csv", "evaluation/per_night_metrics.csv",
        "analysis/single_handler_effects.csv", "analysis/leave_one_out_effects.csv",
        "analysis/interaction_analysis.csv",
    ]
    return {
        "source_root": str(source_root),
        "critical_files": [
            {"path": item, "size_bytes": (source_root / item).stat().st_size, "sha256": sha256(source_root / item)}
            for item in critical
        ],
        "raw_artifact_counts": {
            kind: sum(1 for arm in pd.read_csv(source_root / "arms" / "arm_manifest_36.csv").arm_id for record in RECORDS if (source_root / kind / arm / f"{record}.json.gz").is_file())
            for kind in ("event_banks", "evidence", "predictions", "traces")
        },
        "record_allowlist": list(RECORDS),
    }


def _artifact_manifest(output_root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(p for p in output_root.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"):
        files.append({
            "path": path.relative_to(output_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = {"root": str(output_root), "generated_at": now_iso(), "files": files}
    write_json(output_root / "artifact_manifest.json", manifest)
    return manifest


def _validate_artifacts(output_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    failures = []
    for entry in manifest["files"]:
        path = output_root / entry["path"]
        actual = sha256(path) if path.is_file() else "MISSING"
        if actual != entry["sha256"]:
            failures.append({"path": entry["path"], "expected": entry["sha256"], "actual": actual})
    return {"status": "PASS" if not failures else "FAIL", "checked_files": len(manifest["files"]), "failures": failures}


def run(config_path: Path, max_wall_hours: float, resume: bool) -> Path:
    config = read_config(config_path)
    output_root = _select_output(config, resume)
    output_root.mkdir(parents=True, exist_ok=True)
    previous_state: dict[str, Any] = {}
    if (output_root / "run_state.json").is_file():
        previous_state = json.loads((output_root / "run_state.json").read_text(encoding="utf-8"))
    started = time.time()
    started_at = now_iso()
    original_started_at = previous_state.get("original_started_at", previous_state.get("started_at", started_at))
    attempt_history = list(previous_state.get("attempt_history", []))
    state: dict[str, Any] = {
        "status": "RUNNING", "started_at": started_at,
        "original_started_at": original_started_at,
        "attempt_history": attempt_history,
        "max_wall_hours": max_wall_hours, "output_root": str(output_root),
        "gates": {},
    }
    write_json(output_root / "run_state.json", state)
    source_root = Path(config["source_root"])
    try:
        state["gates"]["P0_HERITAGE"] = {"status": "RUNNING", "started_at": now_iso()}
        reference = load_reference(config)
        if int(reference.in_primary.sum()) != int(config["expected_primary_epochs"]):
            raise RuntimeError("Primary mask count mismatch")
        if int(reference.in_full_record.sum()) != int(config["expected_full_valid_epochs"]):
            raise RuntimeError("Full-valid mask count mismatch")
        registry = _source_registry(source_root)
        write_json(output_root / "source_artifact_registry.json", registry)
        heritage = {
            "source_phase2": str(source_root),
            "source_artifact_manifest_sha256": sha256(source_root / "artifact_manifest.json"),
            "reference_path": config["reference_path"],
            "reference_sha256": sha256(Path(config["reference_path"])),
            "phase2_code_root": "experiments/v4_36arm_ablation",
            "phase2b_code_root": "experiments/v4_mechanistic_attribution",
            "git_head": _git_value(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"]),
            "git_dirty_status": _git_value(["git", "-C", str(Path(__file__).resolve().parents[2]), "status", "--short"]),
            "python": sys.version, "platform": platform.platform(),
            "records": list(RECORDS),
            "protection": "source Phase 2 artifacts read-only; outputs remain under AASM_V4_RESULTS_ROOT",
        }
        write_json(output_root / "heritage_manifest.json", heritage)
        state["gates"]["P0_HERITAGE"] = {"status": "PASS", "completed_at": now_iso()}
        write_json(output_root / "run_state.json", state)

        state["gates"]["P1_TO_P8_ANALYSIS"] = {"status": "RUNNING", "started_at": now_iso()}
        results = run_analysis(config, output_root, reference)
        if (time.time() - started) / 3600 > max_wall_hours:
            raise TimeoutError("Wall-time budget exceeded before packaging")
        generate_figures(output_root, results)
        write_reports(config, output_root, results, original_started_at)
        state["gates"]["P1_TO_P8_ANALYSIS"] = {"status": "PASS", "completed_at": now_iso()}
        write_json(output_root / "run_state.json", state)

        state["gates"]["P9_INDEPENDENT_RECHECK"] = {"status": "RUNNING", "started_at": now_iso()}
        independent = run_independent_recheck(config, output_root, reference, results["arm_ids"])
        state["gates"]["P9_INDEPENDENT_RECHECK"] = {
            "status": independent["status"], "completed_at": now_iso()
        }
        if independent["status"] != "PASS":
            raise RuntimeError("Independent recheck failed")

        elapsed = time.time() - started
        total_wall = time.time() - datetime.fromisoformat(original_started_at).timestamp()
        attempt_history.append({
            "started_at": started_at, "completed_at": now_iso(),
            "status": "PASS", "elapsed_seconds": elapsed,
        })
        write_json(output_root / "wall_time_accounting.json", {
            "started_at": original_started_at, "completed_at": now_iso(),
            "total_wall_seconds": total_wall, "total_wall_hours": total_wall / 3600,
            "latest_attempt_seconds": elapsed,
            "active_attempt_seconds": sum(float(item.get("elapsed_seconds", 0.0)) for item in attempt_history),
            "attempt_history": attempt_history,
            "max_wall_hours": max_wall_hours, "within_budget": total_wall <= max_wall_hours * 3600,
        })
        state["status"] = "PHASE2_MECHANISTIC_ATTRIBUTION_PASS"
        state["completed_at"] = now_iso()
        state["elapsed_seconds"] = total_wall
        state["attempt_history"] = attempt_history
        write_json(output_root / "run_state.json", state)

        manifest = _artifact_manifest(output_root)
        validation = _validate_artifacts(output_root, manifest)
        write_json(output_root / "audits" / "artifact_integrity.json", validation)
        manifest = _artifact_manifest(output_root)
        final_validation = _validate_artifacts(output_root, manifest)
        write_json(output_root / "audits" / "final_delivery_validation.json", final_validation)
        _artifact_manifest(output_root)
        return output_root
    except Exception as error:
        elapsed = time.time() - started
        attempt_history.append({
            "started_at": started_at, "failed_at": now_iso(),
            "status": "FAILED", "elapsed_seconds": elapsed, "error": repr(error),
        })
        state["status"] = "FAILED"
        state["failed_at"] = now_iso()
        state["error"] = repr(error)
        state["attempt_history"] = attempt_history
        write_json(output_root / "run_state.json", state)
        write_json(output_root / "wall_time_accounting.json", {
            "started_at": original_started_at, "failed_at": now_iso(),
            "latest_attempt_seconds": elapsed, "attempt_history": attempt_history,
            "max_wall_hours": max_wall_hours,
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-wall-hours", type=float, default=12.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = run(args.config, args.max_wall_hours, args.resume)
    print(output)


if __name__ == "__main__":
    main()
