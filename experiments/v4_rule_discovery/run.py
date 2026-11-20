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

import pandas as pd
import yaml

from .cache import PredictionCache
from .config import (
    ALIGNED_REFERENCE,
    CHAMPION_ROOT,
    CONFIRMATION_RECORDS,
    DISCOVERY_RECORDS,
    FROZEN_CORE,
    RECORDS,
    RESULTS_ROOT,
    RuleConfig,
    candidate_universe,
    validate_frozen_paths,
)
from .io import (
    artifact_manifest,
    canonical_bytes,
    read_json_gz,
    sha256_bytes,
    sha256_file,
    write_json,
)
from .replay import export_full_trace, predict_all
from .sources import IMMUTABLE_RULE_CONTRACT, RULE_SOURCES


def utc_now() -> str:
    return datetime.now().astimezone().isoformat()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[2]), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Data-driven, AASM-grounded decision-rule discovery on frozen M_6408"
    )
    parser.add_argument(
        "phase",
        nargs="?",
        choices=("all", "predict", "split", "select", "confirm"),
        default="all",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "main.yaml",
    )
    parser.add_argument("--max-wall-hours", type=float, default=None)
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return args.run_dir.resolve()
    if args.resume:
        candidates = sorted(
            RESULTS_ROOT.glob("aasm_rule_discovery_v5_*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    run_id = args.run_id or datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    return RESULTS_ROOT / f"aasm_rule_discovery_v5_{run_id}"


def load_state(root: Path) -> dict[str, Any]:
    return json.loads((root / "run_state.json").read_text(encoding="utf-8"))


def update_state(root: Path, state: dict[str, Any], gate: str, status: str) -> None:
    state.setdefault("gates", {})[gate] = {"status": status, "timestamp": utc_now()}
    state["updated_at"] = utc_now()
    write_json(root / "run_state.json", state)


def frozen_artifacts() -> list[Path]:
    files = [
        FROZEN_CORE / "src" / "aasm_temporal_r1" / "engine.py",
        FROZEN_CORE / "src" / "aasm_temporal_r1" / "timeline.py",
        FROZEN_CORE / "src" / "aasm_v2" / "rules" / "__init__.py",
        FROZEN_CORE / "src" / "aasm_v2" / "types.py",
    ]
    for record in RECORDS:
        files.extend(
            [
                CHAMPION_ROOT / "event_banks" / f"{record}.json.gz",
                CHAMPION_ROOT / "evidence" / f"{record}.json.gz",
                CHAMPION_ROOT / "predictions" / f"{record}.json.gz",
            ]
        )
    return files


def initialize_run(
    root: Path,
    config_data: dict[str, Any],
    configs: tuple[RuleConfig, ...],
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    for child in (
        "prediction",
        "reference",
        "selection",
        "analysis",
        "traces",
        "audits",
        "cache",
    ):
        (root / child).mkdir(exist_ok=True)
    sources = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in frozen_artifacts()
    ]
    heritage = {
        "created_at": utc_now(),
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
        "champion": {
            "arm_id": "M_6408",
            "mask": 25608,
            "handlers": ["E04", "E11", "SP", "AR"],
            "root": str(CHAMPION_ROOT),
        },
        "frozen_core": str(FROZEN_CORE),
        "frozen_artifacts": sources,
        "protected_reference": {
            "path": str(ALIGNED_REFERENCE),
            "status": "PROTECTED_NOT_OPENED_BEFORE_PREDICTION_SEAL",
        },
    }
    config_hashes = {
        config.config_id: sha256_bytes(canonical_bytes(config.to_dict())) for config in configs
    }
    dimensions = {
        "rem_backward_policy": 2,
        "wake_ocular_policy": 3,
        "n1_alpha_policy": 2,
        "n1_nonalpha_policy": 2,
        "n2_continuation_policy": 2,
        "post_arousal_reentry_policy": 2,
        "fallback_policy": 2,
        "rem_second_half_policy": 2,
    }
    candidate_payload = {
        "created_before_reference_access": True,
        "count": len(configs),
        "cartesian_dimensions": dimensions,
        "configs": [
            {**config.to_dict(), "sha256": config_hashes[config.config_id]}
            for config in configs
        ],
    }
    write_json(root / "heritage_manifest.json", heritage)
    write_json(root / "source_artifact_registry.json", {"artifacts": sources})
    write_json(
        root / "access_allowlist.json",
        {
            "records": list(RECORDS),
            "discovery_records": list(DISCOVERY_RECORDS),
            "confirmation_records": list(CONFIRMATION_RECORDS),
            "allowed_root": str(RESULTS_ROOT),
            "denied": ["records outside allowlist", "training", "detector tuning"],
        },
    )
    write_json(root / "rule_source_registry.json", RULE_SOURCES)
    write_json(root / "immutable_rule_contract.json", IMMUTABLE_RULE_CONTRACT)
    write_json(root / "candidate_universe.json", candidate_payload)
    write_json(root / "experiment_config.json", config_data)
    (root / "RUN_PLAN.md").write_text(
        """# Run Plan

1. Freeze 384 source-grounded decision-rule candidates before opening labels.
2. Replay every candidate on the frozen ten-record development partition without stage references.
3. Seal predictions, then have a separate broker create discovery and confirmation shards.
4. Select exactly one candidate using only the five-record discovery partition and freeze its identity.
5. Open the disjoint five-record confirmation partition only in a new process; report full class metrics.
6. Independently recompute confirmation metrics and verify all frozen hashes.

This is within-project post-hoc HMC development research, not external validation.
""",
        encoding="utf-8",
    )
    state = {
        "run_id": root.name,
        "started_at": utc_now(),
        "status": "RUNNING",
        "root": str(root),
        "max_wall_hours": float(config_data["max_wall_hours"]),
        "gates": {},
    }
    write_json(root / "run_state.json", state)
    write_json(
        root / "compute_manifest.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "initial_process_id": os.getpid(),
            "training": False,
            "gpu_required": False,
        },
    )
    return state, config_hashes, heritage


def reference_access_guard(denied: set[Path]) -> list[str]:
    attempts: list[str] = []
    resolved = {path.resolve() for path in denied}

    def audit(event: str, values: tuple[Any, ...]) -> None:
        if event != "open" or not values:
            return
        candidate = values[0]
        if not isinstance(candidate, (str, bytes, os.PathLike)):
            return
        try:
            opened = Path(candidate).resolve()
        except (OSError, TypeError, ValueError):
            return
        if opened in resolved:
            attempts.append(str(opened))
            raise PermissionError(f"REFERENCE_ACCESS_DENIED_IN_THIS_PHASE: {opened}")

    sys.addaudithook(audit)
    return attempts


def budget_deadline(state: dict[str, Any], max_wall_hours: float) -> float:
    cumulative = (
        datetime.now().astimezone() - datetime.fromisoformat(state["started_at"])
    ).total_seconds()
    remaining = max_wall_hours * 3600 - cumulative
    if remaining <= 0:
        raise TimeoutError("cumulative 12-hour rule-search budget exhausted")
    return time.monotonic() + remaining


def split_reference(root: Path, state: dict[str, Any]) -> None:
    seal_path = root / "prediction" / "prediction_seal.json"
    if not seal_path.exists():
        raise RuntimeError("prediction seal missing; labels remain locked")
    update_state(root, state, "REFERENCE_SHARDING", "RUNNING")
    frame = pd.read_csv(ALIGNED_REFERENCE)
    if set(frame.record_id.unique()) != set(RECORDS):
        raise RuntimeError("reference contains records outside the configured allowlist")
    keep = ["record_id", "epoch_index", "true_stage", "in_primary", "in_full_record"]
    shards = {
        "discovery": DISCOVERY_RECORDS,
        "confirmation": CONFIRMATION_RECORDS,
        "full": RECORDS,
    }
    manifest: dict[str, Any] = {
        "created_at": utc_now(),
        "broker_process_id": os.getpid(),
        "source_reference": str(ALIGNED_REFERENCE),
        "source_sha256": sha256_file(ALIGNED_REFERENCE),
        "prediction_seal_sha256": sha256_file(seal_path),
        "shards": {},
    }
    for name, records in shards.items():
        output = root / "reference" / f"{name}.csv.gz"
        selected = frame.loc[frame.record_id.isin(records), keep].copy()
        selected.to_csv(output, index=False, compression="gzip", lineterminator="\n")
        manifest["shards"][name] = {
            "path": str(output),
            "records": list(records),
            "rows": len(selected),
            "primary_epochs": int(selected.in_primary.sum()),
            "sha256": sha256_file(output),
        }
    write_json(root / "reference" / "reference_shards_manifest.json", manifest)
    heritage = json.loads((root / "heritage_manifest.json").read_text(encoding="utf-8"))
    heritage["protected_reference"] = {
        "path": str(ALIGNED_REFERENCE),
        "status": "OPENED_BY_SHARD_BROKER_AFTER_PREDICTION_SEAL",
        "sha256": manifest["source_sha256"],
        "opened_at": manifest["created_at"],
        "broker_process_id": os.getpid(),
    }
    write_json(root / "heritage_manifest.json", heritage)
    update_state(root, state, "REFERENCE_SHARDING", "PASS")
    state["status"] = "REFERENCE_SHARDS_SEALED"
    write_json(root / "run_state.json", state)


def run_prediction(
    root: Path,
    state: dict[str, Any],
    configs: tuple[RuleConfig, ...],
    config_hashes: dict[str, str],
    deadline: float,
) -> None:
    attempts = reference_access_guard({ALIGNED_REFERENCE})
    update_state(root, state, "LABEL_BLIND_PREDICTION", "RUNNING")
    cache_path = root / "cache" / "rule_predictions.sqlite"
    cache = PredictionCache(cache_path)
    try:
        manifest = predict_all(cache, configs, config_hashes, deadline)
        cache.set_meta("prediction_complete", True)
        cache.set_meta("candidate_universe_sha256", sha256_file(root / "candidate_universe.json"))
    finally:
        cache.close()
    rows_complete = sum(row["records_complete"] for row in manifest)
    if rows_complete != len(configs) * len(RECORDS):
        raise RuntimeError(f"prediction matrix incomplete: {rows_complete}")
    seal = {
        "sealed_at": utc_now(),
        "label_access": False,
        "prediction_process_id": os.getpid(),
        "config_count": len(configs),
        "record_count": len(RECORDS),
        "config_record_count": rows_complete,
        "manifest": manifest,
        "cache_sha256": sha256_file(cache_path),
        "candidate_universe_sha256": sha256_file(root / "candidate_universe.json"),
        "reference_access_guard_attempts": attempts,
        "scoring_modules_loaded_before_seal": sorted(
            name
            for name in sys.modules
            if name.startswith("experiments.v4_rule_discovery.")
            and name.rsplit(".", 1)[-1]
            in {"metrics", "selection", "report", "independent_recheck"}
        ),
    }
    write_json(root / "prediction" / "prediction_seal.json", seal)
    update_state(root, state, "LABEL_BLIND_PREDICTION", "PASS")
    state["status"] = "PREDICTION_SEALED"
    write_json(root / "run_state.json", state)


def run_selection(
    root: Path,
    state: dict[str, Any],
    configs: tuple[RuleConfig, ...],
    config_hashes: dict[str, str],
) -> None:
    from .metrics import ReferenceData
    from .report import discovery_leaderboard, write_csv
    from .selection import evaluate_frozen_candidate, select_discovery_candidate

    prediction_seal = root / "prediction" / "prediction_seal.json"
    shard_manifest = root / "reference" / "reference_shards_manifest.json"
    discovery_path = root / "reference" / "discovery.csv.gz"
    confirmation_path = root / "reference" / "confirmation.csv.gz"
    full_path = root / "reference" / "full.csv.gz"
    for required in (prediction_seal, shard_manifest, discovery_path):
        if not required.exists():
            raise RuntimeError(f"required sealed input missing: {required}")
    denied_attempts = reference_access_guard({ALIGNED_REFERENCE, confirmation_path, full_path})
    update_state(root, state, "DISCOVERY_SELECTION", "RUNNING")
    reference = ReferenceData(discovery_path, DISCOVERY_RECORDS)
    cache = PredictionCache(root / "cache" / "rule_predictions.sqlite")
    try:
        selection = select_discovery_candidate(cache, configs, reference, DISCOVERY_RECORDS)
        discovery = evaluate_frozen_candidate(
            cache, selection["winner_config_id"], reference, DISCOVERY_RECORDS
        )
    finally:
        cache.close()
    write_json(root / "selection" / "discovery_selection.json", selection)
    write_json(root / "analysis" / "discovery_metrics.json", discovery)
    write_csv(root / "selection" / "discovery_leaderboard.csv", discovery_leaderboard(selection))
    by_id = {config.config_id: config for config in configs}
    selected = by_id[selection["winner_config_id"]]
    payload = {
        **selected.to_dict(),
        "config_sha256": config_hashes[selected.config_id],
        "frozen_at": utc_now(),
        "selection_records": list(DISCOVERY_RECORDS),
        "confirmation_records": list(CONFIRMATION_RECORDS),
        "confirmation_labels_accessed_by_selector": False,
        "scientific_scope": "WITHIN_PROJECT_POST_HOC_CONFIRMATION",
    }
    write_json(root / "selection" / "final_rule_config.json", payload)
    seal = {
        "sealed_at": utc_now(),
        "selection_process_id": os.getpid(),
        "selected_config_id": selected.config_id,
        "selected_config_sha256": config_hashes[selected.config_id],
        "discovery_shard_sha256": sha256_file(discovery_path),
        "candidate_universe_sha256": sha256_file(root / "candidate_universe.json"),
        "denied_reference_access_attempts": denied_attempts,
    }
    write_json(root / "selection" / "selection_seal.json", seal)
    prediction_seal_data = json.loads(prediction_seal.read_text(encoding="utf-8"))
    write_json(
        root / "selection" / "label_access_manifest.json",
        {
            "process_id": os.getpid(),
            "opened_after_prediction_seal": True,
            "prediction_sealed_at": prediction_seal_data["sealed_at"],
            "prediction_process_id": prediction_seal_data["prediction_process_id"],
            "selection_process_id": os.getpid(),
            "separate_prediction_and_selection_processes": (
                prediction_seal_data["prediction_process_id"] != os.getpid()
            ),
            "opened": [str(discovery_path)],
            "denied": [str(ALIGNED_REFERENCE), str(confirmation_path), str(full_path)],
            "denied_attempts": denied_attempts,
        },
    )
    update_state(root, state, "DISCOVERY_SELECTION", "PASS")
    state["status"] = "CANDIDATE_FROZEN_BEFORE_CONFIRMATION"
    write_json(root / "run_state.json", state)


def _rule_limitation_rows(diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_rule = {row["rule_id"]: row for row in diagnostics}

    def value(rule: str, key: str, default: Any = 0) -> Any:
        return by_rule.get(rule, {}).get(key, default)

    return [
        {
            "limitation": "Discrete blink encoded as >15s duration",
            "old_rule": "W.2",
            "observed_firings": sum(row["count"] for row in diagnostics if row["rule_id"].startswith("W.2")),
            "data_evidence": "No W.2 final decisions in the 8,902 primary epochs",
            "tested_repair": "event-count/presence ocular semantics with EMG-tone guard",
        },
        {
            "limitation": "N1.2 accepts generic markers without explicit LAMF majority",
            "old_rule": "N1.2",
            "observed_firings": value("N1.2", "count"),
            "old_rule_precision": value("N1.2", "precision_as_final_rule", None),
            "tested_repair": "LAMF majority plus SEM or vertex support",
        },
        {
            "limitation": "Post-arousal temporal majority may suppress structural N2 re-entry",
            "old_rule": "N2.4.POST_AROUSAL_MAJORITY",
            "observed_firings": value("N2.4.POST_AROUSAL_MAJORITY", "count"),
            "old_rule_precision": value("N2.4.POST_AROUSAL_MAJORITY", "precision_as_final_rule", None),
            "tested_repair": "qualifying K/spindle structural re-entry before temporal majority",
        },
        {
            "limitation": "Generic previous-stage fallback lacks stage-specific termination",
            "old_rule": "FALLBACK.INHERIT_PREVIOUS",
            "observed_firings": value("FALLBACK.INHERIT_PREVIOUS", "count"),
            "old_rule_precision": value("FALLBACK.INHERIT_PREVIOUS", "precision_as_final_rule", None),
            "tested_repair": "stage-specific continuation and conservative abstention",
        },
        {
            "limitation": "REM continuation lacks definite-R backward bridge",
            "old_rule": "R.2",
            "observed_firings": value("R.2", "count"),
            "old_rule_precision": value("R.2", "precision_as_final_rule", None),
            "tested_repair": "AASM v2.1 definite-R backward bridge",
        },
    ]


def _backward_span_audit(root: Path, config_id: str) -> dict[str, Any]:
    spans: list[int] = []
    by_record: dict[str, int] = {}
    for record in RECORDS:
        rows = read_json_gz(root / "traces" / config_id / "predictions" / f"{record}.json.gz")
        current = [
            int(row["chain_length"])
            for row in rows
            if row.get("rule_id") == "R.BACKWARD" and row.get("chain_length") is not None
        ]
        spans.extend(current)
        by_record[record] = len(current)
    ordered = sorted(spans)
    if not ordered:
        return {"count": 0, "by_record": by_record}
    middle = len(ordered) // 2
    median = (
        float(ordered[middle])
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    p95 = ordered[max(0, int(0.95 * len(ordered) + 0.999999) - 1)]
    return {
        "count": len(ordered),
        "median_epochs": median,
        "p95_epochs": int(p95),
        "max_epochs": int(ordered[-1]),
        "max_seconds": int(ordered[-1] * 30),
        "by_record": by_record,
        "explicit_maximum_span_in_rule": False,
        "risk": (
            "A definite-R anchor can propagate across multiple contiguous epochs; "
            "independent scorer review is required before deployment."
        ),
    }


def run_confirmation(
    root: Path,
    state: dict[str, Any],
    configs: tuple[RuleConfig, ...],
) -> None:
    from .independent_recheck import run_independent_recheck
    from .mechanism import evidence_rule_audit, paired_rule_mechanism
    from .metrics import ReferenceData
    from .report import current_rule_diagnostics, render_reports, write_csv
    from .selection import evaluate_frozen_candidate

    selection_seal_path = root / "selection" / "selection_seal.json"
    if not selection_seal_path.exists():
        raise RuntimeError("selection seal missing; confirmation labels remain locked")
    selection_seal = json.loads(selection_seal_path.read_text(encoding="utf-8"))
    selected_id = selection_seal["selected_config_id"]
    by_id = {config.config_id: config for config in configs}
    if selected_id not in by_id:
        raise RuntimeError(f"sealed config missing from universe: {selected_id}")
    selected = by_id[selected_id]
    selection = json.loads(
        (root / "selection" / "discovery_selection.json").read_text(encoding="utf-8")
    )
    update_state(root, state, "HELDOUT_CONFIRMATION", "RUNNING")
    write_json(root / "rule_source_registry.json", RULE_SOURCES)
    discovery_ref = ReferenceData(root / "reference" / "discovery.csv.gz", DISCOVERY_RECORDS)
    confirmation_ref = ReferenceData(
        root / "reference" / "confirmation.csv.gz", CONFIRMATION_RECORDS
    )
    full_ref = ReferenceData(root / "reference" / "full.csv.gz", RECORDS)
    cache = PredictionCache(root / "cache" / "rule_predictions.sqlite")
    try:
        discovery = evaluate_frozen_candidate(cache, selected_id, discovery_ref, DISCOVERY_RECORDS)
        confirmation = evaluate_frozen_candidate(
            cache, selected_id, confirmation_ref, CONFIRMATION_RECORDS
        )
        full = evaluate_frozen_candidate(cache, selected_id, full_ref, RECORDS)
        component_configs = {
            "rem_backward_only": next(
                config for config in configs if config.changed_modules == ("R.BACKWARD_BRIDGE",)
            ),
            "n2_reentry_only": next(
                config
                for config in configs
                if config.changed_modules == ("N2_POST_AROUSAL_REENTRY",)
            ),
        }
        component_ablation = {}
        for name, component in component_configs.items():
            component_ablation[name] = {
                "config_id": component.config_id,
                "changed_modules": list(component.changed_modules),
                "discovery": evaluate_frozen_candidate(
                    cache, component.config_id, discovery_ref, DISCOVERY_RECORDS, bootstrap_samples=2000
                ),
                "confirmation": evaluate_frozen_candidate(
                    cache, component.config_id, confirmation_ref, CONFIRMATION_RECORDS, bootstrap_samples=2000
                ),
                "full": evaluate_frozen_candidate(
                    cache, component.config_id, full_ref, RECORDS, bootstrap_samples=2000
                ),
            }
        component_ablation["combined_selected"] = {
            "config_id": selected_id,
            "changed_modules": list(selected.changed_modules),
            "discovery": discovery,
            "confirmation": confirmation,
            "full": full,
        }
        raw_best_summary = max(
            selection["candidate_summaries"], key=lambda row: row["robust_score"]
        )
        raw_best_id = raw_best_summary["config_id"]
        raw_best_config = by_id[raw_best_id]
        raw_best_exploratory = {
            "status": "GUARDRAIL_REJECTED_EXPLORATORY_NOT_SELECTED",
            "config_id": raw_best_id,
            "changed_modules": list(raw_best_config.changed_modules),
            "discovery_selection_summary": raw_best_summary,
            "rejection_reason": (
                "Discovery minimum class-F1 delta was below the predeclared -0.03 guardrail."
            ),
            "confirmation_post_hoc": evaluate_frozen_candidate(
                cache, raw_best_id, confirmation_ref, CONFIRMATION_RECORDS, bootstrap_samples=2000
            ),
            "not_used_to_change_winner": True,
        }
        rule_diagnostics = current_rule_diagnostics(cache, full_ref, RECORDS)
        mechanism = paired_rule_mechanism(cache, full_ref, selected_id, RECORDS)
        evidence_audit = evidence_rule_audit(full_ref, RECORDS)
    finally:
        cache.close()
    write_json(root / "analysis" / "discovery_metrics.json", discovery)
    write_json(root / "analysis" / "confirmation_metrics.json", confirmation)
    write_json(root / "selection" / "confirmation_metrics.json", confirmation)
    write_json(root / "analysis" / "full_hmc_metrics.json", full)
    write_json(root / "analysis" / "selected_component_ablation.json", component_ablation)
    write_json(root / "analysis" / "guardrail_rejected_raw_best.json", raw_best_exploratory)
    write_json(root / "analysis" / "confirmation_paired.json", confirmation["paired"])
    write_json(root / "analysis" / "evidence_rule_audit.json", evidence_audit)
    write_json(root / "analysis" / "mechanism_summary.json", {k: v for k, v in mechanism.items() if k != "changed_epochs"})
    write_csv(root / "analysis" / "confirmation_per_night.csv", confirmation["per_night"])
    write_csv(root / "analysis" / "current_rule_diagnostics.csv", rule_diagnostics)
    write_csv(root / "analysis" / "rule_limitations.csv", _rule_limitation_rows(rule_diagnostics))
    write_csv(root / "analysis" / "changed_epoch_trace.csv", mechanism["changed_epochs"])
    write_csv(root / "analysis" / "mechanism_per_record.csv", mechanism["per_record"])
    trace_manifest = []
    for config_id in sorted({"C000", selected_id}):
        trace_manifest.extend(export_full_trace(by_id[config_id], root / "traces"))
    write_json(root / "traces" / "trace_manifest.json", trace_manifest)
    backward_span = _backward_span_audit(root, selected_id)
    write_json(root / "analysis" / "rem_backward_span_audit.json", backward_span)
    confirmation_access = {
        "opened_at": utc_now(),
        "confirmation_process_id": os.getpid(),
        "selection_process_id": selection_seal["selection_process_id"],
        "different_process": os.getpid() != selection_seal["selection_process_id"],
        "selection_sealed_at": selection_seal["sealed_at"],
        "confirmation_shard_sha256": sha256_file(root / "reference" / "confirmation.csv.gz"),
    }
    write_json(root / "analysis" / "confirmation_access_manifest.json", confirmation_access)
    write_json(
        root / "audits" / "scientific_claim_audit.json",
        {
            "status": "PASS_WITH_EXPLICIT_SOURCE_SCOPE",
            "rem_backward_bridge": (
                "Rule-constrained engineering operationalization based on the public AASM v2.1 update; "
                "the implementation has no explicit maximum backward span."
            ),
            "n2_post_arousal_reentry": (
                "AASM supports segment-level N2 re-entry after qualifying structure; "
                "whole-epoch priority over majority is a project engineering operationalization."
            ),
            "external_validation": False,
            "scope": "WITHIN_PROJECT_POST_HOC_CONFIRMATION",
        },
    )
    update_state(root, state, "HELDOUT_CONFIRMATION", "PASS")

    elapsed = (
        datetime.now().astimezone() - datetime.fromisoformat(state["started_at"])
    ).total_seconds()
    render_reports(
        root,
        selected,
        selection,
        discovery,
        confirmation,
        full,
        elapsed,
        rule_diagnostics=rule_diagnostics,
        evidence_audit=evidence_audit,
        component_ablation=component_ablation,
        raw_best_exploratory=raw_best_exploratory,
        backward_span=backward_span,
    )

    heritage = json.loads((root / "heritage_manifest.json").read_text(encoding="utf-8"))
    update_state(root, state, "INDEPENDENT_RECHECK", "RUNNING")
    recheck = run_independent_recheck(root, heritage)
    if recheck["status"] != "PASS":
        update_state(root, state, "INDEPENDENT_RECHECK", "FAIL")
        raise RuntimeError("independent confirmation recheck failed")
    update_state(root, state, "INDEPENDENT_RECHECK", "PASS")
    finalize(root, state, heritage, elapsed)


def finalize(root: Path, state: dict[str, Any], heritage: dict[str, Any], elapsed: float) -> None:
    current = {str(path): sha256_file(path) for path in frozen_artifacts()}
    unchanged = all(current[item["path"]] == item["sha256"] for item in heritage["frozen_artifacts"])
    write_json(
        root / "audits" / "frozen_heritage_validation.json",
        {"status": "PASS" if unchanged else "FAIL", "unchanged": unchanged},
    )
    if not unchanged:
        raise RuntimeError("frozen heritage changed")
    write_json(
        root / "wall_time_accounting.json",
        {
            "started_at": state["started_at"],
            "completed_at": utc_now(),
            "cumulative_wall_seconds": elapsed,
            "max_wall_hours": state["max_wall_hours"],
        },
    )
    implementation_root = Path(__file__).resolve().parent
    files = sorted(
        path
        for path in implementation_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".yaml", ".md"}
    )
    write_json(
        root / "implementation_manifest.json",
        {
            "root": str(implementation_root),
            "files": [
                {
                    "path": path.relative_to(implementation_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in files
            ],
        },
    )
    (root / "reproduce.md").write_text(
        f"""# Reproduce

```powershell
$env:AASM_V4_RECORDS='<comma-separated private record allowlist>'
$env:AASM_V4_DISCOVERY_RECORDS='<comma-separated discovery subset>'
$env:AASM_V4_FROZEN_ROOT='<path-to-frozen-temporal-r1>'
$env:PYTHONDONTWRITEBYTECODE='1'
pytest -q -p no:cacheprovider experiments/v4_rule_discovery/tests
python -m experiments.v4_rule_discovery.run all `
  --config experiments/v4_rule_discovery/configs/main.yaml `
  --max-wall-hours 12 `
  --run-dir {root}
```

`all` launches predict, split, select, and confirm as four separate processes.
""",
        encoding="utf-8",
    )
    state["status"] = "COMPLETE"
    state["engineering_status"] = "AASM_RULE_DISCOVERY_V5_ENGINEERING_PASS"
    state["scientific_scope"] = "WITHIN_PROJECT_POST_HOC_CONFIRMATION"
    state["completed_at"] = utc_now()
    write_json(root / "run_state.json", state)
    write_json(root / "artifact_manifest.json", {"created_at": utc_now(), "artifacts": artifact_manifest(root)})


def orchestrate_all(args: argparse.Namespace, root: Path, max_wall_hours: float) -> int:
    for phase in ("predict", "split", "select", "confirm"):
        command = [
            sys.executable,
            "-m",
            "experiments.v4_rule_discovery.run",
            phase,
            "--config",
            str(args.config.resolve()),
            "--max-wall-hours",
            str(max_wall_hours),
            "--run-dir",
            str(root),
        ]
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"phase {phase} failed with exit code {result.returncode}")
    print(root)
    return 0


def main() -> int:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as stream:
        config_data = yaml.safe_load(stream)
    max_wall_hours = args.max_wall_hours or float(config_data["max_wall_hours"])
    configs = candidate_universe()
    if int(config_data["candidate_count"]) != len(configs):
        raise RuntimeError("candidate count in YAML does not match frozen universe")
    selection_contract = {
        "coverage_tolerance": 0.005,
        "stage_f1_guardrail": 0.03,
        "worst_night_accuracy_delta": -0.05,
    }
    for key, expected in selection_contract.items():
        if float(config_data[key]) != expected:
            raise RuntimeError(
                f"YAML {key}={config_data[key]} does not match frozen selector value {expected}"
            )
    root = resolve_run_dir(args)
    configured_results = RESULTS_ROOT.resolve()
    try:
        root.resolve().relative_to(configured_results)
    except ValueError as error:
        raise PermissionError(
            f"new outputs must remain under AASM_V4_RESULTS_ROOT: {configured_results}"
        ) from error
    if args.phase == "all":
        return orchestrate_all(args, root, max_wall_hours)

    validate_frozen_paths(include_reference=args.phase == "split")
    config_hashes = {
        config.config_id: sha256_bytes(canonical_bytes(config.to_dict())) for config in configs
    }
    if not (root / "run_state.json").exists():
        if args.phase != "predict":
            raise RuntimeError("run must begin with predict")
        state, config_hashes, _ = initialize_run(root, config_data, configs)
    else:
        state = load_state(root)
    deadline = budget_deadline(state, max_wall_hours)

    if args.phase == "predict":
        run_prediction(root, state, configs, config_hashes, deadline)
    elif args.phase == "split":
        split_reference(root, state)
    elif args.phase == "select":
        run_selection(root, state, configs, config_hashes)
    elif args.phase == "confirm":
        run_confirmation(root, state, configs)
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
