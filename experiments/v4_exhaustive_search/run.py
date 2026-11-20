from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sqlite3
import time
from typing import Any

import numpy as np
import pandas as pd
import psutil
import yaml

from experiments.v4_subset_search.config import AR_MASK, ArmSpec
from experiments.v4_subset_search.replay import (
    build_cache_identity,
    replay_full_trace,
)
from experiments.v4_subset_search.storage import CacheDB

from .config import (
    FULL_MASK,
    HANDLERS,
    RECORDS,
    RESULTS,
    SOURCE_CACHE,
    SOURCE_HYBRID_RUN,
    ExtendedLimits,
    deterministic_mask_order,
    validate_paths,
)
from .fast_replay import FastReplayCoordinator, clone_label_blind_cache, sha256
from .independent_recheck import run_independent_recheck
from .selection import (
    ReferenceData,
    full_development_results,
    load_confusion_cube,
    nested_paired_counts,
    score_frozen_winners,
    select_outer_winners,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Any, *, fsync: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())
    temporary.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _state(root: Path, status: str, phase: str, start_epoch: float, **extra: Any) -> None:
    _write_json(root / "run_state.json", {
        "status": status,
        "phase": phase,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - start_epoch,
        **extra,
    })


def _artifact_manifest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.name == "artifact_manifest.json"
            or path.suffix in {".wal", ".shm"}
        ):
            continue
        rows.append({
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return {"root": str(root.resolve()), "file_count": len(rows), "files": rows}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extended exact search over 15 frozen handlers")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "main.yaml",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--seed-cache",
        type=Path,
        help=(
            "Optional label-blind compact cache used to seed a fresh run. "
            "Search history and frozen winners are stripped during cloning."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-wall-hours", type=float)
    parser.add_argument("--replay-wall-hours", type=float)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--pilot-only", action="store_true")
    return parser


def _limits(config: dict[str, Any], args: argparse.Namespace) -> ExtendedLimits:
    return ExtendedLimits(
        max_wall_hours=args.max_wall_hours or float(config["max_wall_hours"]),
        replay_wall_hours=args.replay_wall_hours or float(config["replay_wall_hours"]),
        worker_count=args.workers or int(config["worker_count"]),
        masks_per_chunk=int(config["masks_per_chunk"]),
        shards_per_record=int(config["shards_per_record"]),
        restart_every_chunks=int(config["restart_every_chunks"]),
        benchmark_masks=int(config["benchmark_masks"]),
        exhaustive_rate_threshold_per_hour=float(config["exhaustive_rate_threshold_per_hour"]),
        seed=int(config["seed"]),
        coverage_floor=float(config["coverage_floor"]),
        epsilon=float(config["epsilon"]),
    )


def _compute_manifest(limits: ExtendedLimits) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpu": psutil.cpu_count(logical=True),
        "physical_cpu": psutil.cpu_count(logical=False),
        "ram_gib": memory.total / 2**30,
        "available_ram_gib": memory.available / 2**30,
        "worker_count": limits.worker_count,
        "replay_is_cpu_bound": True,
    }


def _load_safe_warm_masks() -> tuple[dict[str, set[int]], dict[str, Any]]:
    output: dict[str, set[int]] = {}
    evidence = []
    for record in RECORDS:
        fold_id = f"OUTER_{record}"
        path = SOURCE_HYBRID_RUN / "search" / "outer" / f"{fold_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        masks = {int(mask) for mask in payload["evaluated_masks"]}
        output[fold_id] = masks
        evidence.append({
            "fold_id": fold_id,
            "heldout_record": record,
            "safe_mask_count": len(masks),
            "source_path": str(path.resolve()),
            "source_sha256": sha256(path),
        })
    connection = sqlite3.connect(
        f"file:{SOURCE_CACHE.resolve().as_posix()}?mode=ro&immutable=1", uri=True
    )
    failures = []
    event_count = 0
    try:
        for fold_id, payload_text in connection.execute(
            "SELECT fold_id,payload FROM search_events WHERE fold_id LIKE 'OUTER_%'"
        ):
            payload = json.loads(payload_text)
            records_used = payload.get("records_used")
            if records_used is None:
                continue
            event_count += 1
            heldout = fold_id.removeprefix("OUTER_")
            expected = set(RECORDS) - {heldout}
            if set(records_used) != expected or heldout in records_used:
                failures.append(fold_id)
    finally:
        connection.close()
    if failures:
        raise RuntimeError(f"old warm-start provenance leaked heldout records: {failures}")
    return output, {
        "status": "PASS",
        "old_outer_search_events_checked": event_count,
        "heldout_access_failures": failures,
        "folds": evidence,
        "old_heldout_or_final_selection_read": False,
    }


def _next_new_chunk(
    order: tuple[int, ...],
    next_index: int,
    complete_masks: set[int],
    target_new: int,
) -> tuple[tuple[int, ...], int]:
    new_masks = []
    index = next_index
    while index < len(order) and len(new_masks) < target_new:
        mask = order[index]
        if mask not in complete_masks:
            new_masks.append(mask)
        index += 1
    return tuple(new_masks), index


def _structural_hash(masks: set[int]) -> str:
    return hashlib.sha256(
        ",".join(map(str, sorted(masks))).encode("utf-8")
    ).hexdigest()


def _require_budget(progress: dict[str, Any], required_seconds: float, phase: str) -> None:
    remaining = float(progress["deadline_epoch"]) - time.time()
    if remaining < required_seconds:
        raise TimeoutError(
            f"hard 16-hour budget cannot safely enter {phase}: "
            f"{remaining:.1f}s remain, {required_seconds:.1f}s required"
        )


def _write_reports(
    root: Path,
    progress: dict[str, Any],
    full: dict[str, Any],
    nested: dict[str, Any],
    limits: ExtendedLimits,
    start_epoch: float,
    seed_complete_masks: int,
    seed_cache: Path,
    scientific_label: str,
) -> None:
    champion = full["balanced_champion"]
    delta = nested["delta"]
    status = "EXHAUSTIVE" if progress["exhaustive_complete"] else "EXTENDED_PARTIAL"
    text = f"""# {limits.max_wall_hours:g}-hour extended subset search

## 结论

本轮从既有 {seed_complete_masks:,} 个 exact masks 热启动，在不修改 Detector、eventization、AASM rules
或 Temporal R1 的前提下，共完成 **{progress['complete_mask_count']:,}/32,768** 个组合
（{progress['complete_mask_count'] / 32768:.2%}）。搜索状态：`{status}`。

| 结果 | Mask | Handlers | Accuracy | Macro-F1 | Coverage |
|---|---:|---|---:|---:|---:|
| Best eligible development candidate | {int(champion['mask'])} | {champion['handlers'] or 'B000'} | {champion['accuracy']:.6f} | {champion['macro_f1']:.6f} | {champion['coverage']:.6f} |

Extended nested pipeline 相对 B000：Accuracy {delta['accuracy']:+.6f}，
Macro-F1 {delta['macro_f1']:+.6f}，Coverage {delta['coverage']:+.6f}；
Corrected {nested['paired_counts']['corrected']}，Introduced {nested['paired_counts']['introduced']}，
Net {nested['paired_counts']['net_correct']:+d}。

## 证据边界

- 科学标签：`{scientific_label}`。
- outer-fold评分仍由代码严格隔离，但不是新的外部盲测。
- 若达到32,768，可称固定15-bit替换空间、固定规则与当前HMC开发集上的exact optimum；不能外推为外部泛化最优。
- 实际总墙钟：{(time.time() - start_epoch) / 3600:.3f} h；硬预算：{limits.max_wall_hours:.1f} h。
"""
    _write_text(root / "EXECUTIVE_SUMMARY.md", text)
    _write_text(root / "FINAL_EXTENDED_SEARCH_REPORT.md", text + """

## 方法

旧cache仅复用label-blind compact stage predictions；旧search events、winner和report unlock均被清除。
新mask按 `(cardinality, mask)` 的预声明顺序进行exact replay，十四个进程并行，SQLite批量事务封存。
每个outer fold只能使用结构性mask前缀和该折旧selection阶段安全产生的warm-start masks。
全部十个新winner先写盘并冻结，随后才统一读取held-out night。
""")
    _write_text(root / "reproduce.md", f"""# Reproduce / resume

```powershell
Set-Location <repository-root>
python -m experiments.v4_exhaustive_search.run `
  --config "{Path(__file__).resolve().parent / 'configs' / 'full_completion.yaml'}" `
  --max-wall-hours {limits.max_wall_hours} --replay-wall-hours {limits.replay_wall_hours} `
  --workers {limits.worker_count} --seed-cache "{seed_cache}" `
  --resume --run-dir "{root.resolve()}"
```
""")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    validate_paths()
    config_data = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    limits = _limits(config_data, args)
    if limits.replay_wall_hours >= limits.max_wall_hours:
        raise ValueError("replay budget must reserve time for selection and audit")
    run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    root = args.run_dir or RESULTS / f"phase2_extended_exact_search_{run_id}"
    if root.drive.upper() != "D:":
        raise RuntimeError("all new artifacts must be on D:")
    if root.exists() and not args.resume:
        raise FileExistsError(f"run directory exists; use --resume: {root}")
    root.mkdir(parents=True, exist_ok=True)
    seed_cache = (args.seed_cache or SOURCE_CACHE).resolve()
    if not seed_cache.exists() or not seed_cache.is_file():
        raise FileNotFoundError(f"seed cache does not exist: {seed_cache}")

    progress_path = root / "replay" / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        start_epoch = float(progress["start_epoch"])
    else:
        start_epoch = time.time()
        progress = {
            "start_epoch": start_epoch,
            "deadline_epoch": start_epoch + limits.max_wall_hours * 3600,
            "replay_deadline_epoch": start_epoch + limits.replay_wall_hours * 3600,
            "next_order_index": 0,
            "chunks_completed": 0,
            "new_masks_completed": 0,
            "benchmark_complete": False,
            "benchmark_rate_masks_per_hour": None,
            "exhaustive_plausible_from_benchmark": None,
            "failures_recovered": 0,
        }
    _write_json(progress_path, progress, fsync=True)
    _state(root, "RUNNING", "PREFLIGHT", start_epoch)
    _write_json(root / "compute_manifest.json", _compute_manifest(limits))
    _write_json(root / "RUN_PLAN.json", {
        "scientific_label": config_data["scientific_label"],
        "limits": asdict(limits),
        "mask_order": "cardinality_then_integer_label_blind",
        "source_cache": str(SOURCE_CACHE.resolve()),
        "seed_cache": str(seed_cache),
    })

    cache_key, identity = build_cache_identity(args.config)
    source_hash_before = sha256(SOURCE_CACHE)
    seed_cache_hash_before = sha256(seed_cache)
    source_connection = sqlite3.connect(
        f"file:{seed_cache.as_posix()}?mode=ro&immutable=1", uri=True
    )
    try:
        source_seed_complete_masks = int(
            source_connection.execute(
                "SELECT COUNT(*) FROM arms WHERE status='PASS'"
            ).fetchone()[0]
        )
    finally:
        source_connection.close()
    cache_path = root / "cache" / "compact_predictions.sqlite"
    clone = clone_label_blind_cache(seed_cache, cache_path, cache_key)
    safe_warm_masks, warm_audit = _load_safe_warm_masks()
    _write_json(root / "audits" / "warm_start_leakage_audit.json", warm_audit)
    heritage = {
        "source_cache_sha256_before": source_hash_before,
        "source_cache_bytes": SOURCE_CACHE.stat().st_size,
        "seed_cache": str(seed_cache),
        "seed_cache_sha256_before": seed_cache_hash_before,
        "seed_cache_complete_masks": source_seed_complete_masks,
        "cache_clone": clone,
        "cache_identity": identity,
        "protected_source_run": str(SOURCE_HYBRID_RUN.resolve()),
    }
    _write_json(root / "heritage_manifest.json", heritage)

    cache = CacheDB(cache_path, cache_key)
    cache.reset_interrupted()
    cache.set_meta("extended_limits", asdict(limits))
    cache.set_meta("report_unlocked", False)
    order = deterministic_mask_order()
    try:
        _state(root, "RUNNING", "EXACT_REPLAY", start_epoch)
        recent_rates: list[float] = []
        with FastReplayCoordinator(cache, cache_key, limits) as coordinator:
            while int(progress["next_order_index"]) < len(order):
                now = time.time()
                remaining = float(progress["replay_deadline_epoch"]) - now
                if remaining <= 300:
                    break
                complete = cache.complete_masks()
                target = limits.masks_per_chunk
                if recent_rates:
                    conservative_rate = min(recent_rates[-3:])
                    safe_target = int(conservative_rate * remaining / 3600 * 0.80)
                    target = min(target, max(0, safe_target))
                    if target < 64:
                        break
                new_masks, next_index = _next_new_chunk(
                    order,
                    int(progress["next_order_index"]),
                    complete,
                    target,
                )
                if not new_masks:
                    progress["next_order_index"] = next_index
                    if next_index >= len(order):
                        break
                    continue
                result = coordinator.replay(new_masks)
                if result["incomplete_masks"]:
                    retry = coordinator.retry_once(result["incomplete_masks"])
                    progress["failures_recovered"] += len(result["incomplete_masks"])
                    if retry["incomplete_masks"]:
                        raise RuntimeError(
                            f"exact replay failed after retry: {retry['incomplete_masks'][:20]}"
                        )
                progress["next_order_index"] = next_index
                progress["chunks_completed"] += 1
                progress["new_masks_completed"] += int(result["new_complete_masks"])
                rate = (
                    float(result["new_complete_masks"]) / float(result["wall_seconds"]) * 3600
                    if result["wall_seconds"] and result["new_complete_masks"]
                    else None
                )
                if rate:
                    recent_rates.append(rate)
                if not progress["benchmark_complete"]:
                    progress["benchmark_complete"] = True
                    progress["benchmark_rate_masks_per_hour"] = rate
                    progress["exhaustive_plausible_from_benchmark"] = bool(
                        rate and rate >= limits.exhaustive_rate_threshold_per_hour
                    )
                    _write_json(root / "replay" / "benchmark.json", {
                        **result,
                        "rate_masks_per_hour": rate,
                        "threshold": limits.exhaustive_rate_threshold_per_hour,
                    })
                    if args.pilot_only:
                        complete = cache.complete_masks()
                        structural = set(order[: int(progress["next_order_index"])])
                        progress.update({
                            "complete_mask_count": len(complete),
                            "structural_prefix_sha256": _structural_hash(structural),
                            "exhaustive_complete": len(complete) == FULL_MASK + 1,
                        })
                        _write_json(progress_path, progress)
                        _state(root, "PAUSED", "PILOT_COMPLETE", start_epoch)
                        return 0
                complete = cache.complete_masks()
                structural = set(order[: int(progress["next_order_index"])])
                if not structural <= complete:
                    raise RuntimeError("structural prefix contains incomplete mask")
                progress.update({
                    "complete_mask_count": len(complete),
                    "structural_mask_count": len(structural),
                    "structural_prefix_sha256": _structural_hash(structural),
                    "last_chunk": result,
                    "recent_rate_masks_per_hour": recent_rates[-5:],
                    "exhaustive_complete": len(complete) == FULL_MASK + 1,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
                _write_json(progress_path, progress, fsync=True)
                _state(
                    root,
                    "RUNNING",
                    "EXACT_REPLAY",
                    start_epoch,
                    complete_masks=len(complete),
                    next_order_index=progress["next_order_index"],
                    rate_masks_per_hour=rate,
                )
                if len(complete) == FULL_MASK + 1:
                    break

        complete_masks = cache.complete_masks()
        structural_masks = set(order[: int(progress["next_order_index"])])
        progress.update({
            "complete_mask_count": len(complete_masks),
            "new_masks_completed": len(complete_masks) - source_seed_complete_masks,
            "structural_mask_count": len(structural_masks),
            "structural_prefix_sha256": _structural_hash(structural_masks),
            "exhaustive_complete": len(complete_masks) == FULL_MASK + 1,
            "replay_finished_at": datetime.now(timezone.utc).isoformat(),
        })
        _write_json(progress_path, progress, fsync=True)

        _require_budget(progress, 600.0, "NESTED_SELECTION")
        _state(root, "RUNNING", "NESTED_SELECTION", start_epoch)
        cache.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        cache.connection.commit()
        reference = ReferenceData()
        mask_values, cube = load_confusion_cube(cache_path, complete_masks, reference)
        frozen_path = root / "selection" / "outer_winners_frozen.json"
        if frozen_path.exists():
            frozen_payload = json.loads(frozen_path.read_text(encoding="utf-8"))
            winners = frozen_payload["winners"]
            frozen_at = frozen_payload["frozen_at"]
        else:
            winners = select_outer_winners(
                mask_values,
                cube,
                structural_masks,
                safe_warm_masks,
                limits,
                exhaustive=bool(progress["exhaustive_complete"]),
            )
            frozen_at = datetime.now(timezone.utc).isoformat()
            frozen_payload = {
                "frozen_at": frozen_at,
                "all_winners_frozen_before_heldout": True,
                "exhaustive_candidate_universe": bool(progress["exhaustive_complete"]),
                "winners": winners,
            }
            _write_json(frozen_path, frozen_payload, fsync=True)
        existing_winners = {
            row[0]: (int(row[1]), json.loads(row[2]))
            for row in cache.connection.execute(
                "SELECT fold_id,winner_mask,selection_payload FROM fold_winners"
            ).fetchall()
        }
        for row in winners:
            existing = existing_winners.get(row["fold_id"])
            if existing is not None and (
                existing[0] != int(row["winner_mask"]) or existing[1] != row
            ):
                raise RuntimeError(
                    f"resume winner mismatch for frozen fold {row['fold_id']}"
                )
        cache.connection.executemany(
            """INSERT OR REPLACE INTO fold_winners(
                   fold_id,heldout_record,winner_mask,status,selection_payload,frozen_at)
               VALUES(?,?,?,?,?,?)""",
            [
                (
                    row["fold_id"],
                    row["heldout_record"],
                    row["winner_mask"],
                    "WINNER_FROZEN",
                    json.dumps(row, ensure_ascii=False, sort_keys=True),
                    frozen_at,
                )
                for row in winners
            ],
        )
        cache.connection.commit()

        scoring_started = datetime.now(timezone.utc).isoformat()
        scored = score_frozen_winners(mask_values, cube, winners, limits)
        paired = nested_paired_counts(cache_path, winners, reference)
        nested = {
            "heldout_scoring_started_at": scoring_started,
            "heldout_rows": scored["rows"],
            "candidate": scored["candidate"],
            "baseline": scored["baseline"],
            "delta": scored["delta"],
            "bootstrap": scored["bootstrap"],
            "paired_counts": paired,
        }
        _write_json(root / "selection" / "nested_results.json", nested)
        scored_at = datetime.now(timezone.utc).isoformat()
        cache.connection.execute(
            """UPDATE fold_winners SET status='HELDOUT_SCORED',heldout_scored_at=?
               WHERE status='WINNER_FROZEN'""",
            (scored_at,),
        )
        cache.connection.commit()
        cache.set_meta("report_unlocked", True)

        _require_budget(progress, 900.0, "FULL_DEVELOPMENT_REPORT")
        _state(root, "RUNNING", "FULL_DEVELOPMENT_REPORT", start_epoch)
        leaderboard, full = full_development_results(mask_values, cube, limits)
        leaderboard.to_csv(root / "selection" / "exact_leaderboard.csv", index=False)
        _write_json(root / "selection" / "full_development_champions.json", full)

        _require_budget(progress, 900.0, "FINALIST_TRACE_MATERIALIZATION")
        finalist_masks = {
            0,
            AR_MASK,
            25608,
            int(full["accuracy_champion"]["mask"]),
            int(full["balanced_champion"]["mask"]),
        }
        finalist_masks.update(int(row["winner_mask"]) for row in winners)
        traces = []
        for mask in sorted(finalist_masks):
            for record in RECORDS:
                trace = replay_full_trace(mask, record, root / "finalists" / "full_trace")
                if trace["stage_sha256"] != cache.record_hash(mask, record):
                    raise RuntimeError(f"full trace mismatch: {mask}/{record}")
                traces.append(trace)
        _write_json(root / "finalists" / "manifest.json", traces)

        cache.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        cache.connection.commit()
        cache.close()

        _require_budget(progress, 300.0, "INDEPENDENT_RECHECK")
        _state(root, "RUNNING", "INDEPENDENT_RECHECK", start_epoch)
        recheck = run_independent_recheck(root, cache_path)
        _write_json(root / "audits" / "independent_recheck.json", recheck)
        if recheck["status"] != "PASS":
            raise RuntimeError("independent recheck failed")
        if sha256(SOURCE_CACHE) != source_hash_before:
            raise RuntimeError("protected source cache changed")
        if sha256(seed_cache) != seed_cache_hash_before:
            raise RuntimeError("protected seed cache changed")
        ending_cache_key, ending_identity = build_cache_identity(args.config)
        if (
            ending_cache_key != cache_key
            or ending_identity["cache_semantics"] != identity["cache_semantics"]
        ):
            raise RuntimeError("frozen replay semantics changed during execution")
        _write_json(root / "audits" / "frozen_heritage_validation.json", {
            "status": "PASS",
            "source_cache_sha256_before": source_hash_before,
            "source_cache_sha256_after": sha256(SOURCE_CACHE),
            "seed_cache": str(seed_cache),
            "seed_cache_sha256_before": seed_cache_hash_before,
            "seed_cache_sha256_after": sha256(seed_cache),
            "cache_key_before": cache_key,
            "cache_key_after": ending_cache_key,
            "cache_semantics_unchanged": True,
        })

        _write_reports(
            root,
            progress,
            full,
            nested,
            limits,
            start_epoch,
            source_seed_complete_masks,
            seed_cache,
            str(config_data["scientific_label"]),
        )
        wall = {
            "actual_wall_seconds": time.time() - start_epoch,
            "actual_wall_hours": (time.time() - start_epoch) / 3600,
            "max_wall_hours": limits.max_wall_hours,
            "replay_wall_hours": limits.replay_wall_hours,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(root / "wall_time_accounting.json", wall)
        engineering_status = (
            "FULL_EXHAUSTIVE_SEARCH_ENGINEERING_PASS"
            if progress["exhaustive_complete"]
            else "EXTENDED_EXACT_SEARCH_ENGINEERING_PASS"
        )
        _state(
            root,
            "PASS",
            "COMPLETE",
            start_epoch,
            engineering_status=engineering_status,
            complete_masks=progress["complete_mask_count"],
            exhaustive=progress["exhaustive_complete"],
        )
        _write_json(root / "artifact_manifest.json", _artifact_manifest(root))
        print(json.dumps({
            "status": "PASS",
            "run_root": str(root.resolve()),
            "complete_masks": progress["complete_mask_count"],
            "exhaustive": progress["exhaustive_complete"],
            "balanced_champion": full["balanced_champion"],
            "nested_delta": nested["delta"],
            "wall_seconds": wall["actual_wall_seconds"],
        }, ensure_ascii=False, default=_json_default))
        return 0
    except Exception as exc:
        try:
            cache.close()
        except Exception:
            pass
        _state(
            root,
            "FAILED",
            "ERROR",
            start_epoch,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
