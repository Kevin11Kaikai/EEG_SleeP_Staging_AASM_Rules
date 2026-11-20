from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from .config import (
    AR_MASK,
    BASE_BANK_ROOT,
    FROZEN_CORE,
    HANDLERS,
    RECORDS,
    SOURCE_BANK_ROOT,
    SOURCE_SEAL,
    STAGE_TO_CODE,
    ArmSpec,
    SearchLimits,
)

from experiments.v4_36arm_ablation import config as _heritage_config

_heritage_config.BASE_BANK_ROOT = BASE_BANK_ROOT
_heritage_config.FROZEN_CORE = FROZEN_CORE
_heritage_config.R1_ROOT = FROZEN_CORE / "results" / "development_r1"

from experiments.v4_36arm_ablation.io_utils import (
    canonical_hash,
    read_gzip_json,
    read_json,
    sha256,
    write_gzip_json,
    write_json,
)
from experiments.v4_36arm_ablation.shadow_overlay import (
    _frozen_imports,
    _postprocess_unet_qc,
    overlay_handlers,
)

from .storage import CacheDB


_WORKER_CACHE: dict[str, tuple[dict[str, Any], dict[str, dict[str, Any]], Any, Any, Any]] = {}


def _load_record(record_id: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Any, Any, Any]:
    cached = _WORKER_CACHE.get(record_id)
    if cached is not None:
        return cached
    if record_id not in RECORDS:
        raise PermissionError(f"record outside allow-list: {record_id}")
    base_path = BASE_BANK_ROOT / f"{record_id}.json.gz"
    base = read_gzip_json(base_path)
    sources: dict[str, dict[str, Any]] = {}
    for handler in HANDLERS:
        path = SOURCE_BANK_ROOT / handler / f"{record_id}.json.gz"
        payload = read_gzip_json(path)
        payload["_source_bank_sha256"] = sha256(path)
        sources[handler] = payload
    aggregate, engine, index_packets = _frozen_imports()
    cached = (base, sources, aggregate, engine, index_packets)
    # Bound worker RAM even when Windows' process scheduler moves records between workers.
    _WORKER_CACHE.clear()
    _WORKER_CACHE[record_id] = cached
    return cached


def replay_record_batch(record_id: str, masks: tuple[int, ...]) -> list[dict[str, Any]]:
    """Label-blind replay. This worker never imports or receives stage references."""
    base, sources, aggregate, engine, index_packets = _load_record(record_id)
    output: list[dict[str, Any]] = []
    for mask in masks:
        started = time.perf_counter()
        arm = ArmSpec(mask)
        try:
            if mask == 0:
                integrated = deepcopy(base)
                effective = 0
            else:
                integrated, manifest = overlay_handlers(base, sources, arm.handlers)
                effective = int(manifest["effective_replacement_count"])
            rows = aggregate(integrated, "enhanced")
            if mask & AR_MASK:
                _postprocess_unet_qc(rows, integrated, sources["AR"]["joint_valid_mask"])
            predictions = engine(rows, index_packets(integrated, rows))
            stages = np.fromiter(
                (STAGE_TO_CODE[row["stage"]] for row in predictions),
                dtype=np.uint8,
                count=len(predictions),
            )
            if not len(stages) or np.any(stages > 5):
                raise RuntimeError("invalid compact stage prediction")
            digest = hashlib.sha256(stages.tobytes()).hexdigest()
            output.append({
                "mask": mask,
                "record_id": record_id,
                "status": "PASS",
                "stage_codes": stages,
                "epoch_count": len(stages),
                "stage_sha256": digest,
                "runtime_seconds": time.perf_counter() - started,
                "effective_replacements": effective,
                "error": None,
            })
        except Exception as exc:  # preserve every failure for bounded recovery
            output.append({
                "mask": mask,
                "record_id": record_id,
                "status": "FAILED",
                "stage_codes": None,
                "epoch_count": None,
                "stage_sha256": None,
                "runtime_seconds": time.perf_counter() - started,
                "effective_replacements": None,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return output


def build_cache_identity(config_path: Path) -> tuple[str, dict[str, Any]]:
    seal = read_json(SOURCE_SEAL)
    if seal.get("status") != "SEALED" or seal.get("stage_reference_read") is not False:
        raise RuntimeError("frozen neural source-bank seal is not label-blind SEALED")
    if int(seal.get("records", -1)) != len(RECORDS) or int(seal.get("handlers", -1)) != len(HANDLERS):
        raise RuntimeError("frozen neural source-bank dimensions changed")
    if int(seal.get("source_bank_count", -1)) != len(RECORDS) * len(HANDLERS):
        raise RuntimeError("frozen neural source-bank count changed")
    source_manifest_path = SOURCE_SEAL.with_name("model_inference_manifest.json")
    source_manifest = read_json(source_manifest_path)
    expected_rows = {
        (row["record_id"], row["handler_id"]): row["source_bank_sha256"]
        for row in source_manifest["rows"]
    }
    actual_rows = []
    for record in RECORDS:
        for handler in HANDLERS:
            path = SOURCE_BANK_ROOT / handler / f"{record}.json.gz"
            digest = sha256(path)
            if expected_rows.get((record, handler)) != digest:
                raise RuntimeError(f"source bank hash changed: {record}/{handler}")
            actual_rows.append((record, handler, digest))
    if canonical_hash(actual_rows) != seal["source_bank_key_hash"]:
        raise RuntimeError("source-bank canonical seal mismatch")
    freeze = read_json(FROZEN_CORE / "configs" / "repair_freeze.json")
    for relative, expected in freeze["files"].items():
        path = FROZEN_CORE / relative
        if sha256(path) != expected:
            raise RuntimeError(f"frozen core file changed: {relative}")
    base_hashes = {record: sha256(BASE_BANK_ROOT / f"{record}.json.gz") for record in RECORDS}
    overlay_source = Path(__file__).resolve().parents[1] / "v4_36arm_ablation" / "shadow_overlay.py"
    association_source = Path(os.environ.get("AASM_V4_ASSOCIATION_SOURCE", ""))
    stage_runner_source = Path(os.environ.get("AASM_V4_STAGE_RUNNER_SOURCE", ""))
    if not association_source.is_file() or not stage_runner_source.is_file():
        raise RuntimeError(
            "set AASM_V4_ASSOCIATION_SOURCE and AASM_V4_STAGE_RUNNER_SOURCE "
            "to the frozen private replay modules"
        )
    module_root = Path(__file__).resolve().parent
    orchestration_hashes = {
        path.name: sha256(path)
        for path in sorted(module_root.glob("*.py"))
    }
    evaluator_source = module_root.parent / "v4_36arm_ablation" / "metrics.py"
    stage_generation_source = "\n".join((
        inspect.getsource(_load_record),
        inspect.getsource(replay_record_batch),
    ))
    cache_semantics = {
        "handlers": list(HANDLERS),
        "records": list(RECORDS),
        "source_seal_key_hash": seal["source_bank_key_hash"],
        "source_seal_sha256": sha256(SOURCE_SEAL),
        "source_manifest_sha256": sha256(source_manifest_path),
        "base_bank_hashes": base_hashes,
        "frozen_core_hash": freeze["source_config_test_hash"],
        "overlay_source_sha256": sha256(overlay_source),
        "association_rebuild_sha256": sha256(association_source),
        "stage_runner_sha256": sha256(stage_runner_source),
        "stage_generation_source_sha256": hashlib.sha256(
            stage_generation_source.encode("utf-8")
        ).hexdigest(),
        "frozen_evaluator_sha256": sha256(evaluator_source),
        "stage_contract": {"codes": STAGE_TO_CODE, "dtype": "uint8"},
    }
    identity = {
        "cache_semantics": cache_semantics,
        "orchestration_hashes": orchestration_hashes,
        "search_config_sha256": sha256(config_path),
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_semantics, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return cache_key, identity


class ReplayCoordinator:
    def __init__(
        self,
        cache: CacheDB,
        cache_key: str,
        limits: SearchLimits,
    ) -> None:
        self.cache = cache
        self.cache_key = cache_key
        self.limits = limits
        workers = max(1, min(limits.worker_count, len(RECORDS)))
        self.worker_count = workers
        self.executor = self._new_executor()

    def _new_executor(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=self.worker_count,
            mp_context=mp.get_context("spawn"),
        )

    def restart_executor(self) -> None:
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self.executor = self._new_executor()

    def restart_executor(self) -> None:
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self.executor = self._new_executor()

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> "ReplayCoordinator":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _register(self, masks: Iterable[int]) -> tuple[int, ...]:
        values = tuple(sorted(set(int(mask) for mask in masks)))
        known = self.cache.complete_masks()
        new_count = sum(mask not in known for mask in values)
        registered = self.cache.unique_arm_count()
        if registered + sum(
            self.cache.connection.execute("SELECT 1 FROM arms WHERE mask=?", (mask,)).fetchone() is None
            for mask in values
        ) > self.limits.max_unique_arms:
            raise RuntimeError("unique-arm budget would be exceeded")
        for mask in values:
            self.cache.register_arm(mask, self.cache_key)
        self.cache.connection.commit()
        return values

    def replay(self, masks: Iterable[int]) -> dict[str, Any]:
        values = self._register(masks)
        missing = self.cache.missing_records(values)
        if any(missing.values()):
            self.restart_executor()
        futures = {}
        batch_size = self.limits.replay_batch_size
        started = time.perf_counter()
        initially_complete = {mask for mask in values if self.cache.arm_complete(mask)}
        for record_id, record_masks in missing.items():
            for left in range(0, len(record_masks), batch_size):
                batch = tuple(record_masks[left:left + batch_size])
                if not batch:
                    continue
                self.cache.mark_running_batch(record_id, batch)
                future = self.executor.submit(replay_record_batch, record_id, batch)
                futures[future] = (record_id, batch)
        failures: list[dict[str, Any]] = []
        completed = 0
        for future in as_completed(futures):
            record_id, batch = futures[future]
            try:
                rows = future.result()
                returned = [(str(row.get("record_id")), int(row.get("mask", -1))) for row in rows]
                expected = [(record_id, int(mask)) for mask in batch]
                if sorted(returned) != sorted(expected) or len(returned) != len(set(returned)):
                    raise RuntimeError("worker batch result is not a one-to-one match")
            except Exception as exc:
                rows = [{
                    "mask": mask, "record_id": record_id, "status": "FAILED",
                    "stage_codes": None, "epoch_count": None, "stage_sha256": None,
                    "runtime_seconds": None, "effective_replacements": None,
                    "error": f"worker failure {type(exc).__name__}: {exc}",
                } for mask in batch]
            for row in rows:
                self.cache.store_record(row)
                completed += row["status"] == "PASS"
                if row["status"] != "PASS":
                    failures.append(row)
        incomplete = [mask for mask in values if not self.cache.arm_complete(mask)]
        return {
            "requested_masks": len(values),
            "completed_record_tasks": completed,
            "failures": failures,
            "incomplete_masks": incomplete,
            "wall_seconds": time.perf_counter() - started,
            "new_complete_masks": sum(
                mask not in initially_complete and self.cache.arm_complete(mask) for mask in values
            ),
        }

    def retry_incomplete_once(self, masks: Iterable[int]) -> dict[str, Any]:
        incomplete = [mask for mask in sorted(set(masks)) if not self.cache.arm_complete(mask)]
        if not incomplete:
            return {"requested_masks": 0, "failures": [], "incomplete_masks": []}
        return self.replay(incomplete)


def replay_full_trace(mask: int, record_id: str, output_root: Path) -> dict[str, Any]:
    """Recreate complete artifacts only for a selected finalist."""
    base, sources, aggregate, engine, index_packets = _load_record(record_id)
    arm = ArmSpec(mask)
    if mask == 0:
        integrated = deepcopy(base)
        manifest = {"requested_handlers": [], "effective_replacement_count": 0, "empty_swap": True}
    else:
        integrated, manifest = overlay_handlers(base, sources, arm.handlers)
    rows = aggregate(integrated, "enhanced")
    if mask & AR_MASK:
        _postprocess_unet_qc(rows, integrated, sources["AR"]["joint_valid_mask"])
    predictions = engine(rows, index_packets(integrated, rows))
    evidence = [asdict(row) for row in rows]
    traces = [{
        "epoch_index": row["epoch_index"], "rule_id": row["rule_id"],
        "stage": row["stage"], "evidence_event_ids": row["evidence_event_ids"],
        "trace": row["trace"], "temporal": row.get("temporal"),
    } for row in predictions]
    root = output_root / arm.arm_id
    paths = {
        "event_bank": root / "event_banks" / f"{record_id}.json.gz",
        "evidence": root / "evidence" / f"{record_id}.json.gz",
        "predictions": root / "predictions" / f"{record_id}.json.gz",
        "traces": root / "traces" / f"{record_id}.json.gz",
        "overlay": root / "overlay" / f"{record_id}.json",
    }
    write_gzip_json(paths["event_bank"], integrated)
    write_gzip_json(paths["evidence"], evidence)
    write_gzip_json(paths["predictions"], predictions)
    write_gzip_json(paths["traces"], traces)
    write_json(paths["overlay"], manifest)
    stages = np.fromiter((STAGE_TO_CODE[row["stage"]] for row in predictions), dtype=np.uint8)
    return {
        "mask": mask,
        "arm_id": arm.arm_id,
        "record_id": record_id,
        "stage_sha256": hashlib.sha256(stages.tobytes()).hexdigest(),
        "artifact_hashes": {name: sha256(path) for name, path in paths.items()},
    }
