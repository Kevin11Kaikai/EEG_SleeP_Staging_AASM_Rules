from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import shutil
import sqlite3
import time
from typing import Any, Iterable

import numpy as np

from experiments.v4_subset_search.config import ArmSpec, RECORDS
from experiments.v4_subset_search.replay import replay_record_batch
from experiments.v4_subset_search.storage import CacheDB

from .config import ExtendedLimits


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clone_label_blind_cache(source: Path, destination: Path, expected_key: str) -> dict[str, Any]:
    """Copy only reusable label-blind predictions into a new experiment database."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        connection = sqlite3.connect(destination, timeout=120)
        repaired_unsanitized_copy = False
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(f"existing cache integrity failure: {integrity}")
            key_row = connection.execute(
                "SELECT value FROM meta WHERE key='cache_key'"
            ).fetchone()
            if key_row is None or json.loads(key_row[0]) != expected_key:
                raise RuntimeError("existing cache key differs from frozen replay semantics")
            extended_row = connection.execute(
                "SELECT value FROM meta WHERE key='extended_limits'"
            ).fetchone()
            search_count = int(
                connection.execute("SELECT COUNT(*) FROM search_events").fetchone()[0]
            )
            winner_count = int(
                connection.execute("SELECT COUNT(*) FROM fold_winners").fetchone()[0]
            )
            # A crash after the byte-for-byte copy but before sanitization leaves the
            # old run's search history in place and has no extended-run marker yet.
            if extended_row is None and (search_count or winner_count):
                connection.execute("DELETE FROM search_events")
                connection.execute("DELETE FROM fold_winners")
                connection.execute("DELETE FROM meta WHERE key <> 'cache_key'")
                connection.execute(
                    "INSERT INTO meta(key,value) VALUES('report_unlocked','false')"
                )
                connection.commit()
                repaired_unsanitized_copy = True
                search_count = winner_count = 0
        finally:
            connection.close()
        return {
            "status": "EXISTING_DESTINATION_VERIFIED",
            "source_sha256": sha256(source),
            "destination_sha256": sha256(destination),
            "repaired_unsanitized_copy": repaired_unsanitized_copy,
            "search_events": search_count,
            "fold_winners": winner_count,
        }
    source_hash = sha256(source)
    temporary = destination.with_suffix(".copying")
    shutil.copy2(source, temporary)
    if sha256(temporary) != source_hash:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("cache copy hash mismatch")
    temporary.replace(destination)
    connection = sqlite3.connect(destination, timeout=120)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"copied cache integrity failure: {integrity}")
        actual_key = json.loads(
            connection.execute("SELECT value FROM meta WHERE key='cache_key'").fetchone()[0]
        )
        if actual_key != expected_key:
            raise RuntimeError("copied cache key differs from frozen replay semantics")
        seed_arms = int(connection.execute("SELECT COUNT(*) FROM arms WHERE status='PASS'").fetchone()[0])
        seed_rows = int(
            connection.execute("SELECT COUNT(*) FROM arm_records WHERE status='PASS'").fetchone()[0]
        )
        connection.execute("DELETE FROM search_events")
        connection.execute("DELETE FROM fold_winners")
        connection.execute("DELETE FROM meta WHERE key <> 'cache_key'")
        connection.execute(
            "INSERT INTO meta(key,value) VALUES('report_unlocked','false')"
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.commit()
    finally:
        connection.close()
    return {
        "status": "CLONED_LABEL_BLIND_ONLY",
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "source_sha256": source_hash,
        "destination_sha256_after_sanitization": sha256(destination),
        "seed_complete_masks": seed_arms,
        "seed_pass_record_rows": seed_rows,
        "search_events_retained": 0,
        "fold_winners_retained": 0,
        "report_unlocked": False,
    }


class FastReplayCoordinator:
    """Fourteen-worker, batched-SQL exact replay with resumable record-level checkpoints."""

    def __init__(self, cache: CacheDB, cache_key: str, limits: ExtendedLimits) -> None:
        self.cache = cache
        self.cache_key = cache_key
        self.limits = limits
        self.chunk_count = 0
        self.executor = self._new_executor()

    def _new_executor(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=self.limits.worker_count,
            mp_context=mp.get_context("spawn"),
        )

    def restart(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=False)
        self.executor = self._new_executor()

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> "FastReplayCoordinator":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _register(self, masks: tuple[int, ...]) -> None:
        rows = []
        for mask in masks:
            arm = ArmSpec(mask)
            rows.append(
                (mask, arm.arm_id, ";".join(arm.handlers), arm.cardinality, "PENDING", self.cache_key)
            )
        self.cache.connection.executemany(
            """INSERT OR IGNORE INTO arms(mask,arm_id,handlers,cardinality,status,cache_key)
               VALUES(?,?,?,?,?,?)""",
            rows,
        )
        self.cache.connection.commit()

    def _mark_running(self, tasks: list[tuple[str, tuple[int, ...]]]) -> None:
        rows = [
            (mask, record, "RUNNING")
            for record, masks in tasks
            for mask in masks
        ]
        self.cache.connection.executemany(
            """INSERT INTO arm_records(mask,record_id,status)
               VALUES(?,?,?) ON CONFLICT(mask,record_id) DO UPDATE SET
               status='RUNNING',error=NULL,updated_at=CURRENT_TIMESTAMP""",
            rows,
        )
        self.cache.connection.commit()

    def _store_rows(self, rows: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
        encoded = []
        failures: list[dict[str, Any]] = []
        passed = 0
        for row in rows:
            mask = int(row["mask"])
            record = str(row["record_id"])
            if record not in RECORDS:
                raise ValueError(f"worker returned forbidden record: {record}")
            blob = None
            if row["status"] == "PASS":
                values = np.asarray(row["stage_codes"])
                digest = hashlib.sha256(values.tobytes()).hexdigest()
                if (
                    values.dtype != np.uint8
                    or values.ndim != 1
                    or np.any(values > 5)
                    or int(row["epoch_count"]) != len(values)
                    or row["stage_sha256"] != digest
                ):
                    raise ValueError(f"invalid worker stage payload: {mask}/{record}")
                blob = sqlite3.Binary(values.tobytes())
                passed += 1
            else:
                failures.append({
                    "mask": mask,
                    "record_id": record,
                    "error": row.get("error"),
                })
            encoded.append((
                mask,
                record,
                row["status"],
                blob,
                row.get("epoch_count"),
                row.get("stage_sha256"),
                row.get("runtime_seconds"),
                row.get("effective_replacements"),
                row.get("error"),
            ))
        self.cache.connection.executemany(
            """INSERT INTO arm_records(
                   mask,record_id,status,stage_codes,epoch_count,stage_sha256,runtime_seconds,
                   effective_replacements,error)
               VALUES(?,?,?,?,?,?,?,?,?)
               ON CONFLICT(mask,record_id) DO UPDATE SET
                   status=excluded.status,stage_codes=excluded.stage_codes,
                   epoch_count=excluded.epoch_count,stage_sha256=excluded.stage_sha256,
                   runtime_seconds=excluded.runtime_seconds,
                   effective_replacements=excluded.effective_replacements,error=excluded.error,
                   updated_at=CURRENT_TIMESTAMP""",
            encoded,
        )
        self.cache.connection.commit()
        return passed, failures

    def _refresh(self, masks: tuple[int, ...]) -> None:
        for left in range(0, len(masks), 900):
            block = masks[left:left + 900]
            placeholders = ",".join("?" for _ in block)
            rows = self.cache.connection.execute(
                f"""SELECT mask,
                            SUM(CASE WHEN status='PASS' THEN 1 ELSE 0 END),
                            SUM(CASE WHEN status='FAILED' THEN 1 ELSE 0 END)
                     FROM arm_records WHERE mask IN ({placeholders}) GROUP BY mask""",
                block,
            ).fetchall()
            updates = []
            for mask, pass_count, fail_count in rows:
                status = "PASS" if int(pass_count) == len(RECORDS) else (
                    "FAILED" if int(fail_count) else "RUNNING"
                )
                updates.append((status, int(pass_count), int(mask)))
            self.cache.connection.executemany(
                """UPDATE arms SET status=?,completed_records=?,updated_at=CURRENT_TIMESTAMP
                   WHERE mask=?""",
                updates,
            )
        self.cache.connection.commit()

    def replay(self, masks: Iterable[int]) -> dict[str, Any]:
        values = tuple(sorted(set(int(mask) for mask in masks)))
        if not values:
            return {
                "requested_masks": 0,
                "new_complete_masks": 0,
                "completed_record_rows": 0,
                "failures": [],
                "incomplete_masks": [],
                "wall_seconds": 0.0,
            }
        self._register(values)
        initially_complete = {mask for mask in values if self.cache.arm_complete(mask)}
        missing = self.cache.missing_records(values)
        tasks: list[tuple[str, tuple[int, ...]]] = []
        for record in RECORDS:
            record_masks = missing[record]
            if not record_masks:
                continue
            shards = min(self.limits.shards_per_record, len(record_masks))
            shard_size = math.ceil(len(record_masks) / shards)
            for left in range(0, len(record_masks), shard_size):
                tasks.append((record, tuple(record_masks[left:left + shard_size])))
        self._mark_running(tasks)
        started = time.perf_counter()
        futures = {
            self.executor.submit(replay_record_batch, record, batch): (record, batch)
            for record, batch in tasks
        }
        failures: list[dict[str, Any]] = []
        completed_rows = 0
        worker_runtime = 0.0
        for future in as_completed(futures):
            record, batch = futures[future]
            try:
                rows = future.result()
                returned = {(str(row.get("record_id")), int(row.get("mask", -1))) for row in rows}
                expected = {(record, int(mask)) for mask in batch}
                if returned != expected or len(rows) != len(expected):
                    raise RuntimeError("worker result did not match submitted record-mask keys")
            except Exception as exc:
                rows = [{
                    "mask": mask,
                    "record_id": record,
                    "status": "FAILED",
                    "stage_codes": None,
                    "epoch_count": None,
                    "stage_sha256": None,
                    "runtime_seconds": None,
                    "effective_replacements": None,
                    "error": f"worker failure {type(exc).__name__}: {exc}",
                } for mask in batch]
            passed, current_failures = self._store_rows(rows)
            completed_rows += passed
            failures.extend(current_failures)
            worker_runtime += sum(float(row.get("runtime_seconds") or 0.0) for row in rows)
        self._refresh(values)
        incomplete = [mask for mask in values if not self.cache.arm_complete(mask)]
        self.chunk_count += 1
        if self.chunk_count % self.limits.restart_every_chunks == 0:
            self.restart()
        return {
            "requested_masks": len(values),
            "new_complete_masks": sum(
                mask not in initially_complete and self.cache.arm_complete(mask) for mask in values
            ),
            "completed_record_rows": completed_rows,
            "failures": failures,
            "incomplete_masks": incomplete,
            "wall_seconds": time.perf_counter() - started,
            "worker_runtime_seconds": worker_runtime,
            "task_count": len(tasks),
        }

    def retry_once(self, masks: Iterable[int]) -> dict[str, Any]:
        incomplete = tuple(mask for mask in sorted(set(masks)) if not self.cache.arm_complete(mask))
        if not incomplete:
            return {"requested_masks": 0, "incomplete_masks": [], "failures": []}
        self.restart()
        return self.replay(incomplete)
