from __future__ import annotations

import json
import hashlib
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import ArmSpec, FULL_MASK, RECORDS


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS arms (
    mask INTEGER PRIMARY KEY,
    arm_id TEXT NOT NULL UNIQUE,
    handlers TEXT NOT NULL,
    cardinality INTEGER NOT NULL,
    status TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    completed_records INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS arm_records (
    mask INTEGER NOT NULL,
    record_id TEXT NOT NULL,
    status TEXT NOT NULL,
    stage_codes BLOB,
    epoch_count INTEGER,
    stage_sha256 TEXT,
    runtime_seconds REAL,
    effective_replacements INTEGER,
    error TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(mask, record_id)
);
CREATE TABLE IF NOT EXISTS search_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fold_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    round_index INTEGER,
    mask INTEGER,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS fold_winners (
    fold_id TEXT PRIMARY KEY,
    heldout_record TEXT,
    winner_mask INTEGER NOT NULL,
    status TEXT NOT NULL,
    selection_payload TEXT NOT NULL,
    frozen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    heldout_scored_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_arm_records_status ON arm_records(status);
CREATE INDEX IF NOT EXISTS idx_search_events_fold ON search_events(fold_id, event_id);
"""


class CacheDB:
    def __init__(self, path: Path, cache_key: str) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=120.0)
        self.connection.executescript(SCHEMA)
        encoded_key = json.dumps(cache_key)
        self.connection.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('cache_key',?)", (encoded_key,))
        actual = json.loads(self.connection.execute("SELECT value FROM meta WHERE key='cache_key'").fetchone()[0])
        if actual != cache_key:
            raise RuntimeError("cache identity differs from frozen inputs/config")
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def set_meta(self, key: str, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
        self.connection.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, payload),
        )
        self.connection.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row[0])

    def register_arm(self, mask: int, cache_key: str) -> None:
        if not 0 <= int(mask) <= FULL_MASK:
            raise ValueError(f"invalid mask: {mask}")
        arm = ArmSpec(mask)
        self.connection.execute(
            """INSERT OR IGNORE INTO arms(mask,arm_id,handlers,cardinality,status,cache_key)
               VALUES(?,?,?,?,?,?)""",
            (mask, arm.arm_id, ";".join(arm.handlers), arm.cardinality, "PENDING", cache_key),
        )

    def mark_running(self, mask: int, record_id: str) -> None:
        if record_id not in RECORDS or not 0 <= int(mask) <= FULL_MASK:
            raise ValueError(f"invalid arm-record key: {mask}/{record_id}")
        self.connection.execute(
            """INSERT INTO arm_records(mask,record_id,status)
               VALUES(?,?,?) ON CONFLICT(mask,record_id) DO UPDATE SET status='RUNNING',error=NULL,
               updated_at=CURRENT_TIMESTAMP""",
            (mask, record_id, "RUNNING"),
        )
        self.connection.commit()

    def mark_running_batch(self, record_id: str, masks: Iterable[int]) -> None:
        values = tuple(int(mask) for mask in masks)
        if record_id not in RECORDS or any(not 0 <= mask <= FULL_MASK for mask in values):
            raise ValueError(f"invalid arm-record batch: {record_id}")
        rows = [(mask, record_id, "RUNNING") for mask in values]
        self.connection.executemany(
            """INSERT INTO arm_records(mask,record_id,status)
               VALUES(?,?,?) ON CONFLICT(mask,record_id) DO UPDATE SET status='RUNNING',error=NULL,
               updated_at=CURRENT_TIMESTAMP""",
            rows,
        )
        self.connection.commit()

    def store_record(self, result: dict[str, Any]) -> None:
        mask = int(result["mask"])
        record_id = str(result["record_id"])
        if not 0 <= mask <= FULL_MASK or record_id not in RECORDS:
            raise ValueError(f"invalid arm-record result: {mask}/{record_id}")
        stages = result.get("stage_codes")
        if result["status"] == "PASS":
            values = np.asarray(stages)
            if values.dtype != np.uint8 or values.ndim != 1 or not len(values) or np.any(values > 5):
                raise ValueError(f"invalid PASS stage codes: {mask}/{record_id}")
            digest = hashlib.sha256(values.tobytes()).hexdigest()
            if int(result.get("epoch_count", -1)) != len(values) or result.get("stage_sha256") != digest:
                raise ValueError(f"PASS stage metadata mismatch: {mask}/{record_id}")
            blob = sqlite3.Binary(values.tobytes())
        else:
            blob = None
        self.connection.execute(
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
            (
                mask, record_id, result["status"], blob,
                result.get("epoch_count"), result.get("stage_sha256"), result.get("runtime_seconds"),
                result.get("effective_replacements"), result.get("error"),
            ),
        )
        self._refresh_arm(mask)
        self.connection.commit()

    def _refresh_arm(self, mask: int) -> None:
        rows = self.connection.execute(
            "SELECT record_id,status FROM arm_records WHERE mask=?", (mask,)
        ).fetchall()
        mapping = {record_id: state for record_id, state in rows if record_id in RECORDS}
        completed = sum(mapping.get(record) == "PASS" for record in RECORDS)
        status = "PASS" if completed == len(RECORDS) else (
            "FAILED" if any(mapping.get(record) == "FAILED" for record in RECORDS) else "RUNNING"
        )
        self.connection.execute(
            "UPDATE arms SET status=?,completed_records=?,updated_at=CURRENT_TIMESTAMP WHERE mask=?",
            (status, completed, mask),
        )

    def reset_interrupted(self) -> int:
        cursor = self.connection.execute(
            "UPDATE arm_records SET status='PENDING',error='recovered interrupted RUNNING state' WHERE status='RUNNING'"
        )
        self.connection.execute("UPDATE arms SET status='PENDING' WHERE status='RUNNING'")
        self.connection.commit()
        return int(cursor.rowcount)

    def missing_records(self, masks: Iterable[int]) -> dict[str, list[int]]:
        output = {record: [] for record in RECORDS}
        for mask in sorted(set(int(value) for value in masks)):
            passed = {
                row[0] for row in self.connection.execute(
                    "SELECT record_id FROM arm_records WHERE mask=? AND status='PASS'", (mask,)
                ).fetchall()
            }
            for record in RECORDS:
                if record not in passed:
                    output[record].append(mask)
        return output

    def arm_complete(self, mask: int) -> bool:
        row = self.connection.execute("SELECT status FROM arms WHERE mask=?", (mask,)).fetchone()
        return row is not None and row[0] == "PASS"

    def complete_masks(self) -> set[int]:
        return {int(row[0]) for row in self.connection.execute("SELECT mask FROM arms WHERE status='PASS'")}

    def stage_codes(self, mask: int, record_id: str) -> np.ndarray:
        row = self.connection.execute(
            "SELECT stage_codes,epoch_count,status FROM arm_records WHERE mask=? AND record_id=?",
            (mask, record_id),
        ).fetchone()
        if row is None or row[2] != "PASS" or row[0] is None:
            raise KeyError(f"missing sealed stage codes: {ArmSpec(mask).arm_id}/{record_id}")
        values = np.frombuffer(row[0], dtype=np.uint8).copy()
        if len(values) != int(row[1]):
            raise RuntimeError("stage-code length mismatch")
        digest = hashlib.sha256(values.tobytes()).hexdigest()
        expected = self.connection.execute(
            "SELECT stage_sha256 FROM arm_records WHERE mask=? AND record_id=?",
            (mask, record_id),
        ).fetchone()[0]
        if digest != expected or np.any(values > 5):
            raise RuntimeError("stage-code integrity mismatch")
        return values

    def record_hash(self, mask: int, record_id: str) -> str:
        row = self.connection.execute(
            "SELECT stage_sha256 FROM arm_records WHERE mask=? AND record_id=? AND status='PASS'",
            (mask, record_id),
        ).fetchone()
        if row is None:
            raise KeyError((mask, record_id))
        return str(row[0])

    def log_event(
        self,
        fold_id: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        round_index: int | None = None,
        mask: int | None = None,
    ) -> None:
        self.connection.execute(
            "INSERT INTO search_events(fold_id,event_type,round_index,mask,payload) VALUES(?,?,?,?,?)",
            (fold_id, event_type, round_index, mask, json.dumps(payload, ensure_ascii=False, sort_keys=True)),
        )
        self.connection.commit()

    def freeze_winner(
        self,
        fold_id: str,
        heldout_record: str | None,
        winner_mask: int,
        payload: dict[str, Any],
    ) -> None:
        existing = self.winner(fold_id)
        if existing is not None:
            if existing[0] != winner_mask or existing[1] != heldout_record:
                raise RuntimeError(f"winner already frozen with different mask: {fold_id}/{existing}")
            if existing[2] == "HELDOUT_SCORED":
                raise RuntimeError(f"heldout already scored; winner is immutable: {fold_id}")
            if existing[2] == "WINNER_FROZEN":
                return
        self.connection.execute(
            """INSERT INTO fold_winners(fold_id,heldout_record,winner_mask,status,selection_payload)
               VALUES(?,?,?,?,?)
               ON CONFLICT(fold_id) DO NOTHING""",
            (fold_id, heldout_record, winner_mask, "WINNER_FROZEN", json.dumps(payload, ensure_ascii=False, sort_keys=True)),
        )
        self.connection.commit()

    def winner(self, fold_id: str) -> tuple[int, str | None, str] | None:
        row = self.connection.execute(
            "SELECT winner_mask,heldout_record,status FROM fold_winners WHERE fold_id=?", (fold_id,)
        ).fetchone()
        return None if row is None else (int(row[0]), row[1], str(row[2]))

    def mark_heldout_scored(self, fold_id: str) -> None:
        existing = self.winner(fold_id)
        if existing is not None and existing[2] == "HELDOUT_SCORED":
            return
        cursor = self.connection.execute(
            """UPDATE fold_winners SET status='HELDOUT_SCORED',heldout_scored_at=CURRENT_TIMESTAMP
               WHERE fold_id=? AND status='WINNER_FROZEN'""",
            (fold_id,),
        )
        if cursor.rowcount != 1:
            self.connection.rollback()
            raise RuntimeError(f"heldout score transition requires WINNER_FROZEN: {fold_id}")
        self.connection.commit()

    def all_winners(self) -> list[dict[str, Any]]:
        columns = ("fold_id", "heldout_record", "winner_mask", "status", "selection_payload")
        rows = self.connection.execute(
            """SELECT fold_id,heldout_record,winner_mask,status,selection_payload
               FROM fold_winners ORDER BY fold_id"""
        ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            item = dict(zip(columns, row))
            item["winner_mask"] = int(item["winner_mask"])
            item["selection_payload"] = json.loads(str(item["selection_payload"]))
            output.append(item)
        return output

    def unlock_reporting(self) -> None:
        outer = [row for row in self.all_winners() if row["fold_id"].startswith("OUTER_")]
        if len(outer) != len(RECORDS) or any(row["status"] != "HELDOUT_SCORED" for row in outer):
            raise RuntimeError("reporting requires all ten outer heldouts to be scored after freeze")
        self.set_meta("report_unlocked", True)

    def unique_arm_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM arms").fetchone()[0])

    def failed_rows(self) -> list[dict[str, Any]]:
        columns = ("mask", "record_id", "error")
        return [dict(zip(columns, row)) for row in self.connection.execute(
            "SELECT mask,record_id,error FROM arm_records WHERE status='FAILED' ORDER BY mask,record_id"
        )]
