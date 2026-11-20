from __future__ import annotations

import json
import sqlite3
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from .config import RuleConfig


class PredictionCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta(
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS configs(
                config_id TEXT PRIMARY KEY,
                config_json TEXT NOT NULL,
                config_sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS config_records(
                config_id TEXT NOT NULL,
                record_id TEXT NOT NULL,
                status TEXT NOT NULL,
                epoch_count INTEGER NOT NULL,
                stage_codes BLOB NOT NULL,
                rule_ids_zlib BLOB NOT NULL,
                output_sha256 TEXT NOT NULL,
                baseline_exact INTEGER NOT NULL,
                runtime_seconds REAL NOT NULL,
                diagnostics_json TEXT NOT NULL,
                PRIMARY KEY(config_id, record_id),
                FOREIGN KEY(config_id) REFERENCES configs(config_id)
            );
            CREATE TABLE IF NOT EXISTS fold_winners(
                fold_id TEXT PRIMARY KEY,
                heldout_record TEXT,
                config_id TEXT NOT NULL,
                status TEXT NOT NULL,
                selection_json TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def set_meta(self, key: str, value: Any) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO meta(key,value_json) VALUES(?,?)",
            (key, json.dumps(value, sort_keys=True)),
        )
        self.connection.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.connection.execute(
            "SELECT value_json FROM meta WHERE key=?", (key,)
        ).fetchone()
        return default if row is None else json.loads(row[0])

    def register_config(self, config: RuleConfig, config_sha256: str) -> None:
        encoded = json.dumps(config.to_dict(), sort_keys=True)
        existing = self.connection.execute(
            "SELECT config_json,config_sha256 FROM configs WHERE config_id=?",
            (config.config_id,),
        ).fetchone()
        if existing is not None and existing != (encoded, config_sha256):
            raise RuntimeError(f"config identity changed: {config.config_id}")
        self.connection.execute(
            "INSERT OR IGNORE INTO configs(config_id,config_json,config_sha256) VALUES(?,?,?)",
            (config.config_id, encoded, config_sha256),
        )
        self.connection.commit()

    def record_complete(self, config_id: str, record_id: str) -> bool:
        row = self.connection.execute(
            "SELECT status FROM config_records WHERE config_id=? AND record_id=?",
            (config_id, record_id),
        ).fetchone()
        return row is not None and row[0] == "PASS"

    def save_record(
        self,
        config_id: str,
        record_id: str,
        stage_codes: np.ndarray,
        rule_ids: list[str],
        output_sha256: str,
        baseline_exact: bool,
        runtime_seconds: float,
        diagnostics: dict[str, Any],
    ) -> None:
        if stage_codes.dtype != np.uint8 or len(stage_codes) != len(rule_ids):
            raise ValueError("invalid cached prediction payload")
        compressed_rules = zlib.compress(
            json.dumps(rule_ids, separators=(",", ":")).encode("utf-8"), level=9
        )
        with self.connection:
            self.connection.execute(
                """INSERT OR REPLACE INTO config_records(
                    config_id,record_id,status,epoch_count,stage_codes,rule_ids_zlib,
                    output_sha256,baseline_exact,runtime_seconds,diagnostics_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    config_id,
                    record_id,
                    "PASS",
                    len(stage_codes),
                    stage_codes.tobytes(),
                    compressed_rules,
                    output_sha256,
                    int(baseline_exact),
                    runtime_seconds,
                    json.dumps(diagnostics, sort_keys=True),
                ),
            )

    def stage_codes(self, config_id: str, record_id: str) -> np.ndarray:
        row = self.connection.execute(
            "SELECT stage_codes,epoch_count,status FROM config_records WHERE config_id=? AND record_id=?",
            (config_id, record_id),
        ).fetchone()
        if row is None or row[2] != "PASS":
            raise KeyError((config_id, record_id))
        values = np.frombuffer(row[0], dtype=np.uint8).copy()
        if len(values) != int(row[1]):
            raise RuntimeError("cache payload length mismatch")
        return values

    def rule_ids(self, config_id: str, record_id: str) -> list[str]:
        row = self.connection.execute(
            "SELECT rule_ids_zlib FROM config_records WHERE config_id=? AND record_id=?",
            (config_id, record_id),
        ).fetchone()
        if row is None:
            raise KeyError((config_id, record_id))
        return json.loads(zlib.decompress(row[0]).decode("utf-8"))

    def record_rows(self) -> list[tuple]:
        return self.connection.execute(
            """SELECT config_id,record_id,status,epoch_count,output_sha256,
                      baseline_exact,runtime_seconds,diagnostics_json
               FROM config_records ORDER BY config_id,record_id"""
        ).fetchall()

    def freeze_winner(
        self,
        fold_id: str,
        heldout_record: str | None,
        config_id: str,
        selection: dict[str, Any],
    ) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT OR REPLACE INTO fold_winners(
                    fold_id,heldout_record,config_id,status,selection_json
                ) VALUES(?,?,?,?,?)""",
                (
                    fold_id,
                    heldout_record,
                    config_id,
                    "WINNER_FROZEN",
                    json.dumps(selection, sort_keys=True),
                ),
            )

    def winners(self) -> list[tuple]:
        return self.connection.execute(
            "SELECT fold_id,heldout_record,config_id,status,selection_json FROM fold_winners ORDER BY fold_id"
        ).fetchall()
