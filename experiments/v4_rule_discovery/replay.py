from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .cache import PredictionCache
from .config import RECORDS, STAGE_TO_CODE, RuleConfig
from .io import (
    canonical_bytes,
    load_bank,
    load_rows,
    load_sealed_predictions,
    sha256_bytes,
    write_json_gz,
)
from .shadow_engine import build_packets, engine


def replay_record(config: RuleConfig, record: str) -> dict[str, Any]:
    bank = load_bank(record)
    rows, _ = load_rows(record)
    packets = build_packets(bank, rows)
    started = time.perf_counter()
    decisions, diagnostics = engine(rows, packets, config)
    runtime = time.perf_counter() - started
    sealed = load_sealed_predictions(record) if config.config_id == "C000" else None
    baseline_exact = sealed is None or canonical_bytes(decisions) == canonical_bytes(sealed)
    if sealed is not None and not baseline_exact:
        mismatch = next(
            (
                index
                for index, (left, right) in enumerate(zip(decisions, sealed))
                if canonical_bytes(left) != canonical_bytes(right)
            ),
            None,
        )
        raise RuntimeError(f"C000 exact regression failed: {record}/epoch={mismatch}")
    stage_codes = np.asarray(
        [STAGE_TO_CODE[decision["stage"]] for decision in decisions], dtype=np.uint8
    )
    rule_ids = [decision["rule_id"] for decision in decisions]
    invalidations = sum(bool(row.get("invalidated_by", [])) for row in diagnostics)
    rebuilds = sum(bool(row.get("rebuilt_by", [])) for row in diagnostics)
    backward = sum(bool(row.get("rem_backward_relabelled", False)) for row in diagnostics)
    return {
        "decisions": decisions,
        "anchor_diagnostics": diagnostics,
        "stage_codes": stage_codes,
        "rule_ids": rule_ids,
        "output_sha256": sha256_bytes(canonical_bytes(decisions)),
        "baseline_exact": baseline_exact,
        "runtime_seconds": runtime,
        "summary": {
            "epochs": len(decisions),
            "stage_counts": dict(Counter(decision["stage"] for decision in decisions)),
            "rule_counts": dict(Counter(rule_ids)),
            "rem_anchor_invalidations": invalidations,
            "rem_anchor_rebuilds": rebuilds,
            "rem_backward_relabels": backward,
        },
    }


def predict_all(
    cache: PredictionCache,
    configs: tuple[RuleConfig, ...],
    config_hashes: dict[str, str],
    budget_deadline: float,
) -> list[dict[str, Any]]:
    manifest = []
    for config in configs:
        cache.register_config(config, config_hashes[config.config_id])
        for record in RECORDS:
            if time.monotonic() >= budget_deadline:
                raise TimeoutError("wall-time budget reached during label-blind replay")
            if cache.record_complete(config.config_id, record):
                continue
            result = replay_record(config, record)
            cache.save_record(
                config.config_id,
                record,
                result["stage_codes"],
                result["rule_ids"],
                result["output_sha256"],
                result["baseline_exact"],
                result["runtime_seconds"],
                result["summary"],
            )
        manifest.append(
            {
                "config_id": config.config_id,
                "records_complete": sum(
                    cache.record_complete(config.config_id, record) for record in RECORDS
                ),
            }
        )
    return manifest


def export_full_trace(config: RuleConfig, root: Path) -> list[dict[str, Any]]:
    manifest = []
    for record in RECORDS:
        result = replay_record(config, record)
        prediction_path = root / config.config_id / "predictions" / f"{record}.json.gz"
        diagnostic_path = root / config.config_id / "diagnostics" / f"{record}.json.gz"
        write_json_gz(prediction_path, result["decisions"])
        write_json_gz(diagnostic_path, result["anchor_diagnostics"])
        manifest.append(
            {
                "config_id": config.config_id,
                "record_id": record,
                "prediction_path": str(prediction_path),
                "output_sha256": result["output_sha256"],
                "baseline_exact": result["baseline_exact"],
            }
        )
    return manifest
