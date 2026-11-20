from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
import time
from typing import Any

from experiments.v3_detector_integration.association import (
    rebuild_k_arousal,
    validate_associations,
)

from .config import ARMS, BASE_BANK_ROOT, EVENT_TYPE, FROZEN_CORE, HANDLERS, R1_ROOT, RECORDS
from .io_utils import (
    canonical_hash,
    read_gzip_json,
    sha256,
    write_csv,
    write_gzip_json,
    write_json,
)


VALID_SOURCE = {"AVAILABLE", "PARTIALLY_SUPPORTED", "NO_CANDIDATE"}


def _postprocess_unet_qc(rows, bank: dict[str, Any], valid: list[bool]) -> None:
    """Propagate frozen U-Net channel QC into dependent evidence fields."""

    from aasm_v2.types import Evidence

    events = bank["events"]["enhanced"]
    by_epoch: dict[int, list[dict[str, Any]]] = {}
    onsets: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        first = max(0, int(event["onset"] // 30))
        last = min(len(rows), int(-(-event["offset"] // 30)))
        for index in range(first, max(first + 1, last)):
            by_epoch.setdefault(index, []).append(event)
        onsets.setdefault(int(event["onset"] // 30), []).append(event)
    for index, row in enumerate(rows):
        if valid[index]:
            row.attributes["arousal_multichannel_qc"] = "AVAILABLE"
            continue
        for key in ("arousal", "n2_kcomplex", "arousal_then_sem"):
            row.events[key] = Evidence(
                None, [], detector_status="INVALID_SIGNAL", negative_validated=False
            )
        start = index * 30.0
        structural = [
            event
            for event in onsets.get(index, []) + onsets.get(index - 1, [])
            if max(0.0, start - 15.0) <= event["onset"] < start + 15.0
        ]
        spindles = [event for event in structural if event["event_type"] == "spindle"]
        first_spindles = [
            event for event in spindles if start <= event["onset"] < start + 15.0
        ]
        row.events["first_half_n2"] = Evidence(
            float(len(first_spindles)) if first_spindles else None,
            [event["event_id"] for event in first_spindles],
            detector_status="PARTIALLY_SUPPORTED" if first_spindles else "INVALID_SIGNAL",
            negative_validated=False,
        )
        row.attributes.update(
            arousal_multichannel_qc="INVALID_SIGNAL",
            last_arousal_end=None,
            n2_after_arousal_ids=[],
            n2_window_onsets=[
                [event["event_id"], event["onset"]] for event in structural
            ],
            n2_event_in_epoch=(
                True
                if any(
                    event["event_type"] == "spindle"
                    for event in by_epoch.get(index, [])
                )
                else None
            ),
        )


def _frozen_imports():
    source = str((FROZEN_CORE / "src").resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from aasm_v2.evidence import aggregate
    from aasm_temporal_r1.engine import engine
    from aasm_temporal_r1.timeline import index_packets

    return aggregate, engine, index_packets


def _type_hash(events: list[dict[str, Any]], event_type: str) -> str:
    return canonical_hash([event for event in events if event["event_type"] == event_type])


def overlay_handlers(
    base_bank: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    selected: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(selected) != len(set(selected)) or not set(selected) <= set(HANDLERS):
        raise ValueError("invalid replacement set")
    original_base_hash = canonical_hash(base_bank)
    original_layer_hash = canonical_hash(base_bank["events"]["original"])
    before_events = base_bank["events"]["enhanced"]
    integrated = deepcopy(base_bank)
    rows = integrated["events"]["enhanced"]
    replacements = []
    requested_types = {EVENT_TYPE[handler] for handler in selected}
    rows = [event for event in rows if event["event_type"] not in requested_types]
    for handler in selected:
        source = sources[handler]
        status = source["status"]
        applied = status in VALID_SOURCE
        if applied:
            rows.extend(deepcopy(source["events"]))
            availability = source["overlay_availability"]
            reason = "sealed neural source inserted" if source["events"] else "valid inference with no candidate"
        else:
            availability = "UNAVAILABLE" if status in {"UNAVAILABLE", "DETECTOR_FAILURE"} else status
            reason = "fail-closed: frozen candidates removed; neural source unusable"
        integrated["availability"]["enhanced"][EVENT_TYPE[handler]] = availability
        replacements.append({
            "requested_handler": handler,
            "event_type": EVENT_TYPE[handler],
            "actual_source": source["model"] if applied else None,
            "replacement_applied": applied,
            "availability_status": availability,
            "source_status": status,
            "reason": reason,
            "valid_seconds": source["valid_signal_seconds"],
            "inserted_event_count": len(source["events"]) if applied else 0,
            "source_bank_sha256": source["_source_bank_sha256"],
        })
    rows.sort(key=lambda event: (float(event["onset"]), float(event["offset"]),
                                 event["event_type"], event["event_id"]))
    identifiers = [event["event_id"] for event in rows]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("overlay created duplicate event IDs")
    integrated["events"]["enhanced"] = rows
    if canonical_hash(integrated["events"]["original"]) != original_layer_hash:
        raise RuntimeError("original event layer changed")
    for handler in HANDLERS:
        event_type = EVENT_TYPE[handler]
        if handler not in selected and _type_hash(rows, event_type) != _type_hash(before_events, event_type):
            raise RuntimeError(f"unselected event type changed before dependency rebuild: {event_type}")
    association = {"rebuild_required": False, "changed_kcomplex_count": 0}
    if "E02" in selected or "AR" in selected:
        integrated, association = rebuild_k_arousal(integrated)
        association["rebuild_required"] = True
        errors = validate_associations(integrated)
        if errors:
            raise RuntimeError(f"association validation failed: {errors[:3]}")
    if canonical_hash(base_bank) != original_base_hash:
        raise RuntimeError("source bank was mutated")
    manifest = {
        "requested_handlers": list(selected),
        "requested_replacement_count": len(selected),
        "effective_replacement_count": sum(row["replacement_applied"] for row in replacements),
        "partial_effectiveness": any(not row["replacement_applied"] for row in replacements),
        "replacements": replacements,
        "base_bank_hash": original_base_hash,
        "original_layer_unchanged": True,
        "source_bank_unchanged": canonical_hash(base_bank) == original_base_hash,
        "association": association,
    }
    return integrated, manifest


def _load_sources(run_root: Path, record: str) -> dict[str, dict[str, Any]]:
    sources = {}
    for handler in HANDLERS:
        path = run_root / "source_banks" / handler / f"{record}.json.gz"
        payload = read_gzip_json(path)
        payload["_source_bank_sha256"] = sha256(path)
        sources[handler] = payload
    return sources


def _write_arm_outputs(
    run_root: Path,
    arm: str,
    record: str,
    bank: dict[str, Any],
    rows: list[Any],
    predictions: list[dict[str, Any]],
    overlay_manifest: dict[str, Any],
) -> dict[str, Any]:
    evidence = [asdict(row) for row in rows]
    traces = [{
        "epoch_index": row["epoch_index"], "rule_id": row["rule_id"],
        "stage": row["stage"], "evidence_event_ids": row["evidence_event_ids"],
        "trace": row["trace"], "temporal": row.get("temporal"),
    } for row in predictions]
    bank_path = run_root / "event_banks" / arm / f"{record}.json.gz"
    evidence_path = run_root / "evidence" / arm / f"{record}.json.gz"
    prediction_path = run_root / "predictions" / arm / f"{record}.json.gz"
    trace_path = run_root / "traces" / arm / f"{record}.json.gz"
    overlay_path = run_root / "overlay" / "manifests" / arm / f"{record}.json"
    write_gzip_json(bank_path, bank)
    write_gzip_json(evidence_path, evidence)
    write_gzip_json(prediction_path, predictions)
    write_gzip_json(trace_path, traces)
    write_json(overlay_path, overlay_manifest)
    return {
        "arm_id": arm, "record_id": record, "status": "PASS",
        "requested_replacements": len(ARMS[arm]),
        "effective_replacements": overlay_manifest["effective_replacement_count"],
        "partial_effectiveness": overlay_manifest["partial_effectiveness"],
        "epochs": len(predictions), "event_count": len(bank["events"]["enhanced"]),
        "event_bank_sha256": sha256(bank_path), "evidence_sha256": sha256(evidence_path),
        "prediction_sha256": sha256(prediction_path), "trace_sha256": sha256(trace_path),
    }


def execute_arms(run_root: Path) -> dict[str, Any]:
    source_seal = read_gzip_json if False else None  # keep label-free module import explicit
    seal_path = run_root / "neural_inference" / "prediction_seal.json"
    from .io_utils import read_json

    seal = read_json(seal_path)
    if seal.get("status") != "SEALED" or seal.get("source_bank_count") != 150:
        raise RuntimeError("neural source banks are not sealed")
    aggregate, engine, index_packets = _frozen_imports()
    started = time.perf_counter()
    execution_rows = []
    baseline_rows = []
    for arm, selected in ARMS.items():
        for record in RECORDS:
            base_path = BASE_BANK_ROOT / f"{record}.json.gz"
            base = read_gzip_json(base_path)
            if arm == "B000":
                integrated = base
                manifest = {
                    "requested_handlers": [], "requested_replacement_count": 0,
                    "effective_replacement_count": 0, "partial_effectiveness": False,
                    "replacements": [], "base_bank_hash": canonical_hash(base),
                    "original_layer_unchanged": True, "source_bank_unchanged": True,
                    "association": {"rebuild_required": False, "changed_kcomplex_count": 0},
                    "empty_swap": True,
                }
            else:
                integrated, manifest = overlay_handlers(base, _load_sources(run_root, record), selected)
            rows = aggregate(integrated, "enhanced")
            if "AR" in selected:
                arousal = _load_sources(run_root, record)["AR"]
                _postprocess_unet_qc(rows, integrated, arousal["joint_valid_mask"])
            predictions = engine(rows, index_packets(integrated, rows))
            result = _write_arm_outputs(run_root, arm, record, integrated, rows, predictions, manifest)
            execution_rows.append(result)
            if arm == "B000":
                expected_evidence = read_gzip_json(R1_ROOT / "records" / record / "evidence.json.gz")
                expected_prediction = read_gzip_json(R1_ROOT / "records" / record / "predictions.json.gz")["R1"]
                actual_evidence = [asdict(row) for row in rows]
                baseline_rows.append({
                    "record_id": record,
                    "bank_exact": canonical_hash(integrated) == canonical_hash(base),
                    "evidence_exact": canonical_hash(actual_evidence) == canonical_hash(expected_evidence),
                    "prediction_exact": canonical_hash(predictions) == canonical_hash(expected_prediction),
                    "evidence_hash": canonical_hash(actual_evidence),
                    "prediction_hash": canonical_hash(predictions),
                })
    if len(execution_rows) != 360:
        raise AssertionError(f"expected 360 night-arm executions, found {len(execution_rows)}")
    if not all(row["bank_exact"] and row["evidence_exact"] and row["prediction_exact"] for row in baseline_rows):
        raise RuntimeError("B000 exact regression failed")
    write_csv(run_root / "arms" / "arm_execution_10night.csv", execution_rows)
    arm_status = []
    for arm in ARMS:
        rows = [row for row in execution_rows if row["arm_id"] == arm]
        arm_status.append({
            "arm_id": arm, "status": "PASS" if len(rows) == 10 and all(row["status"] == "PASS" for row in rows) else "FAIL",
            "night_count": len(rows), "requested_handlers": ";".join(ARMS[arm]),
            "partial_nights": sum(row["partial_effectiveness"] for row in rows),
        })
    write_csv(run_root / "arms" / "arm_execution_status.csv", arm_status)
    write_json(run_root / "arms" / "baseline_regression.json", {
        "rows": baseline_rows, "all_exact": all(row["bank_exact"] and row["evidence_exact"] and row["prediction_exact"] for row in baseline_rows),
    })
    prediction_items = [(row["arm_id"], row["record_id"], row["prediction_sha256"]) for row in execution_rows]
    prediction_seal = {
        "status": "SEALED", "registered_arms": len(ARMS), "records": len(RECORDS),
        "night_arm_count": len(execution_rows), "prediction_key_hash": canonical_hash(prediction_items),
        "label_access_started": False, "runtime_seconds": time.perf_counter() - started,
    }
    write_json(run_root / "arms" / "prediction_seal.json", prediction_seal)
    return prediction_seal
