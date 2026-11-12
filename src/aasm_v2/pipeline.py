"""Predict a signal EDF. This module never imports an annotation or evaluator."""
from dataclasses import asdict
from pathlib import Path
import gzip
import json
import os
import time
import psutil
from .events import detect
from .evidence import aggregate
from .rules import original_engine, enhanced_engine
from .provenance import file_hash, object_hash, write_json, source_manifest


def gzip_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf8") as out:
        json.dump(value, out, allow_nan=False)
    tmp.replace(path)


def predict(path: Path, destination: Path, cfg: dict, root: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    source = source_manifest(root)
    identity = {"psg": file_hash(path), "configuration": object_hash(cfg), "source": object_hash(source)}
    marker = destination / "checkpoint.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous.get("status") == "SUCCESS":
            if previous["identity"] != identity:
                raise RuntimeError("Existing success has different inputs/source; use a new run directory")
            if all(file_hash(destination / name) == digest for name, digest in previous["artifacts"].items()):
                return dict(previous, resumed=True)
            raise RuntimeError("Checkpoint output hash mismatch")
    start = time.perf_counter()
    write_json(marker, {"status": "RUNNING", "identity": identity})
    try:
        print(f"{path.stem}: reading real EDF and detecting both banks", flush=True)
        bank = detect(path, cfg)
        gzip_json(destination / "events.json.gz", bank)
        print(f"{path.stem}: events sealed; aggregating and staging", flush=True)
        evidences = {mode: aggregate(bank, mode) for mode in ("original", "enhanced")}
        gzip_json(destination / "evidence.json.gz", {k: [asdict(e) for e in v] for k, v in evidences.items()})
        predictions = {}
        for variant, (detectors, engine, fallback) in cfg["variants"].items():
            rows = evidences[detectors]
            outputs = original_engine(rows, fallback) if engine == "original" else enhanced_engine(
                rows, fallback, cfg["conditional_inherit_max_chain"])
            predictions[variant] = [x.to_dict() for x in outputs]
        # Prespecified chain-capped sensitivity; not a replacement for V3 or V4.
        predictions["CONDITIONAL"] = [x.to_dict() for x in enhanced_engine(
            evidences["enhanced"], "CONDITIONAL_INHERIT", cfg["conditional_inherit_max_chain"])]
        predictions["STRICT_EVIDENCE"] = [x.to_dict() for x in enhanced_engine(
            evidences["enhanced"], "ABSTAIN", cfg["conditional_inherit_max_chain"], "clinical_strict")]
        gzip_json(destination / "predictions.json.gz", predictions)
        write_json(destination / "diagnostics.json", bank["diagnostics"])
        mem = psutil.Process(os.getpid()).memory_info()
        result = {"status": "SUCCESS", "record_id": path.stem, "identity": identity,
                  "elapsed_seconds": time.perf_counter() - start,
                  "peak_working_set_bytes": getattr(mem, "peak_wset", mem.rss),
                  "epochs": len(evidences["original"]),
                  "artifacts": {name: file_hash(destination / name) for name in
                      ("events.json.gz", "evidence.json.gz", "predictions.json.gz", "diagnostics.json")}}
        write_json(marker, result)
        return result
    except Exception as exc:
        write_json(marker, {"status": "FAILED", "identity": identity, "error": repr(exc),
                            "elapsed_seconds": time.perf_counter() - start})
        raise
