"""Prediction orchestration. Reads only PSG/config; outputs sealed per-record artifacts."""
import gc
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import psutil
import yaml
from .io_utils import sha256, code_hash, atomic_text, write_json, csv_write
from .edf_loader import inspect_psg
from .detectors import detect_record, EVENT_COLUMNS
from .features import aggregate
from .rules import run_rules


SEMANTIC_FILES = ["events.csv", "features.json", "epoch_features.csv", "quality_flags.csv",
                  "rule_availability.csv", "epoch_predictions.csv", "rule_trace.jsonl"]


def record_valid(folder, key):
    checkpoint = folder / "checkpoint.json"
    if not checkpoint.exists():
        return False
    saved = json.loads(checkpoint.read_text())
    return (saved.get("key") == key and saved.get("status") == "PREDICTIONS_FROZEN" and
            all((folder / name).exists() and sha256(folder / name) == digest
                for name, digest in saved["output_hashes"].items()))


def prediction_frame(rows):
    frame = pd.DataFrame(rows)
    for name in ["event_ids"]:
        if name in frame:
            frame[name] = frame[name].map(lambda x: json.dumps(x, sort_keys=True))
    return frame


def predict_one(psg, folder, cfg, config_sha, source_sha):
    started = time.monotonic()
    key = {"psg_sha256": sha256(psg), "config_sha256": config_sha, "source_sha256": source_sha}
    if record_valid(folder, key):
        return json.loads((folder / "checkpoint.json").read_text()) | {"cache_reused": True}
    if (folder / "checkpoint.json").exists():
        raise RuntimeError("existing prediction cache invalid; create a new run instead of overwriting")
    folder.mkdir(parents=True, exist_ok=True)
    info = inspect_psg(psg, cfg)
    write_json(folder / "signal_info.json", info)
    csv_write(pd.DataFrame(info["channels"]).assign(record_id=psg.stem), folder / "channel_manifest.csv")
    events, quality, availability, backgrounds, goods = detect_record(psg, info, cfg)
    features = aggregate(events, goods, availability, info, cfg)
    predictions, traces = run_rules(features, cfg, "C")
    csv_write(pd.DataFrame(events, columns=EVENT_COLUMNS), folder / "events.csv")
    write_json(folder / "features.json", features)
    feature_frame = pd.DataFrame(features)
    for col in feature_frame.columns:
        if col.endswith("_ids"):
            feature_frame[col] = feature_frame[col].map(json.dumps)
    csv_write(feature_frame, folder / "epoch_features.csv")
    csv_write(pd.DataFrame(quality), folder / "quality_flags.csv")
    csv_write(pd.DataFrame([{"record_id": psg.stem, "detector": k, "status": v}
                           for k, v in sorted(availability.items())]), folder / "rule_availability.csv")
    write_json(folder / "background_statistics.json", backgrounds)
    csv_write(prediction_frame(predictions), folder / "epoch_predictions.csv")
    atomic_text(folder / "rule_trace.jsonl", "".join(json.dumps(t, sort_keys=True, allow_nan=False) + "\n" for t in traces))
    checkpoint = {"status": "PREDICTIONS_FROZEN", "record_id": psg.stem, "key": key,
                  "epochs": len(predictions), "events": len(events),
                  "elapsed_sec": time.monotonic() - started,
                  "process_rss_bytes": psutil.Process().memory_info().rss,
                  "output_hashes": {n: sha256(folder / n) for n in SEMANTIC_FILES}}
    write_json(folder / "checkpoint.json", checkpoint)
    del events, quality, features, predictions, traces, goods
    gc.collect()
    return checkpoint


def merge_csv(run, records, name):
    import csv
    temp = run / (name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as target:
        writer = None
        for record in records:
            path = run / "records" / record / name
            if not path.exists():
                continue
            with path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                if writer is None:
                    writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
                    writer.writeheader()
                writer.writerows(reader)
    temp.replace(run / name)


def predict(config, subset, data_root, run_dir):
    config, data_root, run = Path(config), Path(data_root), Path(run_dir)
    cfg = yaml.safe_load(config.read_text())
    selected = json.loads((data_root / "pilot_selection.json").read_text())
    records = selected["pilot"] if subset == "pilot" else selected["smoke"] if subset == "smoke" else sorted(selected["pilot"] + selected["untouched"])
    # Official metadata/hash inventory is allowed; never inspect annotation contents.
    download_status = json.loads((data_root / "download_status.json").read_text())
    cfg_sha, src_sha = sha256(config), code_hash()
    run.mkdir(parents=True, exist_ok=True)
    frozen_path = run / "frozen_run_manifest.json"
    if frozen_path.exists():
        manifest = json.loads(frozen_path.read_text())
        if (manifest["configuration_sha256"] != cfg_sha or manifest["prediction_source_sha256"] != src_sha
                or manifest["selected_records"] != records):
            raise RuntimeError("run configuration/source/selection changed; use new run directory")
    else:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--short"], capture_output=True, text=True).stdout
        versions = {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "mne", "yasa", "pandas", "scikit-learn", "matplotlib", "PyYAML", "psutil"]}
        manifest = {"status": "PREDICTING", "created_utc": datetime.now(timezone.utc).isoformat(),
                    "run_id": run.name, "configuration_sha256": cfg_sha, "prediction_source_sha256": src_sha,
                    "git_revision": revision, "initial_git_dirty": dirty, "dependencies": versions,
                    "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
                    "physical_memory_bytes": psutil.virtual_memory().total, "cpu_logical": psutil.cpu_count(),
                    "data_root": str(data_root.resolve()), "subset": subset, "selected_records": records,
                    "pilot_selection": selected, "channel_policy": cfg["channel_map"],
                    "evaluation_policy": cfg["evaluation"], "official_count": selected["official_count"],
                    "annotation_access_during_prediction": False, "records": {},
                    "source_files": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}}
        atomic_text(run / "run_config.yaml", config.read_text())
        write_json(frozen_path, manifest)
    manifest_rows = []
    sums = {line.split(maxsplit=1)[1].strip(): line.split()[0]
            for line in (data_root / "SHA256SUMS.txt").read_text().splitlines()}
    for record in records:
        files = download_status.get(record, [])
        data_row = {"record_id": record, "subject_id": record, "status": "MISSING",
                    "official_psg_sha256": sums.get(f"recordings/{record}.edf")}
        try:
            if len(files) != 3 or any(x["status"] != "VERIFIED" for x in files):
                raise RuntimeError("incomplete verified three-file pair")
            psg = data_root / "recordings" / f"{record}.edf"
            actual = sha256(psg)
            if actual != data_row["official_psg_sha256"]:
                data_row["status"] = "CORRUPT"
                raise RuntimeError("PSG official checksum mismatch")
            data_row.update({"status": "VERIFIED", "psg_bytes": psg.stat().st_size,
                             "psg_sha256": actual})
            result = predict_one(psg, run / "records" / record, cfg, cfg_sha, src_sha)
            manifest["records"][record] = result
            print(f"{record} PREDICTIONS_FROZEN epochs={result['epochs']} events={result['events']} sec={result['elapsed_sec']:.1f}", flush=True)
        except Exception as exc:
            manifest["records"][record] = {"status": "BLOCKED", "error": repr(exc)}
            atomic_text(run / "errors" / f"{record}.txt", traceback.format_exc())
            data_row["error"] = repr(exc)
            print(record, "BLOCKED", repr(exc), flush=True)
        manifest_rows.append(data_row)
        csv_write(pd.DataFrame(manifest_rows), run / "data_manifest.csv")
        write_json(frozen_path, manifest)
    success = [r for r in records if manifest["records"][r]["status"] == "PREDICTIONS_FROZEN"]
    for name in ["channel_manifest.csv", "events.csv", "epoch_features.csv", "quality_flags.csv", "rule_availability.csv", "epoch_predictions.csv"]:
        merge_csv(run, success, name)
    with (run / "rule_trace.jsonl.tmp").open("wb") as target:
        for r in success:
            with (run / "records" / r / "rule_trace.jsonl").open("rb") as source:
                import shutil
                shutil.copyfileobj(source, target)
    (run / "rule_trace.jsonl.tmp").replace(run / "rule_trace.jsonl")
    manifest["status"] = "PREDICTIONS_FROZEN" if success else "BLOCKED"
    manifest["prediction_artifacts_sealed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["successfully_predicted"] = success
    manifest["prediction_output_hashes"] = {n: sha256(run / n) for n in SEMANTIC_FILES if (run / n).exists()}
    write_json(frozen_path, manifest)
    return manifest
