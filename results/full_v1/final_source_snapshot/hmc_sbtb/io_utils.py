"""Atomic persistence and content fingerprints."""
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def write_json(path, obj):
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, allow_nan=False))


def code_hash():
    h = hashlib.sha256()
    names = ["__init__.py", "io_utils.py", "edf_loader.py", "preprocessing.py",
             "detectors.py", "temporal.py", "features.py", "rules.py", "pipeline.py"]
    for p in sorted(Path(__file__).parent / n for n in names):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def csv_write(frame, path):
    atomic_text(path, frame.to_csv(index=False, lineterminator="\n"))
