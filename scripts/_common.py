"""Portable adapters; scientific core is preserved byte-for-byte."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
for key,value in {"MNE_USE_NUMBA":"false","PYTHONDONTWRITEBYTECODE":"1",
                  "OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
                  "NUMBA_CACHE_DIR":str(ROOT/".runtime/numba"),
                  "MPLCONFIGDIR":str(ROOT/".runtime/matplotlib")}.items():os.environ[key]=value


def read_json(path):return json.loads(Path(path).read_text(encoding="utf8"))


def check_core():
    from aasm_v2.provenance import file_hash
    freeze=read_json(ROOT/"configs/repair_freeze.json")
    for relative,digest in freeze["files"].items():
        # replay.py was the private, machine-specific runner. This package uses
        # portable CLI adapters, not a rebranded claim of identical run orchestration.
        if relative=="scripts/replay.py":continue
        assert file_hash(ROOT/relative)==digest,relative
    return freeze

