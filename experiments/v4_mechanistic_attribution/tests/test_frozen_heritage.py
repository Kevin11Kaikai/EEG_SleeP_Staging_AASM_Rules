import json
import os
from pathlib import Path

import pytest

from experiments.v4_mechanistic_attribution.core import sha256


ROOT = Path(os.environ.get("AASM_V4_PHASE2_RUN", "."))


def _require_private_run() -> None:
    if "AASM_V4_PHASE2_RUN" not in os.environ:
        pytest.skip("private sealed 36-arm run is not configured")


def test_critical_frozen_artifact_hash() -> None:
    _require_private_run()
    manifest = json.loads((ROOT / "artifact_manifest.json").read_text(encoding="utf-8"))
    lookup = {item["path"]: item["sha256"] for item in manifest["files"]}
    relative = "evaluation/all_36_arm_metrics.csv"
    assert sha256(ROOT / relative) == lookup[relative]


def test_arm_and_night_counts() -> None:
    _require_private_run()
    import pandas as pd

    assert len(pd.read_csv(ROOT / "arms" / "arm_manifest_36.csv")) == 36
    assert len(pd.read_csv(ROOT / "arms" / "arm_execution_10night.csv")) == 360
