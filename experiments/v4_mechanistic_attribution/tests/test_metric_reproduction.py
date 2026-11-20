import os
from pathlib import Path

import pytest

from experiments.v4_mechanistic_attribution.core import (
    arm_prediction_frame,
    load_reference,
    metric_for_arm,
    read_config,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "main.yaml"


def test_b000_primary_metric_reproduction() -> None:
    if not all(
        os.environ.get(name)
        for name in ("AASM_V4_PHASE2_RUN", "AASM_V4_RESULTS_ROOT", "AASM_V4_REFERENCE_PATH")
    ):
        pytest.skip("private frozen replay paths are not configured")
    config = read_config(CONFIG)
    reference = load_reference(config)
    frame = arm_prediction_frame(Path(config["source_root"]), "B000", reference)
    metric = metric_for_arm(frame)
    assert metric["n"] == 8902
    assert abs(metric["accuracy"] - 0.5199955066277241) < 1e-12
    assert abs(metric["macro_f1"] - 0.40081488140174376) < 1e-12
    assert abs(metric["coverage"] - 0.9939339474275444) < 1e-12
