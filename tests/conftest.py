"""Only packaging/data-availability adaptations; the original test bodies remain."""
from pathlib import Path
import sys
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
(ROOT / ".runtime_tmp").mkdir(exist_ok=True)

def pytest_collection_modifyitems(config, items):
    needed = ROOT / "results/smoke_v2/records/SN001"
    for item in items:
        if "requires_hmc" not in item.keywords:
            continue
        name = item.originalname
        if name == "test_units_native_sampling_and_eog_polarity":
            requirements = [ROOT / "data/recordings/SN001.edf"]
        elif name == "test_actual_psg_annotation_deletion_replacement_permutation":
            requirements = [ROOT / "data/recordings/SN001.edf", needed / "checkpoint.json"]
        else:
            requirements = [needed / name for name in ["checkpoint.json", "signal_info.json",
                "events.csv", "features.json", "epoch_features.csv", "quality_flags.csv",
                "rule_availability.csv", "epoch_predictions.csv", "rule_trace.jsonl"]]
        if not all(p.exists() for p in requirements):
            item.add_marker(pytest.mark.skip(reason="required original PSG/cache files are not shipped"))
