"""Run real PSG twice: no annotations vs replaced/permuted annotation sidecars."""
import json
import pytest
from pathlib import Path
from hmc_sbtb.pipeline import predict_one, SEMANTIC_FILES
from hmc_sbtb.io_utils import sha256, code_hash, write_json
from test_system import ROOT, CFG


@pytest.mark.requires_hmc
def test_actual_psg_annotation_deletion_replacement_permutation(tmp_path):
    # Same real PSG, same ID, no annotation files; no synthetic signal enters evaluation.
    psg = tmp_path / "SN001.edf"
    psg.hardlink_to(ROOT / "data/recordings/SN001.edf")
    first = predict_one(psg, tmp_path / "without_hypnogram", CFG,
                        sha256(ROOT / "configs/paper_inspired.yaml"), code_hash())
    labels = ["R", "N3", "W", "N1", "N2"]
    (tmp_path / "SN001_sleepscoring.txt").write_text("\n".join(reversed(labels)))
    (tmp_path / "SN001_sleepscoring.edf").write_bytes(b"deliberately corrupt annotation")
    second = predict_one(psg, tmp_path / "permuted_corrupt_hypnogram", CFG,
                         sha256(ROOT / "configs/paper_inspired.yaml"), code_hash())
    baseline = json.loads((ROOT / "results/smoke_v2/records/SN001/checkpoint.json").read_text())
    assert first["output_hashes"] == second["output_hashes"] == baseline["output_hashes"]
    write_json(ROOT / "results/smoke_v2/determinism_and_leakage.json", {
        "status": "TESTED", "real_record": "SN001", "comparison": "original annotations untouched vs no hypnogram vs replaced/permuted corrupt annotations",
        "semantic_hashes_equal": True, "semantic_hashes": first["output_hashes"],
        "real_prediction_repetitions": 3})
