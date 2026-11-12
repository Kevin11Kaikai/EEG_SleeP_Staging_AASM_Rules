"""Boundary, physical-signal, leakage, state, metric and real EDF regression tests."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score
from hmc_sbtb.temporal import Tri, intersection, coverage, onset_in, pair_overlap
from hmc_sbtb.rules import run_rules, condition
from hmc_sbtb.preprocessing import (bandpass, quality_epochs, sample_mask,
                                    psd_ratio, emg_envelope, segments, filter_valid_blocks)
from hmc_sbtb.edf_loader import volts_to_uv, inspect_psg, load_channel
from hmc_sbtb.annotations import align, txt_annotations
from hmc_sbtb.evaluation import metrics, bootstrap_subjects
from hmc_sbtb.io_utils import code_hash, sha256, write_json
from hmc_sbtb.pipeline import record_valid, SEMANTIC_FILES

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/paper_inspired.yaml").read_text())


def feature(index=0, **kwargs):
    f = {"record_id": "S", "subject_id": "S", "epoch_index": index,
         "start_sec": index * 30, "end_sec": (index + 1) * 30,
         "quality_status": "AVAILABLE", "quality_barrier": False,
         "generates_alpha_rhythm": True, "rem_low_overlap_sec": 0,
         "n2_window_available": True, "n2_kcomplex_available": True,
         "n2_spindle_ids": [], "n2_kcomplex_ids": []}
    for name in ["slow_wave", "spindle", "alpha", "lamf", "low_emg", "rem_candidate", "kcomplex", "arousal_candidate"]:
        f[name + "_sec"] = 0.0
        f[name + "_available"] = True
        f[name + "_ids"] = []
    f.update(kwargs)
    return f


@pytest.mark.parametrize("a,b,and_result,or_result", [
    (Tri.TRUE, Tri.UNKNOWN, Tri.UNKNOWN, Tri.TRUE),
    (Tri.FALSE, Tri.UNKNOWN, Tri.FALSE, Tri.UNKNOWN),
    (Tri.UNKNOWN, Tri.UNKNOWN, Tri.UNKNOWN, Tri.UNKNOWN),
    (Tri.TRUE, Tri.FALSE, Tri.FALSE, Tri.TRUE),
])
def test_kleene(a, b, and_result, or_result):
    assert a & b == and_result
    assert a | b == or_result
    assert ~Tri.UNKNOWN == Tri.UNKNOWN


@pytest.mark.parametrize("seconds,pred", [(5.9, "UNDETERMINED"), (6, "UNDETERMINED"), (6.01, "N3")])
def test_strict_slow_wave(seconds, pred):
    assert run_rules([feature(slow_wave_sec=seconds)], CFG, "A")[0][0]["final_prediction"] == pred


@pytest.mark.parametrize("seconds,pred", [(14.99, "UNDETERMINED"), (15, "UNDETERMINED"), (15.01, "W")])
def test_strict_alpha(seconds, pred):
    assert run_rules([feature(alpha_sec=seconds)], CFG, "A")[0][0]["final_prediction"] == pred


def test_multichannel_union_and_duplicates():
    assert coverage([(28, 34), (29, 34), (31, 37), (31, 37)], 30, 60) == 7
    assert coverage([(28, 34), (29, 34)], 0, 30) == 2
    assert intersection((0, 30), (30, 31)) == 0
    assert pair_overlap([(0, 10), (5, 15)], [(4, 6), (8, 20)], 0, 30) == 9


@pytest.mark.parametrize("onset,expected", [(14.999, False), (15, True), (30, True), (44.999, True), (45, False)])
def test_n2_onset_boundaries(onset, expected):
    assert onset_in(onset, 15, 45) == expected


@pytest.mark.requires_hmc
def test_units_native_sampling_and_eog_polarity():
    assert np.allclose(volts_to_uv([1e-6, -50e-6]), [1, -50])
    info = inspect_psg(ROOT / "data/recordings/SN001.edf", CFG)
    assert all(r["native_sampling_hz"] == 256 for r in info["channels"])
    assert info["selected_channels"]["eog_left"] == "EOG E1-M2"
    assert info["selected_channels"]["eog_right"] == "EOG E2-M2"
    assert all(r["polarity_multiplier"] == 1 for r in info["channels"])


def test_real_eog_detector_synthetic_and_artifact():
    import yasa
    sf = 128
    t = np.arange(sf * 60) / sf
    rng = np.random.default_rng(78)
    eye = 120 * (np.exp(-((t - 20) / .22) ** 2) + np.exp(-((t - 40) / .22) ** 2))
    left, right = eye + rng.normal(0, 2, len(t)), -eye + rng.normal(0, 2, len(t))
    found = yasa.rem_detect(left, right, sf, hypno=None, verbose=False, **CFG["detectors"]["rem"])
    assert found is not None and len(found.summary()) >= 2
    same_polarity = yasa.rem_detect(left, left.copy(), sf, hypno=None, verbose=False, **CFG["detectors"]["rem"])
    assert same_polarity is None or len(same_polarity.summary()) == 0
    artifact = left.copy()
    artifact[100:200] = 10000
    good, _ = quality_epochs(artifact, sf, "eog", CFG["quality"])
    assert not good[0]


def test_spectral_alpha_lamf_features():
    sf = 128
    t = np.arange(sf * 10) / sf
    rng = np.random.default_rng(4)
    alpha = 20 * np.sin(2 * np.pi * 10 * t) + rng.normal(0, .1, len(t))
    theta = 2 * np.sin(2 * np.pi * 5 * t) + rng.normal(0, .1, len(t))
    assert psd_ratio(alpha, sf, [8, 12], [.3, 35]) > .95
    assert psd_ratio(theta, sf, [4, 7], [.3, 35]) > .95
    threshold = np.mean(abs(theta)) - .01 * np.std(theta)
    assert len(segments(abs(theta) < threshold, sf, 1)) == 0  # literal continuous low amplitude


def test_emg_processing_flat_not_low():
    sf = 128
    rng = np.random.default_rng(7)
    x = rng.normal(0, 4, sf * 60)
    x[:sf * 30] *= .3
    clean = bandpass(x, sf, [10, 45])
    events, rms, baseline = emg_envelope(clean, sf, np.ones(len(x), bool), CFG["detectors"]["emg"])
    assert baseline["status"] == "AVAILABLE" and events and rms
    flat = np.zeros(sf * 60)
    good, _ = quality_epochs(flat, sf, "emg", CFG["quality"])
    events, _, baseline = emg_envelope(flat, sf, sample_mask(good, len(flat), sf), CFG["detectors"]["emg"])
    assert not good.any() and events == [] and baseline["status"] == "INVALID_SIGNAL"


def test_invalid_blocks_do_not_pollute_filter():
    sf = 128
    rng = np.random.default_rng(2)
    x = rng.normal(0, 10, sf * 90)
    mask = np.ones(len(x), bool)
    mask[sf * 30:sf * 60] = False
    y = x.copy()
    y[~mask] = 1e8
    assert np.array_equal(filter_valid_blocks(x, sf, mask, [.3, 35]),
                          filter_valid_blocks(y, sf, mask, [.3, 35]))


def test_missing_kcomplex_unknown_rem_not_confirmed():
    f = feature(lamf_sec=20, rem_candidate_sec=1, low_emg_sec=20, rem_low_overlap_sec=.5,
                kcomplex_available=False, kcomplex_sec=None)
    output, traces = run_rules([f], CFG, "A")
    assert output[0]["final_prediction"] != "R"
    assert traces[0]["actual_execution"][2]["conditions"]["no_kcomplex"] == Tri.UNKNOWN
    assert condition({"x_available": True, "x_sec": None}, "x", lambda x: x > 0) == Tri.UNKNOWN
    assert condition({"x_available": True, "x_sec": float("nan")}, "x", lambda x: x > 0) == Tri.UNKNOWN


def test_priority_not_evaluated_and_no_else_n1():
    result, traces = run_rules([feature(slow_wave_sec=7, alpha_sec=20, n2_spindle_ids=["sp"])], CFG)
    assert result[0]["final_prediction"] == "N3"
    assert all(t["status"] == "NOT_EVALUATED" for t in traces[0]["actual_execution"][1:])
    assert run_rules([feature()], CFG)[0][0]["final_prediction"] == "UNDETERMINED"


def test_n1_requires_actual_previous_pass1_wake():
    f = [feature(alpha_sec=20), feature(1), feature(2, lamf_sec=20, alpha_sec=1)]
    result, _ = run_rules(f, CFG)
    assert result[1]["decision_source"] == "INHERITANCE"
    assert result[2]["pass1_prediction"] != "N1"
    actual, _ = run_rules([f[0], feature(1, lamf_sec=20, alpha_sec=1)], CFG)
    assert actual[1]["pass1_prediction"] == "N1"


def test_continuation_multiple_epochs_and_interrupt():
    f = [feature(n2_spindle_ids=["sp"]), feature(1), feature(2), feature(3, alpha_sec=20), feature(4)]
    result, _ = run_rules(f, CFG, "B")
    assert [x["final_prediction"] for x in result] == ["N2", "N2", "N2", "W", "UNDETERMINED"]
    assert result[2]["anchor_epoch"] == 0 and result[2]["chain_length"] == 2
    assert result[3]["pass1_prediction"] == result[3]["final_prediction"] == "W"


def test_rem_continuation_and_incompatible_spindle():
    f = [feature(lamf_sec=20, rem_candidate_sec=1, low_emg_sec=20, rem_low_overlap_sec=.5),
         feature(1, lamf_sec=20, low_emg_sec=20), feature(2, lamf_sec=20, low_emg_sec=20, spindle_sec=1)]
    result, _ = run_rules(f, CFG, "B")
    assert [x["final_prediction"] for x in result] == ["R", "R", "UNDETERMINED"]


def test_inheritance_chains_quality_barrier_and_first_epoch():
    result, _ = run_rules([feature(), feature(1, alpha_sec=20), feature(2), feature(3),
                           feature(4, quality_barrier=True, quality_status="INVALID_SIGNAL"), feature(5)], CFG)
    assert result[0]["final_prediction"] == "UNDETERMINED"
    assert result[3]["decision_source"] == "INHERITANCE"
    assert (result[3]["parent_epoch"], result[3]["anchor_epoch"], result[3]["chain_length"]) == (2, 1, 2)
    assert result[4]["final_prediction"] == result[5]["final_prediction"] == "UNDETERMINED"


@pytest.mark.parametrize("change", [{"record_id": "T"}, {"subject_id": "T"}, {"start_sec": 90}, {"epoch_index": 5}])
def test_cross_record_subject_gap_isolation(change):
    result, _ = run_rules([feature(alpha_sec=20), feature(1, **change)], CFG)
    assert result[1]["final_prediction"] == "UNDETERMINED"


def test_alignment_gaps_conflicts_and_unknown():
    rows = [{"start": 0, "end": 30, "label": "W", "raw_label": "Sleep stage W"},
            {"start": 30, "end": 60, "label": "N2", "raw_label": "Sleep stage 2"},
            {"start": 45, "end": 65, "label": "N3", "raw_label": "Sleep stage 3"}]
    result = align(rows, 3)
    assert result[0]["true_stage"] == "W"
    assert result[1]["alignment_status"] == "CONFLICT"
    assert result[2]["true_stage"] == "UNALIGNED"


def test_txt_uses_relative_time_not_rows(tmp_path):
    p = tmp_path / "labels.txt"
    p.write_text("Date, Time, Recording onset, Duration, Annotation, Linked channel\n"
                 "2020-01-01, 22:00:00, 0, 60, Sleep stage W,\n"
                 "2020-01-01, 22:01:00, 60, 30, Sleep stage 2,\n")
    result = align(txt_annotations(p, "2020-01-01T22:00:00+00:00"), 3)
    assert [x["true_stage"] for x in result] == ["W", "W", "N2"]


def test_hmc_txt_n_label_and_dot_clock(tmp_path):
    p = tmp_path / "labels.txt"
    p.write_text("Date, Time, Recording onset, Duration, Annotation, Linked channel\n"
                 "01.01.01, 23.59.30, 0, 30, Sleep stage N1,\n"
                 "02.01.01, 00.00.00, 30, 30, Sleep stage N2,\n"
                 "02.01.01, 00.00.30, 60, 30, Sleep stage N3,\n")
    parsed = txt_annotations(p, "2001-01-01T23:59:30+00:00")
    assert [x["true_stage"] for x in align(parsed, 3)] == ["N1", "N2", "N3"]
    assert all(x["absolute_relative_discrepancy_sec"] == 0 for x in parsed)


def test_yasa_slow_wave_and_spindle_on_synthetic_signals():
    import yasa
    sf = 128
    rng = np.random.default_rng(12)
    t = np.arange(sf * 120) / sf
    noise = rng.normal(0, 3, len(t))
    sw = noise.copy()
    for start in [20, 40, 60, 80]:
        window = (t >= start) & (t < start + 3)
        sw[window] += 100 * np.sin(2 * np.pi * 1 * t[window])
    found = yasa.sw_detect(sw, sf, hypno=None, verbose=False, **CFG["detectors"]["sw"])
    assert found is not None and len(found.summary()) >= 4
    spindle = noise.copy()
    for start in [20, 40, 60, 80]:
        window = (t >= start) & (t < start + 1)
        spindle[window] += 30 * np.hanning(window.sum()) * np.sin(2 * np.pi * 13 * t[window])
    found = yasa.spindles_detect(spindle, sf, hypno=None, verbose=False, **CFG["detectors"]["spindle"])
    assert found is not None and len(found.summary()) >= 3


def test_unknown_in_metrics_fn_and_joint_kappa():
    true, pred = ["W", "N2", "N2", "R"], ["W", "N2", "UNDETERMINED", "N2"]
    result = metrics(true, pred)
    assert np.array(result["confusion_matrix"]).shape == (5, 6)
    assert result["accuracy"] == .5
    assert result["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(true, pred))
    assert result["stage_metrics"]["N2"]["fn"] == 1
    assert result["kappa"] == pytest.approx(cohen_kappa_score(true, pred, labels=["W", "N1", "N2", "N3", "R", "UNDETERMINED"]))
    assert metrics(["W"], ["W"])["kappa"] is None


def test_bootstrap_resamples_subjects_with_all_nights_together():
    import pandas as pd
    frame = pd.DataFrame({"subject_id": ["A", "A", "A", "B"],
                          "record_id": ["A1", "A1", "A2", "B1"],
                          "true_stage": ["W", "N2", "R", "N2"],
                          "final_prediction": ["W", "N2", "UNDETERMINED", "N2"]})
    cfg = {"bootstrap_seed": 19, "bootstrap_iterations": 50}
    result = bootstrap_subjects(frame, cfg)
    rng = np.random.default_rng(19)
    values = []
    for _ in range(50):
        ids = rng.integers(0, 2, 2)
        sample = pd.concat([frame[frame.subject_id == ["A", "B"][i]] for i in ids])
        values.append(float((sample.true_stage == sample.final_prediction).mean()))
    assert result["subjects"] == 2
    assert np.allclose(result["percentile_95"]["accuracy"], np.quantile(values, [.025, .975]))


def test_abc_independence_and_determinism():
    f = [feature(n2_spindle_ids=["sp"]), feature(1), feature(2, arousal_candidate_sec=3)]
    original = copy.deepcopy(f)
    a, _ = run_rules(f, CFG, "A")
    b, _ = run_rules(f, CFG, "B")
    c, trace = run_rules(f, CFG, "C")
    assert a[1]["final_prediction"] == "UNDETERMINED" and b[1]["final_prediction"] == "N2"
    assert b[2]["final_prediction"] == "UNDETERMINED" and c[2]["decision_source"] == "INHERITANCE"
    assert f == original
    assert (c, trace) == run_rules(f, CFG, "C")


def test_checkpoint_content_and_key_invalidation(tmp_path):
    for name in SEMANTIC_FILES:
        (tmp_path / name).write_text("valid")
    key = {"psg": "a", "config": "b", "code": "c"}
    write_json(tmp_path / "checkpoint.json", {"key": key, "status": "PREDICTIONS_FROZEN", "output_hashes": {n: sha256(tmp_path / n) for n in SEMANTIC_FILES}})
    assert record_valid(tmp_path, key)
    assert not record_valid(tmp_path, {**key, "code": "changed"})
    (tmp_path / SEMANTIC_FILES[0]).write_text("tampered")
    assert not record_valid(tmp_path, key)


@pytest.mark.requires_hmc
def test_real_edf_frozen_end_to_end():
    folder = ROOT / "results/smoke_v2/records/SN001"
    checkpoint = json.loads((folder / "checkpoint.json").read_text())
    assert checkpoint["status"] == "PREDICTIONS_FROZEN" and checkpoint["epochs"] > 100
    assert checkpoint["events"] > 0
    assert all(sha256(folder / name) == digest for name, digest in checkpoint["output_hashes"].items())
    assert json.loads((folder / "signal_info.json").read_text())["selected_channels"]["emg"] == "EMG chin"


def test_prediction_source_has_no_annotation_access():
    for name in ["pipeline", "detectors", "features", "rules", "edf_loader", "preprocessing"]:
        source = (ROOT / "src" / "hmc_sbtb" / f"{name}.py").read_text()
        assert "from .annotations" not in source and "from .evaluation" not in source
        assert "SleepStaging(" not in source
