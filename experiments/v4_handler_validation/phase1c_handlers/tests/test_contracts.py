from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.v4_handler_validation.phase1c_handlers.common import (
    interval_iou,
    intervals_from_mask,
    match_intervals,
)
from experiments.v4_handler_validation.phase1c_handlers.adapters import NeuralHandlerAdapter
from experiments.v4_handler_validation.phase1c_handlers.data import HANDLER_ROLES, SPLIT_GROUPS
from experiments.v4_handler_validation.phase1c_handlers.edf import read_header
from experiments.v4_handler_validation.phase1c_handlers.inference import apply_output_semantics
from experiments.v4_handler_validation.phase1c_handlers.models import FAMILY_SPECS, build_model


@pytest.mark.parametrize("family", sorted(FAMILY_SPECS))
def test_dense_model_temporal_contract(family: str) -> None:
    spec = FAMILY_SPECS[family]
    seconds = 4
    value = torch.zeros(2, spec.input_channels, seconds * spec.input_rate_hz)
    output = build_model(family)(value)
    assert output.shape == (
        2,
        len(spec.output_heads),
        seconds * spec.output_rate_hz,
    )
    assert torch.isfinite(output).all()


def test_emg_high_is_also_normal() -> None:
    probability = np.zeros((3, 10), dtype=np.float32)
    probability[1] = 0.1
    probability[2, 3:7] = 0.9
    output = apply_output_semantics(probability, "chin_emg")
    assert np.all(output[1] >= output[2])


def test_interval_contract() -> None:
    mask = np.asarray([0, 1, 1, 0, 1, 0], dtype=bool)
    intervals = intervals_from_mask(mask, 2.0, minimum_seconds=0.5)
    assert intervals == [(0.5, 1.5), (2.0, 2.5)]
    assert all(0 <= onset < offset <= 3 for onset, offset in intervals)


def test_sweep_interval_matching_equals_all_pairs_reference() -> None:
    reference = [(9.0, 10.5), (1.0, 2.0), (4.0, 5.5), (7.0, 8.0)]
    predicted = [(4.2, 5.3), (0.8, 1.8), (12.0, 13.0), (7.4, 8.2)]
    candidates = []
    for truth_index, truth in enumerate(reference):
        for guess_index, guess in enumerate(predicted):
            score = interval_iou(truth, guess)
            if score >= 0.2:
                candidates.append((score, truth_index, guess_index))
    matched_truth: set[int] = set()
    matched_guess: set[int] = set()
    scores = []
    for score, truth_index, guess_index in sorted(candidates, reverse=True):
        if truth_index not in matched_truth and guess_index not in matched_guess:
            matched_truth.add(truth_index)
            matched_guess.add(guess_index)
            scores.append(score)
    expected = {
        "tp": len(scores),
        "fp": len(predicted) - len(scores),
        "fn": len(reference) - len(scores),
    }
    actual = match_intervals(reference, predicted, minimum_iou=0.2)
    assert {key: actual[key] for key in expected} == expected
    assert actual["mean_matched_iou"] == pytest.approx(np.mean(scores))


def test_frozen_group_split_has_no_overlap() -> None:
    values = [set(group_ids) for group_ids in SPLIT_GROUPS.values()]
    assert all(
        not values[i] & values[j]
        for i in range(len(values))
        for j in range(i + 1, len(values))
    )


def test_handler_specific_source_roles() -> None:
    assert HANDLER_ROLES["E05"] == ("o",)
    assert HANDLER_ROLES["E08"] == ("c",)
    assert HANDLER_ROLES["E12"] == ("c", "o")


def test_continuous_adapter_export_contract(tmp_path: Path) -> None:
    adapter = NeuralHandlerAdapter(
        {
            "handler_id": "E05",
            "family": "eeg_background",
            "checkpoint": "unused.pt",
            "checkpoint_sha256": "test-checkpoint",
            "threshold": 0.5,
            "eventization": {"minimum_seconds": 1.0, "merge_gap_seconds": 0.0},
            "window_seconds": 10,
        }
    )
    adapter.checkpoint_sha256 = "test-checkpoint"
    prediction = np.zeros((3, 10), dtype=np.float32)
    prediction[0, 2:5] = 0.9
    events = [
        {
            "onset": 2.0,
            "offset": 5.0,
        }
    ]
    output = adapter.export_events(events, prediction, tmp_path / "events.json")
    with np.load(output["probability"], allow_pickle=False) as payload:
        assert set(
            [
                "time_axis",
                "activity_score",
                "activity_intervals",
                "valid_mask",
                "temporal_resolution_seconds",
                "signal_unit",
                "source_model_sha256",
                "output_kind",
            ]
        ).issubset(payload.files)
        assert payload["activity_score"].shape == (10,)
        assert payload["activity_intervals"].tolist() == [[2.0, 5.0]]


def test_real_public_edf_header_if_available() -> None:
    root = os.environ.get("AASM_V4_SLEEPEDF_ROOT")
    if not root:
        pytest.skip("AASM_V4_SLEEPEDF_ROOT is not configured")
    candidates = sorted(Path(root).glob("*-PSG.edf"))
    path = candidates[0] if candidates else Path(root) / "missing.edf"
    if not path.exists():
        pytest.skip("Local public SleepEDF signal is unavailable")
    header = read_header(path)
    by_name = {signal.name: signal for signal in header.signals}
    assert by_name["EEG Fpz-Cz"].sampling_rate == 100
    assert by_name["EEG Pz-Oz"].sampling_rate == 100
    assert by_name["EMG submental"].sampling_rate == 1
    adapter = NeuralHandlerAdapter(
        {
            "handler_id": "E05",
            "family": "eeg_background",
            "checkpoint": "unused.pt",
            "checkpoint_sha256": "test-checkpoint",
            "threshold": 0.5,
            "eventization": {"minimum_seconds": 1.0, "merge_gap_seconds": 0.0},
            "window_seconds": 10,
        }
    )
    contract = adapter.validate_input({"file_path": str(path)})
    assert contract["handler_source_channels"] == ["EEG Pz-Oz"]
    assert contract["handler_source_sampling_rate_hz"] == 100.0
    adapter.checkpoint_sha256 = "test-checkpoint"
    prediction = np.zeros((3, 10), dtype=np.float32)
    prediction[0, 2:5] = 0.9
    events = adapter.postprocess(
        prediction,
        {"file_path": str(path), "recording_id": "sleepedf-contract-test"},
    )
    assert events[0]["channel"] == "EEG Pz-Oz"
    assert events[0]["source_signal_pointer"]["sampling_rate_hz"] == 100.0
