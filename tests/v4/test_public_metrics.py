from __future__ import annotations

import json

import pytest

from aasm_v4.contracts import STAGES, UNDETERMINED
from aasm_v4.evaluation.metrics import compute_fixed_mask_metrics, confusion_matrix
from aasm_v4.profiles import repository_root


def _expand(confusion: list[list[int]]) -> tuple[list[str], list[str]]:
    guesses = (*STAGES, UNDETERMINED)
    reference: list[str] = []
    prediction: list[str] = []
    for truth, row in zip(STAGES, confusion[:5]):
        for guess, count in zip(guesses, row):
            reference.extend([truth] * count)
            prediction.extend([guess] * count)
    return reference, prediction


def test_m6408_confusion_recomputes_released_metrics() -> None:
    path = (
        repository_root()
        / "results"
        / "release"
        / "v4"
        / "exhaustive_search"
        / "full_development_champions.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload["balanced_champion_metrics"]
    reference, prediction = _expand(expected["confusion"])
    actual = compute_fixed_mask_metrics(reference, prediction)
    assert actual["n"] == 8902
    assert actual["accuracy"] == pytest.approx(expected["accuracy"], abs=1e-12)
    assert actual["macro_f1"] == pytest.approx(expected["macro_f1"], abs=1e-12)
    assert actual["coverage"] == pytest.approx(expected["coverage"], abs=1e-12)


def test_abstention_is_error_on_fixed_mask() -> None:
    actual = compute_fixed_mask_metrics(["W", "N2"], ["UNDETERMINED", "N2"])
    assert actual["accuracy"] == 0.5
    assert actual["coverage"] == 0.5
    assert actual["conditional_accuracy"] == 1.0


def test_metric_contract_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError):
        confusion_matrix(["N2"], ["UNKNOWN"])
