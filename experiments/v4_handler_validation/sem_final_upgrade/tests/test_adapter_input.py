from __future__ import annotations

import pytest

from experiments.v4_handler_validation.sem_final_upgrade.handler.adapter import (
    E07SEMNeuralHandler,
    InputContractError,
)


def _handler_without_checkpoint() -> E07SEMNeuralHandler:
    handler = object.__new__(E07SEMNeuralHandler)
    handler.config = {
        "expected_channel_roles": ["left_eog", "right_eog"],
        "expected_reference": "M2",
        "expected_polarity": "as_recorded",
    }
    return handler


def _metadata() -> dict[str, object]:
    return {
        "channels": ["EOG E1-M2", "EOG E2-M2"],
        "channel_roles": ["left_eog", "right_eog"],
        "reference": "M2",
        "polarity": "as_recorded",
        "sampling_rate_hz": 256,
        "signal_unit": "uV",
    }


def test_adapter_rejects_wrong_channel_order() -> None:
    metadata = _metadata()
    metadata["channel_roles"] = ["right_eog", "left_eog"]
    with pytest.raises(InputContractError):
        _handler_without_checkpoint().validate_input(metadata)


def test_adapter_rejects_rate_incompatible_with_filter() -> None:
    metadata = _metadata()
    metadata["sampling_rate_hz"] = 30
    with pytest.raises(InputContractError):
        _handler_without_checkpoint().validate_input(metadata)
