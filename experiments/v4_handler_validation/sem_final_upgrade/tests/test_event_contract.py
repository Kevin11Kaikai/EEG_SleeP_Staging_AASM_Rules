from __future__ import annotations

import numpy as np
import pytest

from experiments.v4_handler_validation.sem_final_upgrade.common import eventize
from experiments.v4_handler_validation.sem_final_upgrade.handler.event_contract import validate_events


def test_merge_happens_before_minimum_duration() -> None:
    probability = np.zeros(50, dtype=np.float32)
    probability[10:16] = 0.9
    probability[18:24] = 0.9
    valid = np.ones_like(probability, dtype=np.uint8)
    intervals = eventize(probability, valid, 0.5, 10, 1.0, 5.0, 0.3)
    assert intervals == [(1.0, 2.4)]


def test_exact_merge_gap_is_not_lost_to_float_rounding() -> None:
    probability = np.zeros(60, dtype=np.float32)
    probability[5:15] = 1.0
    probability[18:28] = 1.0
    valid = np.ones_like(probability, dtype=np.uint8)
    assert eventize(probability, valid, 0.5, 10, 1.0, 5.0, 0.3) == [(0.5, 2.8)]


def test_eventization_does_not_merge_across_invalid_signal() -> None:
    probability = np.zeros(50, dtype=np.float32)
    probability[10:16] = 0.9
    probability[18:24] = 0.9
    valid = np.ones_like(probability, dtype=np.uint8)
    valid[16:18] = 0
    assert eventize(probability, valid, 0.5, 10, 1.0, 5.0, 0.3) == []


def test_event_contract_rejects_artificially_long_interval() -> None:
    event = {
        "event_id": "x",
        "event_type": "sem",
        "record_id": "r",
        "onset": 1.0,
        "offset": 7.0,
        "channel": "L+R",
        "detector_version": "v",
        "parameters": {},
        "quality_status": "AVAILABLE",
        "evidence_status": "CANDIDATE_PROXY",
        "measurements": {},
        "source_model_sha256": "0" * 64,
        "source_signal_pointer": {},
        "adapter_version": "v",
    }
    with pytest.raises(ValueError):
        validate_events([event], 10.0)
