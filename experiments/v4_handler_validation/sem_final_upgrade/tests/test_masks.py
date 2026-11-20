from __future__ import annotations

import numpy as np

from experiments.v4_handler_validation.sem_final_upgrade.data.audit import teacher_fidelity_scoring_mask


def test_teacher_fidelity_scoring_mask_is_label_blind() -> None:
    signal_qc = np.array([1, 1, 0, 1], dtype=np.uint8)
    scoring = teacher_fidelity_scoring_mask(signal_qc)
    assert np.array_equal(scoring, signal_qc)
    scoring[0] = 0
    assert signal_qc[0] == 1
