from __future__ import annotations

import torch

from experiments.v4_handler_validation.sem_final_upgrade.models import build_model


def test_tcn_dense_temporal_output() -> None:
    model = build_model("inherited_tcn").eval()
    with torch.inference_mode():
        output = model(torch.zeros(2, 2, 2000))
    assert output.shape == (2, 1, 200)


def test_bigru_dense_temporal_output() -> None:
    model = build_model("cnn_bigru").eval()
    with torch.inference_mode():
        output = model(torch.zeros(2, 2, 2000))
    assert output.shape == (2, 1, 200)
