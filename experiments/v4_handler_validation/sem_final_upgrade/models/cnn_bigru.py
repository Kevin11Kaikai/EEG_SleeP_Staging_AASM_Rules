from __future__ import annotations

import torch
from torch import nn


class CNNBiGRUSegmenter(nn.Module):
    """Optional long-context fallback; it is not run unless the TCN is degenerate."""

    input_rate_hz = 100
    output_rate_hz = 10
    architecture_name = "cnn_bigru"

    def __init__(self):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv1d(2, 24, kernel_size=11, stride=2, padding=5),
            nn.GroupNorm(4, 24),
            nn.GELU(),
            nn.Conv1d(24, 32, kernel_size=7, stride=5, padding=3),
            nn.GroupNorm(4, 32),
            nn.GELU(),
        )
        self.context = nn.GRU(
            input_size=32,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(64, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        features = self.frontend(value).transpose(1, 2)
        context, _ = self.context(features)
        return self.head(context).transpose(1, 2)
