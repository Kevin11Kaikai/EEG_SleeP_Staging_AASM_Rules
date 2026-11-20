from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                padding=2 * dilation,
                dilation=dilation,
            ),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GroupNorm(4, channels),
        )
        self.activation = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.net(value))


class InheritedTCNSegmenter(nn.Module):
    """Phase1C EOG TCN encoder with a dedicated SEM-only dense head."""

    input_rate_hz = 100
    output_rate_hz = 10
    architecture_name = "inherited_tcn"

    def __init__(self):
        super().__init__()
        layers: list[nn.Module] = []
        current = 2
        for index, stride in enumerate((2, 5)):
            target = 12 * min(2**index, 4)
            kernel = 11 if index == 0 else 7
            layers.extend(
                [
                    nn.Conv1d(
                        current,
                        target,
                        kernel_size=kernel,
                        stride=stride,
                        padding=kernel // 2,
                    ),
                    nn.GroupNorm(4, target),
                    nn.GELU(),
                ]
            )
            current = target
        layers.extend(ResidualBlock(current, 2 ** (index % 4)) for index in range(3))
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Conv1d(current, 1, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(value))


def receptive_field_seconds() -> float:
    # Exact receptive field for strides (2,5), kernels (11,7), and residual
    # dilations (1,2,4), including two convolutions per residual block.
    receptive_field = 1
    jump = 1
    for kernel, stride in ((11, 2), (7, 5)):
        receptive_field += (kernel - 1) * jump
        jump *= stride
    for dilation in (1, 2, 4):
        receptive_field += (5 - 1) * dilation * jump
    return receptive_field / 100.0
