from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    family: str
    input_channels: int
    output_heads: tuple[str, ...]
    input_rate_hz: int
    output_rate_hz: int
    strides: tuple[int, ...]
    base_channels: int
    residual_blocks: int

    @property
    def total_stride(self) -> int:
        product = 1
        for value in self.strides:
            product *= value
        return product

    def to_dict(self) -> dict:
        value = asdict(self)
        value["output_heads"] = list(self.output_heads)
        value["strides"] = list(self.strides)
        return value


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


class TemporalHandlerNet(nn.Module):
    """Small temporal encoder with a genuine dense task head."""

    def __init__(self, spec: ModelSpec):
        super().__init__()
        if spec.input_rate_hz // spec.output_rate_hz != spec.total_stride:
            raise ValueError("Convolution strides do not match temporal output contract")
        channels = spec.base_channels
        layers: list[nn.Module] = []
        current = spec.input_channels
        for index, stride in enumerate(spec.strides):
            target = channels * min(2 ** index, 4)
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
        layers.extend(
            ResidualBlock(current, 2 ** (block % 4))
            for block in range(spec.residual_blocks)
        )
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Conv1d(current, len(spec.output_heads), kernel_size=1)
        self.spec = spec

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(value))


FAMILY_SPECS = {
    "eeg_background": ModelSpec(
        family="eeg_background",
        input_channels=2,
        output_heads=("E05", "E08", "E12"),
        input_rate_hz=100,
        output_rate_hz=1,
        strides=(5, 4, 5),
        base_channels=12,
        residual_blocks=3,
    ),
    "eog_events": ModelSpec(
        family="eog_events",
        input_channels=2,
        output_heads=("E06", "E07"),
        input_rate_hz=100,
        output_rate_hz=10,
        strides=(2, 5),
        base_channels=12,
        residual_blocks=3,
    ),
    "chin_emg": ModelSpec(
        family="chin_emg",
        input_channels=1,
        output_heads=("E09", "E10", "E11"),
        input_rate_hz=100,
        output_rate_hz=2,
        strides=(5, 5, 2),
        base_channels=12,
        residual_blocks=3,
    ),
    "major_movement": ModelSpec(
        family="major_movement",
        input_channels=3,
        output_heads=("E13",),
        input_rate_hz=100,
        output_rate_hz=1,
        strides=(5, 4, 5),
        base_channels=16,
        residual_blocks=3,
    ),
}


def build_model(family: str) -> TemporalHandlerNet:
    return TemporalHandlerNet(FAMILY_SPECS[family])
