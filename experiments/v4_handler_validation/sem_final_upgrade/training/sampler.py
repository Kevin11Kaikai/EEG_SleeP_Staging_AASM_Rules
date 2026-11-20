from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from ..data.audit import RecordData


@dataclass(frozen=True)
class WindowRef:
    record_id: str
    input_start: int
    output_start: int
    positive: bool
    hard_negative: bool


class SEMWindowDataset(Dataset):
    def __init__(
        self,
        records: dict[str, RecordData],
        record_ids: list[str],
        window_seconds: int,
        stride_seconds: int,
        *,
        shuffled_targets: bool = False,
        seed: int = 0,
    ):
        self.records = records
        self.input_length = window_seconds * 100
        self.output_length = window_seconds * 10
        self.refs: list[WindowRef] = []
        self.shuffled_targets = shuffled_targets
        for record_id in record_ids:
            record = records[record_id]
            duration = min(record.signal.shape[-1] // 100, len(record.target) // 10)
            starts = list(range(0, max(1, duration - window_seconds + 1), stride_seconds))
            final = max(0, duration - window_seconds)
            if not starts or starts[-1] != final:
                starts.append(final)
            for start in starts:
                output_start = start * 10
                right = output_start + self.output_length
                valid = record.scoring_mask[output_start:right]
                if not valid.any():
                    continue
                self.refs.append(
                    WindowRef(
                        record_id=record_id,
                        input_start=start * 100,
                        output_start=output_start,
                        positive=bool(record.target[output_start:right].any()),
                        hard_negative=bool(record.hard_negative_mask[output_start:right].any()),
                    )
                )
        if not self.refs:
            raise ValueError("No valid SEM training windows")
        self.permutation = np.random.default_rng(seed).permutation(len(self.refs))

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int):
        ref = self.refs[index]
        signal_record = self.records[ref.record_id]
        signal = signal_record.signal[
            :, ref.input_start : ref.input_start + self.input_length
        ]
        target_ref = self.refs[int(self.permutation[index])] if self.shuffled_targets else ref
        target_record = self.records[target_ref.record_id]
        left = target_ref.output_start
        right = left + self.output_length
        return (
            torch.from_numpy(np.ascontiguousarray(signal)).float(),
            torch.from_numpy(target_record.target[left:right][None].copy()).float(),
            torch.from_numpy(target_record.scoring_mask[left:right][None].copy()).float(),
            ref.record_id,
            ref.output_start,
        )


def build_weighted_sampler(
    dataset: SEMWindowDataset,
    positive_weight: float,
    hard_negative_weight: float,
    seed: int,
) -> WeightedRandomSampler:
    weights = []
    for index, ref in enumerate(dataset.refs):
        if dataset.shuffled_targets:
            ref = dataset.refs[int(dataset.permutation[index])]
        if ref.positive:
            weights.append(positive_weight)
        elif ref.hard_negative:
            weights.append(hard_negative_weight)
        else:
            weights.append(1.0)
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
