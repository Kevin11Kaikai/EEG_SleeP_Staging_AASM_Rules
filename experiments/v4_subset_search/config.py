from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

from experiments.v4_36arm_ablation.config import (
    EVENT_TYPE,
    FROZEN_CORE,
    HANDLERS,
    RECORDS,
    RESULTS,
)


WORKTREE = Path(__file__).resolve().parents[2]
R1_ROOT = FROZEN_CORE / "results" / "development_r1"
ALIGNED_REFERENCE = R1_ROOT / "evaluation" / "aligned_development.csv.gz"
SOURCE_RUN = Path(
    os.environ.get(
        "AASM_V4_PHASE2_RUN",
        RESULTS / "aasm_v4_phase2_36arm_20261117T205438-0500",
    )
).resolve()
BASE_BANK_ROOT = SOURCE_RUN / "event_banks" / "B000"
SOURCE_BANK_ROOT = SOURCE_RUN / "source_banks"
SOURCE_SEAL = SOURCE_RUN / "neural_inference" / "prediction_seal.json"

FULL_MASK = (1 << len(HANDLERS)) - 1
AR_BIT = HANDLERS.index("AR")
AR_MASK = 1 << AR_BIT
STAGE_NAMES = ("W", "N1", "N2", "N3", "R", "UNDETERMINED")
STAGE_TO_CODE = {stage: index for index, stage in enumerate(STAGE_NAMES)}


@dataclass(frozen=True, order=True)
class ArmSpec:
    mask: int

    def __post_init__(self) -> None:
        if not 0 <= self.mask <= FULL_MASK:
            raise ValueError(f"mask outside 15-bit space: {self.mask}")

    @property
    def arm_id(self) -> str:
        return f"M_{self.mask:04X}"

    @property
    def handlers(self) -> tuple[str, ...]:
        return tuple(handler for bit, handler in enumerate(HANDLERS) if self.mask & (1 << bit))

    @property
    def cardinality(self) -> int:
        return self.mask.bit_count()

    @property
    def contains_ar(self) -> bool:
        return bool(self.mask & AR_MASK)


def mask_for_handlers(handlers: Iterable[str]) -> int:
    selected = tuple(handlers)
    if len(selected) != len(set(selected)) or not set(selected) <= set(HANDLERS):
        raise ValueError(f"invalid handler set: {selected}")
    return sum(1 << HANDLERS.index(handler) for handler in selected)


def masks_of_size(size: int) -> tuple[int, ...]:
    if not 0 <= size <= len(HANDLERS):
        raise ValueError(size)
    return tuple(mask_for_handlers(items) for items in combinations(HANDLERS, size))


def mandatory_masks() -> tuple[int, ...]:
    return tuple(mask for size in range(4) for mask in masks_of_size(size))


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def hamming_neighbors(mask: int, radius: int = 2, include_center: bool = False) -> tuple[int, ...]:
    if radius not in {1, 2}:
        raise ValueError("only radius one or two is supported")
    output = {mask} if include_center else set()
    bits = tuple(range(len(HANDLERS)))
    for size in range(1, radius + 1):
        for changed in combinations(bits, size):
            candidate = mask
            for bit in changed:
                candidate ^= 1 << bit
            output.add(candidate)
    return tuple(sorted(output))


@dataclass(frozen=True)
class SearchLimits:
    max_wall_hours: float = 12.0
    max_unique_arms: int = 7300
    search_fraction: float = 0.8
    worker_count: int = 10
    replay_batch_size: int = 32
    adaptive_rounds: int = 12
    proposals_per_round: int = 32
    beam_width: int = 4
    seed: int = 2026
    coverage_floor: float = 0.99
    epsilon: float = 1e-6


def validate_frozen_paths() -> None:
    required = (SOURCE_RUN, SOURCE_BANK_ROOT, SOURCE_SEAL, BASE_BANK_ROOT, ALIGNED_REFERENCE, FROZEN_CORE)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen Hybrid Search input(s): {missing}")
    if tuple(f"SN{i:03d}" for i in range(1, 11)) != RECORDS:
        raise RuntimeError("record allow-list changed")
    if len(HANDLERS) != 15 or len(set(EVENT_TYPE.values())) != 15:
        raise RuntimeError("fifteen-handler contract changed")
