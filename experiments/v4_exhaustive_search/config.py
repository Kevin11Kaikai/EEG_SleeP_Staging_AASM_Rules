from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from experiments.v4_subset_search.config import (
    ALIGNED_REFERENCE,
    AR_MASK,
    BASE_BANK_ROOT,
    FROZEN_CORE,
    FULL_MASK,
    HANDLERS,
    RECORDS,
    SOURCE_BANK_ROOT,
    SOURCE_RUN,
    SOURCE_SEAL,
    STAGE_NAMES,
    STAGE_TO_CODE,
    ArmSpec,
)


WORKTREE = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("AASM_V4_RESULTS_ROOT", WORKTREE / "results" / "local")).resolve()
SOURCE_HYBRID_RUN = Path(
    os.environ.get(
        "AASM_V4_SUBSET_RUN",
        RESULTS / "phase2_hybrid_subset_search_20261118T131657-0500",
    )
).resolve()
SOURCE_CACHE = Path(
    os.environ.get("AASM_V4_SUBSET_CACHE", SOURCE_HYBRID_RUN / "cache" / "compact_predictions.sqlite")
).resolve()


@dataclass(frozen=True)
class ExtendedLimits:
    max_wall_hours: float = 16.0
    replay_wall_hours: float = 13.0
    worker_count: int = 14
    masks_per_chunk: int = 512
    shards_per_record: int = 8
    restart_every_chunks: int = 4
    benchmark_masks: int = 512
    exhaustive_rate_threshold_per_hour: float = 2350.0
    seed: int = 2026
    coverage_floor: float = 0.99
    epsilon: float = 1e-6


def deterministic_mask_order() -> tuple[int, ...]:
    """Label-blind anytime order: complete low-cardinality shells first."""
    return tuple(sorted(range(FULL_MASK + 1), key=lambda mask: (mask.bit_count(), mask)))


def validate_paths() -> None:
    required = (
        SOURCE_HYBRID_RUN,
        SOURCE_CACHE,
        SOURCE_RUN,
        SOURCE_BANK_ROOT,
        SOURCE_SEAL,
        BASE_BANK_ROOT,
        ALIGNED_REFERENCE,
        FROZEN_CORE,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen input(s): {missing}")
    if len(RECORDS) != 10:
        raise RuntimeError("record allow-list must contain exactly ten frozen records")
    if len(HANDLERS) != 15:
        raise RuntimeError("handler contract changed")
