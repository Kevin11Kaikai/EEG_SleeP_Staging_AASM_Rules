from __future__ import annotations

import hashlib
import sqlite3

import numpy as np
import pytest

from experiments.v4_exhaustive_search.config import (
    ALIGNED_REFERENCE,
    FULL_MASK,
    SOURCE_CACHE,
    SOURCE_HYBRID_RUN,
    deterministic_mask_order,
)
from experiments.v4_exhaustive_search.fast_replay import (
    clone_label_blind_cache,
    sha256,
)
from experiments.v4_exhaustive_search.run import (
    _load_safe_warm_masks,
    _next_new_chunk,
)
from experiments.v4_exhaustive_search.selection import (
    ReferenceData,
    load_confusion_cube,
    metric_dict,
)
from experiments.v4_subset_search.replay import build_cache_identity


def _require_private_artifacts(*paths) -> None:
    """Skip sealed replay checks when the non-redistributable inputs are absent."""

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        pytest.skip("requires sealed private replay artifacts: " + ", ".join(missing))


def test_deterministic_order_is_complete_and_shell_monotone() -> None:
    order = deterministic_mask_order()
    assert len(order) == FULL_MASK + 1
    assert set(order) == set(range(FULL_MASK + 1))
    cardinalities = [mask.bit_count() for mask in order]
    assert cardinalities == sorted(cardinalities)


def test_next_chunk_skips_cached_without_breaking_prefix() -> None:
    order = deterministic_mask_order()
    cached = set(order[:20:2])
    chunk, next_index = _next_new_chunk(order, 0, cached, 8)
    assert len(chunk) == 8
    assert not (set(chunk) & cached)
    assert set(order[:next_index]) <= cached | set(chunk)


def test_old_warm_start_is_fold_isolated() -> None:
    _require_private_artifacts(SOURCE_HYBRID_RUN)
    masks, audit = _load_safe_warm_masks()
    assert audit["status"] == "PASS"
    assert audit["heldout_access_failures"] == []
    assert len(masks) == 10
    assert all(900 <= len(values) <= 1000 for values in masks.values())


def test_b000_confusion_and_metrics_reproduce() -> None:
    _require_private_artifacts(ALIGNED_REFERENCE, SOURCE_CACHE)
    reference = ReferenceData()
    masks, cube = load_confusion_cube(SOURCE_CACHE, {0}, reference)
    assert masks.tolist() == [0]
    matrix = cube[0].sum(axis=0, dtype=np.int64)
    metrics = metric_dict(matrix)
    assert metrics["n"] == 8902
    assert metrics["correct"] == 4629
    assert abs(metrics["accuracy"] - 0.5199955066277241) < 1e-15
    assert abs(metrics["macro_f1"] - 0.40081488140174376) < 1e-15


def test_clone_strips_search_history_and_preserves_source(tmp_path) -> None:
    _require_private_artifacts(SOURCE_CACHE)
    source_hash = sha256(SOURCE_CACHE)
    destination = tmp_path / "compact_predictions.sqlite"
    config_path = (
        SOURCE_CACHE.parents[4]
        / "experiments"
        / "v4_exhaustive_search"
        / "configs"
        / "main.yaml"
    )
    cache_key, _ = build_cache_identity(config_path)
    result = clone_label_blind_cache(SOURCE_CACHE, destination, cache_key)
    assert result["status"] == "CLONED_LABEL_BLIND_ONLY"
    connection = sqlite3.connect(destination)
    try:
        assert connection.execute("SELECT COUNT(*) FROM search_events").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM fold_winners").fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM arms WHERE status='PASS'"
        ).fetchone()[0] == 2262
        assert connection.execute(
            "SELECT COUNT(*) FROM arm_records WHERE status='PASS'"
        ).fetchone()[0] == 22620
    finally:
        connection.close()
    assert sha256(SOURCE_CACHE) == source_hash
