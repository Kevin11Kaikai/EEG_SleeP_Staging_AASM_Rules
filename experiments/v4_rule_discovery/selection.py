from __future__ import annotations

from typing import Any

import numpy as np

from .cache import PredictionCache
from .config import SEED, STAGES, RuleConfig
from .metrics import ReferenceData, candidate_summary, choose_config, metrics_from_confusion


def select_discovery_candidate(
    cache: PredictionCache,
    configs: tuple[RuleConfig, ...],
    reference: ReferenceData,
    records: tuple[str, ...],
) -> dict[str, Any]:
    summaries = [candidate_summary(reference, cache, config, records) for config in configs]
    winner, reason = choose_config(summaries)
    selection = {
        "selection_scope": "HMC_FROZEN_DISCOVERY_PARTITION",
        "selection_records": list(records),
        "winner_config_id": winner,
        "selection_reason": reason,
        "candidate_count": len(configs),
        "candidate_summaries": summaries,
        "confirmation_labels_accessed": False,
    }
    cache.freeze_winner("DISCOVERY", None, winner, selection)
    return selection


def evaluate_frozen_candidate(
    cache: PredictionCache,
    config_id: str,
    reference: ReferenceData,
    records: tuple[str, ...],
    bootstrap_samples: int = 10000,
) -> dict[str, Any]:
    candidate_matrices = [reference.matrix(cache, config_id, record) for record in records]
    baseline_matrices = [reference.matrix(cache, "C000", record) for record in records]
    candidate = metrics_from_confusion(sum(candidate_matrices, np.zeros((6, 6), dtype=np.int64)))
    baseline = metrics_from_confusion(sum(baseline_matrices, np.zeros((6, 6), dtype=np.int64)))
    per_night = []
    for index, record in enumerate(records):
        current = metrics_from_confusion(candidate_matrices[index])
        base = metrics_from_confusion(baseline_matrices[index])
        per_night.append(
            {
                "record_id": record,
                "candidate_accuracy": current["accuracy"],
                "baseline_accuracy": base["accuracy"],
                "delta_accuracy": current["accuracy"] - base["accuracy"],
                "candidate_macro_f1": current["macro_f1"],
                "baseline_macro_f1": base["macro_f1"],
                "delta_macro_f1": current["macro_f1"] - base["macro_f1"],
                "candidate_coverage": current["coverage"],
                "baseline_coverage": base["coverage"],
                "delta_coverage": current["coverage"] - base["coverage"],
            }
        )
    rng = np.random.default_rng(SEED)
    bootstrap = {key: [] for key in ("accuracy", "macro_f1", "coverage")}
    for _ in range(bootstrap_samples):
        indices = rng.integers(0, len(records), size=len(records))
        left = metrics_from_confusion(sum((baseline_matrices[index] for index in indices), np.zeros((6, 6), dtype=np.int64)))
        right = metrics_from_confusion(sum((candidate_matrices[index] for index in indices), np.zeros((6, 6), dtype=np.int64)))
        for key in bootstrap:
            bootstrap[key].append(right[key] - left[key])
    delta = {key: candidate[key] - baseline[key] for key in ("accuracy", "macro_f1", "coverage")}
    stage_deltas = {
        stage: candidate["per_stage"][stage]["f1"] - baseline["per_stage"][stage]["f1"]
        for stage in STAGES[:5]
    }
    return {
        "config_id": config_id,
        "records": list(records),
        "candidate": candidate,
        "baseline": baseline,
        "delta": delta,
        "stage_f1_deltas": stage_deltas,
        "paired": reference.paired(cache, config_id, records),
        "per_night": per_night,
        "record_equal": {
            key: float(np.mean([row[f"delta_{key}"] for row in per_night]))
            for key in ("accuracy", "macro_f1", "coverage")
        },
        "positive_night_fraction": {
            key: float(np.mean([row[f"delta_{key}"] > 0 for row in per_night]))
            for key in ("accuracy", "macro_f1")
        },
        "bootstrap": {
            key: {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
            }
            for key, values in bootstrap.items()
        },
    }
