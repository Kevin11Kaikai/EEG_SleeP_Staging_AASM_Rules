from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, value in sorted(model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def state_l2_delta(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> float:
    total = 0.0
    for key, value in before.items():
        difference = after[key].detach().cpu().float() - value.detach().cpu().float()
        total += float(torch.sum(difference * difference))
    return math.sqrt(total)


def intervals_from_mask(
    mask: np.ndarray,
    rate_hz: float,
    *,
    minimum_seconds: float = 0.0,
    merge_gap_seconds: float = 0.0,
) -> list[tuple[float, float]]:
    values = np.asarray(mask, dtype=bool)
    edges = np.diff(np.pad(values.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(edges == 1) / rate_hz
    ends = np.flatnonzero(edges == -1) / rate_hz
    raw = [
        (float(start), float(end))
        for start, end in zip(starts, ends)
        if end - start >= minimum_seconds
    ]
    if not raw or merge_gap_seconds <= 0:
        return raw
    merged = [raw[0]]
    for start, end in raw[1:]:
        if start - merged[-1][1] <= merge_gap_seconds:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def rasterize(
    intervals: Iterable[tuple[float, float]], length: int, rate_hz: float
) -> np.ndarray:
    output = np.zeros(length, dtype=np.uint8)
    for onset, offset in intervals:
        left = max(0, int(math.floor(onset * rate_hz)))
        right = min(length, int(math.ceil(offset * rate_hz)))
        if right > left:
            output[left:right] = 1
    return output


def binary_counts(
    target: np.ndarray, prediction: np.ndarray, valid: np.ndarray
) -> dict[str, int | float]:
    truth = np.asarray(target, dtype=bool)
    guess = np.asarray(prediction, dtype=bool)
    mask = np.asarray(valid, dtype=bool)
    tp = int(np.sum(truth & guess & mask))
    fp = int(np.sum(~truth & guess & mask))
    fn = int(np.sum(truth & ~guess & mask))
    tn = int(np.sum(~truth & ~guess & mask))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": tp / union if union else 0.0,
        "valid_samples": int(mask.sum()),
    }


def interval_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return overlap / union if union > 0 else 0.0


def match_intervals(
    reference: list[tuple[float, float]],
    predicted: list[tuple[float, float]],
    minimum_iou: float = 0.2,
) -> dict[str, int | float]:
    # Intervals emitted by ``intervals_from_mask`` are ordered and disjoint, but
    # retain original indices here so this evaluator is also correct for callers
    # that supply an arbitrary order.  IoU can only exceed zero for temporally
    # overlapping intervals; the sweep avoids the quadratic all-pairs scan that
    # becomes prohibitive for continuous evidence such as LAMF.
    ordered_reference = sorted(enumerate(reference), key=lambda row: row[1][0])
    ordered_predicted = sorted(enumerate(predicted), key=lambda row: row[1][0])
    candidates: list[tuple[float, int, int]] = []
    first_possible = 0
    for i, truth in ordered_reference:
        while (
            first_possible < len(ordered_predicted)
            and ordered_predicted[first_possible][1][1] <= truth[0]
        ):
            first_possible += 1
        candidate_index = first_possible
        while (
            candidate_index < len(ordered_predicted)
            and ordered_predicted[candidate_index][1][0] < truth[1]
        ):
            j, guess = ordered_predicted[candidate_index]
            score = interval_iou(truth, guess)
            if score >= minimum_iou:
                candidates.append((score, i, j))
            candidate_index += 1
    matched_truth: set[int] = set()
    matched_guess: set[int] = set()
    scores: list[float] = []
    for score, i, j in sorted(candidates, reverse=True):
        if i not in matched_truth and j not in matched_guess:
            matched_truth.add(i)
            matched_guess.add(j)
            scores.append(score)
    tp = len(scores)
    fp = len(predicted) - tp
    fn = len(reference) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": float(np.mean(scores)) if scores else 0.0,
    }


def artifact_manifest(root: str | Path) -> dict[str, Any]:
    base = Path(root)
    rows = []
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        rows.append(
            {
                "path": path.relative_to(base).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {
        "schema_version": 1,
        "generated_at": now_iso(),
        "root": str(base.resolve()),
        "artifact_count": len(rows),
        "artifacts": rows,
        "manifest_identity": canonical_hash(rows),
    }
