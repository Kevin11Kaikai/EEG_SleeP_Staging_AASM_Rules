from __future__ import annotations

import csv
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
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


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


def raw_intervals(mask: np.ndarray, rate_hz: float) -> list[tuple[float, float]]:
    values = np.asarray(mask, dtype=bool)
    edges = np.diff(np.pad(values.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(edges == 1) / rate_hz
    ends = np.flatnonzero(edges == -1) / rate_hz
    return [(float(a), float(b)) for a, b in zip(starts, ends)]


def eventize(
    probability: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    rate_hz: float,
    minimum_seconds: float,
    maximum_seconds: float,
    merge_gap_seconds: float,
) -> list[tuple[float, float]]:
    """Merge threshold fragments first, then apply the physiological duration gate."""
    mask = (np.asarray(probability) >= threshold) & np.asarray(valid, dtype=bool)
    edges = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    fragments = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    merged: list[tuple[int, int]] = []
    maximum_gap = int(round(merge_gap_seconds * rate_hz))
    for start, end in fragments:
        if merged and start - merged[-1][1] <= maximum_gap:
            # Never merge across invalid samples.
            if np.asarray(valid, dtype=bool)[merged[-1][1] : start].all():
                merged[-1] = (merged[-1][0], end)
                continue
        merged.append((start, end))
    return [
        (start / rate_hz, end / rate_hz)
        for start, end in merged
        if minimum_seconds <= (end - start) / rate_hz <= maximum_seconds
    ]


def interval_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return overlap / union if union > 0 else 0.0


def match_intervals(
    reference: list[tuple[float, float]],
    predicted: list[tuple[float, float]],
    minimum_iou: float = 0.2,
) -> dict[str, Any]:
    candidates: list[tuple[float, int, int]] = []
    for i, truth in enumerate(reference):
        for j, guess in enumerate(predicted):
            if guess[1] <= truth[0] or guess[0] >= truth[1]:
                continue
            score = interval_iou(truth, guess)
            if score >= minimum_iou:
                candidates.append((score, i, j))
    used_truth: set[int] = set()
    used_guess: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for score, i, j in sorted(candidates, reverse=True):
        if i in used_truth or j in used_guess:
            continue
        used_truth.add(i)
        used_guess.add(j)
        matched.append((i, j, score))
    tp = len(matched)
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
        "mean_matched_iou": (
            float(np.mean([row[2] for row in matched])) if matched else 0.0
        ),
        "matches": matched,
    }


def reference_intervals(
    target: np.ndarray, valid: np.ndarray, rate_hz: float
) -> list[tuple[float, float]]:
    return raw_intervals(
        np.asarray(target, dtype=bool) & np.asarray(valid, dtype=bool), rate_hz
    )
