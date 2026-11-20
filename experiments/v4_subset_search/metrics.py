from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .config import ALIGNED_REFERENCE, RECORDS, STAGE_NAMES, STAGE_TO_CODE, SearchLimits
from .storage import CacheDB


def confusion_from_codes(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    if truth.shape != prediction.shape:
        raise ValueError(f"truth/prediction length mismatch: {truth.shape}/{prediction.shape}")
    if np.any((truth < 0) | (truth > 4)) or np.any((prediction < 0) | (prediction > 5)):
        raise ValueError("stage code outside frozen six-stage contract")
    matrix = np.zeros((6, 6), dtype=np.int64)
    np.add.at(matrix, (truth, prediction), 1)
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (6, 6):
        raise ValueError(matrix.shape)
    n = int(matrix.sum())
    if n == 0:
        raise ValueError("no evaluable epochs")
    diagonal = matrix.diagonal()
    correct = int(np.trace(matrix))
    predicted = matrix.sum(axis=0)
    support = matrix.sum(axis=1)
    precision = np.divide(diagonal, predicted, out=np.zeros(6), where=predicted > 0)
    recall = np.divide(diagonal, support, out=np.zeros(6), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(6), where=precision + recall > 0)
    accepted = n - int(predicted[5])
    conditional = correct / accepted if accepted else None
    expected = float(predicted @ support) / n**2
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "macro_f1": float(f1[:5].mean()),
        "coverage": accepted / n,
        "abstention_rate": int(predicted[5]) / n,
        "conditional_accuracy": conditional,
        "accepted_risk": None if conditional is None else 1 - conditional,
        "kappa": (correct / n - expected) / (1 - expected) if expected < 1 else None,
        "per_stage": {
            stage: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, stage in enumerate(STAGE_NAMES[:5])
        },
        "confusion": matrix.tolist(),
    }


@dataclass(frozen=True)
class FoldAccess:
    fold_id: str
    selection_records: tuple[str, ...]
    inner_validation_records: tuple[tuple[str, ...], ...]
    heldout_record: str | None

    def __post_init__(self) -> None:
        selection = set(self.selection_records)
        if len(selection) != len(self.selection_records) or not selection or not selection <= set(RECORDS):
            raise ValueError(f"invalid selection records: {self.fold_id}")
        if self.heldout_record is not None:
            if self.heldout_record not in RECORDS or selection != set(RECORDS) - {self.heldout_record}:
                raise ValueError(f"outer heldout/selection contract changed: {self.fold_id}")
        elif selection != set(RECORDS):
            raise ValueError(f"full-development split must contain all records: {self.fold_id}")
        for fold in self.inner_validation_records:
            if not fold or not set(fold) <= selection:
                raise ValueError(f"invalid inner fold: {self.fold_id}/{fold}")
        flattened = [record for fold in self.inner_validation_records for record in fold]
        if sorted(flattened) != sorted(self.selection_records):
            raise ValueError(f"inner validation folds do not partition selection: {self.fold_id}")


def build_nested_splits(seed: int = 2026) -> tuple[FoldAccess, ...]:
    output = []
    records = tuple(RECORDS)
    for outer_index, heldout in enumerate(records):
        selection = tuple(record for record in records if record != heldout)
        splitter = KFold(n_splits=3, shuffle=True, random_state=seed + outer_index)
        inner = []
        values = np.asarray(selection)
        for _, validation_indices in splitter.split(values):
            inner.append(tuple(values[validation_indices].tolist()))
        output.append(FoldAccess(f"OUTER_{heldout}", selection, tuple(inner), heldout))
    return tuple(output)


def build_full_development_split(seed: int = 2026) -> FoldAccess:
    values = np.asarray(RECORDS)
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    inner = tuple(tuple(values[index].tolist()) for _, index in splitter.split(values))
    return FoldAccess("FULL_DEV", tuple(RECORDS), inner, None)


class MetricOracle:
    """Reference-aware scorer. Replay workers never receive this object or reference data."""

    def __init__(self, cache: CacheDB, limits: SearchLimits) -> None:
        self.cache = cache
        self.limits = limits
        frame = pd.read_csv(ALIGNED_REFERENCE)
        if set(frame.record_id.unique()) != set(RECORDS):
            raise RuntimeError("reference record set differs from explicit allow-list")
        self.reference: dict[str, dict[str, np.ndarray]] = {}
        for record in RECORDS:
            rows = frame.loc[frame.record_id == record].sort_values("epoch_index")
            if rows.epoch_index.tolist() != list(range(len(rows))):
                raise RuntimeError(f"reference index mismatch: {record}")
            truth = np.asarray([STAGE_TO_CODE.get(value, 255) for value in rows.true_stage], dtype=np.uint8)
            primary = rows.in_primary.to_numpy(dtype=bool)
            full = rows.in_full_record.to_numpy(dtype=bool)
            if np.any(truth[primary] > 4) or np.any(truth[full] > 4):
                raise RuntimeError(f"non-stage reference included in scored mask: {record}")
            self.reference[record] = {
                "truth": truth,
                "primary": primary,
                "full": full,
            }
        if sum(int(self.reference[record]["primary"].sum()) for record in RECORDS) != 8902:
            raise RuntimeError("primary mask changed")
        if sum(int(self.reference[record]["full"].sum()) for record in RECORDS) != 9211:
            raise RuntimeError("full-valid mask changed")
        if any(np.any(ref["primary"] & ~ref["full"]) for ref in self.reference.values()):
            raise RuntimeError("primary mask is not a subset of full-valid")
        self._matrix_cache: dict[tuple[int, str, str], np.ndarray] = {}

    def _matrix(self, mask: int, record: str, scope: str = "primary") -> np.ndarray:
        key = (mask, record, scope)
        if not self.cache.arm_complete(mask):
            raise RuntimeError(f"cannot score incomplete arm: {mask}")
        cached = self._matrix_cache.get(key)
        if cached is not None:
            return cached.copy()
        if scope not in {"primary", "full"}:
            raise ValueError(scope)
        prediction = self.cache.stage_codes(mask, record)
        reference = self.reference[record]
        if len(prediction) != len(reference["truth"]):
            raise RuntimeError(f"prediction/reference mismatch: {mask}/{record}")
        valid = reference[scope]
        matrix = confusion_from_codes(reference["truth"][valid], prediction[valid])
        self._matrix_cache[key] = matrix
        return matrix.copy()

    def _score(self, mask: int, records: Iterable[str], scope: str = "primary") -> dict[str, Any]:
        selected = tuple(records)
        if not selected or not set(selected) <= set(RECORDS):
            raise PermissionError(f"invalid scoring record set: {selected}")
        matrix = sum((self._matrix(mask, record, scope) for record in selected), np.zeros((6, 6), dtype=np.int64))
        result = metrics_from_confusion(matrix)
        result["records"] = list(selected)
        return result

    def search_score(self, mask: int, access: FoldAccess, records: Iterable[str] | None = None) -> dict[str, Any]:
        if access.heldout_record is None and not self.cache.get_meta("report_unlocked", False):
            raise PermissionError("TEST_LABEL_ACCESS_DENIED: full-development search is still locked")
        selected = access.selection_records if records is None else tuple(records)
        if not set(selected) <= set(access.selection_records):
            raise PermissionError(f"TEST_LABEL_ACCESS_DENIED: {access.fold_id}/{selected}")
        return self._score(mask, selected)

    def heldout_score(self, mask: int, access: FoldAccess) -> dict[str, Any]:
        if access.heldout_record is None:
            raise PermissionError("full-development search has no heldout score")
        winner = self.cache.winner(access.fold_id)
        if winner is None or winner[0] != mask or winner[1] != access.heldout_record or winner[2] not in {"WINNER_FROZEN", "HELDOUT_SCORED"}:
            raise PermissionError(f"TEST_LABEL_ACCESS_DENIED: {access.fold_id}")
        result = self._score(mask, (access.heldout_record,))
        if winner[2] == "WINNER_FROZEN":
            self.cache.mark_heldout_scored(access.fold_id)
        return result

    def unrestricted_report_score(self, mask: int, records: Iterable[str] = RECORDS) -> dict[str, Any]:
        if not self.cache.get_meta("report_unlocked", False):
            raise PermissionError("REPORT_ACCESS_DENIED: nested outer winners not all frozen")
        return self._score(mask, records)

    def summary(self, mask: int, access: FoldAccess, baseline_mask: int = 0) -> dict[str, Any]:
        pooled = self.search_score(mask, access)
        base = self.search_score(baseline_mask, access)
        fold_metrics = [self.search_score(mask, access, records) for records in access.inner_validation_records]
        base_folds = [self.search_score(baseline_mask, access, records) for records in access.inner_validation_records]
        accuracy_deltas = [row["accuracy"] - ref["accuracy"] for row, ref in zip(fold_metrics, base_folds)]
        macro_deltas = [row["macro_f1"] - ref["macro_f1"] for row, ref in zip(fold_metrics, base_folds)]
        eligible = (
            pooled["coverage"] + self.limits.epsilon >= self.limits.coverage_floor
            and pooled["accuracy"] + self.limits.epsilon >= base["accuracy"]
            and pooled["macro_f1"] + self.limits.epsilon >= base["macro_f1"]
        )
        return {
            "mask": mask,
            "accuracy": pooled["accuracy"],
            "macro_f1": pooled["macro_f1"],
            "coverage": pooled["coverage"],
            "delta_accuracy": pooled["accuracy"] - base["accuracy"],
            "delta_macro_f1": pooled["macro_f1"] - base["macro_f1"],
            "mean_inner_accuracy_delta": float(np.mean(accuracy_deltas)),
            "mean_inner_macro_delta": float(np.mean(macro_deltas)),
            "worst_inner_accuracy_delta": float(np.min(accuracy_deltas)),
            "eligible": bool(eligible),
            "inner_accuracy_deltas": accuracy_deltas,
            "inner_macro_deltas": macro_deltas,
        }

    def paired_counts(self, mask: int, comparator: int = 0, records: Iterable[str] = RECORDS) -> dict[str, int]:
        if not self.cache.get_meta("report_unlocked", False):
            raise PermissionError("REPORT_ACCESS_DENIED: nested outer winners not all frozen")
        corrected = introduced = both_correct = both_incorrect = 0
        for record in records:
            ref = self.reference[record]
            valid = ref["primary"]
            truth = ref["truth"][valid]
            left = self.cache.stage_codes(comparator, record)[valid]
            right = self.cache.stage_codes(mask, record)[valid]
            left_correct = left == truth
            right_correct = right == truth
            corrected += int(np.sum(~left_correct & right_correct))
            introduced += int(np.sum(left_correct & ~right_correct))
            both_correct += int(np.sum(left_correct & right_correct))
            both_incorrect += int(np.sum(~left_correct & ~right_correct))
        return {
            "corrected": corrected,
            "introduced": introduced,
            "net_correct": corrected - introduced,
            "both_correct": both_correct,
            "both_incorrect": both_incorrect,
        }
