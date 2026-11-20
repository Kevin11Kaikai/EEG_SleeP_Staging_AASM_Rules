from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt
from torch.utils.data import Dataset

from .edf import read_channels, read_header
from .models import FAMILY_SPECS


GROUPS = {
    "G01": ("kc_excerpt1", "rem_excerpt1", "rem_excerpt2"),
    "G02": ("kc_excerpt2", "rem_excerpt3", "rem_excerpt4"),
    "G03": ("kc_excerpt9", "rem_excerpt5", "rem_excerpt6"),
    "G04": ("kc_excerpt4", "rem_excerpt7", "rem_excerpt8"),
    "G05": ("kc_excerpt8", "rem_excerpt9"),
    "G06": ("kc_excerpt3",),
    "G07": ("kc_excerpt5",),
    "G08": ("kc_excerpt6",),
    "G09": ("kc_excerpt7",),
    "G10": ("kc_excerpt10",),
}

SPLIT_GROUPS = {
    "train": ("G01", "G02", "G03", "G06", "G07", "G08"),
    "validation": ("G04", "G09"),
    "test": ("G05", "G10"),
}

ROLE_CHANNEL_VARIANTS = {
    "f": ("FP1-A2", "FP2-A1", "EEG Fpz-Cz"),
    "c": ("CZ-A1", "EEG Fpz-Cz"),
    "o": ("O1-A2", "O2-A1", "EEG Pz-Oz"),
    "l": ("EOG1", "EOG1-A2"),
    "r": ("EOG2", "EOG2-A2"),
    "m": ("EMG1",),
}

FAMILY_ROLES = {
    "eeg_background": ("c", "o"),
    "eog_events": ("l", "r"),
    "chin_emg": ("m",),
    "major_movement": ("f", "c", "m"),
}

HANDLER_ROLES = {
    "E05": ("o",),
    "E06": ("l", "r"),
    "E07": ("l", "r"),
    "E08": ("c",),
    "E09": ("m",),
    "E10": ("m",),
    "E11": ("m",),
    "E12": ("c", "o"),
    "E13": ("f", "c", "m"),
}

CHANNEL_KINDS = {
    "FP1-A2": "eeg",
    "FP2-A1": "eeg",
    "CZ-A1": "eeg",
    "O1-A2": "eeg",
    "O2-A1": "eeg",
    "EEG Fpz-Cz": "eeg",
    "EEG Pz-Oz": "eeg",
    "EOG1": "eog",
    "EOG2": "eog",
    "EOG1-A2": "eog",
    "EOG2-A2": "eog",
    "EMG1": "emg",
}

SCALES_UV = {"eeg": 100.0, "eog": 200.0, "emg": 50.0}
BANDS_HZ = {"eeg": (0.3, 35.0), "eog": (0.3, 15.0), "emg": (10.0, 45.0)}


def resolve_role_channels(header) -> dict[str, str | None]:
    names = {signal.name for signal in header.signals}
    if {"EEG Fpz-Cz", "EEG Pz-Oz"}.issubset(names):
        return {
            "f": "EEG Fpz-Cz",
            "c": "EEG Fpz-Cz",
            "o": "EEG Pz-Oz",
            "l": None,
            "r": None,
            "m": None,
        }
    return {
        role: next((name for name in variants if name in names), None)
        for role, variants in ROLE_CHANNEL_VARIANTS.items()
    }


def discover_records(
    tournament_root: str | Path, sleepedf_root: str | Path | None = None
) -> list[dict[str, Any]]:
    manifest_path = Path(tournament_root) / "data_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    lookup = {
        member: group_id for group_id, members in GROUPS.items() for member in members
    }
    records: list[dict[str, Any]] = []
    for task, prefix in (("kcomplex", "kc"), ("rem", "rem")):
        for original_id, row in manifest["records"][task].items():
            record_id = f"{prefix}_{original_id}"
            source = Path(row["edf_path"])
            if not source.exists():
                raise FileNotFoundError(f"Historical signal EDF is unavailable: {source}")
            header = read_header(source)
            channels = {signal.name for signal in header.signals}
            role_channels = resolve_role_channels(header)
            missing = [role for role, channel in role_channels.items() if not channel]
            if missing:
                raise ValueError(f"{record_id} missing required DREAMS roles: {missing}")
            group_id = lookup[record_id]
            split = next(
                name for name, group_ids in SPLIT_GROUPS.items() if group_id in group_ids
            )
            records.append(
                {
                    "dataset_id": "DREAMS_KC_REM_PSG_EXCERPTS",
                    "recording_id": record_id,
                    "source_task": task,
                    "source_recording_id": original_id,
                    "subject_group_id": group_id,
                    "subject_identity": "SOURCE_NIGHT_GROUP; PERSON_IDENTITY_UNVERIFIED",
                    "file_path": str(source.resolve()),
                    "duration_seconds": header.duration_seconds,
                    "sampling_rate_hz": 200.0,
                    "signal_unit": "uV",
                    "channels": sorted(channels),
                    "role_channels": role_channels,
                    "label_types": ["FROZEN_V3_WEAK_TEACHER"],
                    "label_provenance": "signal-only frozen detector; no hypnogram",
                    "license": "DREAMS CC BY-NC-ND 3.0; redistribution not performed",
                    "qc_status": "HEADER_CONTRACT_PASS",
                    "access_status": "LOCAL_PUBLIC_RESEARCH_ASSET",
                    "split": split,
                    "eligible_handlers": [
                        "E06",
                        "E07",
                        "E09",
                        "E10",
                        "E11",
                        "E13",
                    ],
                }
            )
    if sleepedf_root is not None:
        root = Path(sleepedf_root)
        candidates: dict[str, Path] = {}
        for path in sorted(root.glob("SC*-PSG.edf")):
            subject_id = path.name[:5]
            candidates.setdefault(subject_id, path)
        fixed_test = ("SC402", "SC408", "SC421")
        fixed_validation = ("SC405", "SC412", "SC424")
        remaining = sorted(set(candidates) - set(fixed_test) - set(fixed_validation))
        fixed_train = tuple(remaining[:12])
        selection = {
            "train": fixed_train,
            "validation": fixed_validation,
            "test": fixed_test,
        }
        for split, subjects in selection.items():
            for subject_id in subjects:
                source = candidates.get(subject_id)
                if source is None:
                    raise FileNotFoundError(f"SleepEDF subject not locally available: {subject_id}")
                header = read_header(source)
                roles = resolve_role_channels(header)
                if not all(roles[role] for role in ("c", "o")):
                    raise ValueError(f"SleepEDF EEG role mapping failed: {source.name}")
                records.append(
                    {
                        "dataset_id": "SLEEP_EDF_EXPANDED_SC_EEG_SUBSET",
                        "recording_id": "sleepedf_" + source.stem,
                        "source_task": "eeg_background",
                        "source_recording_id": source.stem,
                        "subject_group_id": "sleepedf_" + subject_id,
                        "subject_identity": subject_id,
                        "file_path": str(source.resolve()),
                        "duration_seconds": header.duration_seconds,
                        "sampling_rate_hz": 100.0,
                        "signal_unit": "uV",
                        "channels": [signal.name for signal in header.signals],
                        "role_channels": roles,
                        "label_types": ["FROZEN_V3_WEAK_TEACHER"],
                        "label_provenance": "signal-only frozen detector; hypnogram path excluded",
                        "license": "PhysioNet Sleep-EDF Expanded 1.0.0",
                        "qc_status": "HEADER_CONTRACT_PASS",
                        "access_status": "LOCAL_PUBLIC_RESEARCH_ASSET",
                        "split": split,
                        "eligible_handlers": ["E05", "E08", "E12"],
                    }
                )
    return sorted(records, key=lambda row: row["recording_id"])


def split_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    splits = {
        split: sorted(
            row["recording_id"] for row in records if row["split"] == split
        )
        for split in SPLIT_GROUPS
    }
    group_sets = {
        split: sorted(
            {row["subject_group_id"] for row in records if row["split"] == split}
        )
        for split in SPLIT_GROUPS
    }
    if any(
        set(group_sets[left]) & set(group_sets[right])
        for index, left in enumerate(group_sets)
        for right in list(group_sets)[index + 1 :]
    ):
        raise AssertionError("Source-night group leakage detected")
    return {
        "schema_version": 1,
        "frozen_before_training": True,
        "split_policy": "SOURCE_NIGHT_GROUP_DISJOINT",
        "subject_disjointness": "PERSON_IDENTITY_UNVERIFIED",
        "dreams_group_mapping": {key: list(value) for key, value in GROUPS.items()},
        "split_groups": {key: list(value) for key, value in SPLIT_GROUPS.items()},
        "records": splits,
        "records_by_family": {
            family: {
                split: sorted(
                    row["recording_id"]
                    for row in records
                    if row["split"] == split
                    and any(head in row["eligible_handlers"] for head in FAMILY_SPECS[family].output_heads)
                )
                for split in SPLIT_GROUPS
            }
            for family in FAMILY_SPECS
        },
        "group_overlap": False,
        "test_labels_loaded_by_orchestrator": True,
        "test_labels_indexed_by_training_or_validation_dataset": False,
    }


def _filter_and_resample(
    values: np.ndarray, source_rate: float, target_rate: int, kind: str
) -> np.ndarray:
    sos = butter(4, BANDS_HZ[kind], btype="bandpass", fs=source_rate, output="sos")
    filtered = sosfiltfilt(sos, np.asarray(values, dtype=np.float64))
    if abs(source_rate - target_rate) > 1e-6:
        filtered = resample_poly(filtered, target_rate, int(round(source_rate)))
    return np.clip(filtered / SCALES_UV[kind], -10.0, 10.0).astype(np.float32)


def load_family_signals(
    records: list[dict[str, Any]], family: str
) -> dict[str, np.ndarray]:
    target_rate = FAMILY_SPECS[family].input_rate_hz
    roles = FAMILY_ROLES[family]
    output: dict[str, np.ndarray] = {}
    for row in records:
        source = Path(row["file_path"])
        channels = tuple(row["role_channels"][role] for role in roles)
        raw = read_channels(source, channels, header=read_header(source))
        values = [
            _filter_and_resample(raw[channel][0], raw[channel][1], target_rate, CHANNEL_KINDS[channel])
            for channel in channels
        ]
        length = min(map(len, values))
        output[row["recording_id"]] = np.stack(
            [value[:length] for value in values], axis=0
        )
    return output


def load_teacher_labels(
    cache_dir: str | Path, record_ids: list[str], heads: tuple[str, ...]
) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    for record_id in record_ids:
        with np.load(Path(cache_dir) / f"{record_id}.npz", allow_pickle=False) as payload:
            output[record_id] = {
                key: payload[key].copy()
                for head in heads
                for key in (head + "_target", head + "_valid")
            }
    return output


@dataclass(frozen=True)
class WindowRef:
    record_id: str
    input_start: int
    output_start: int


class TemporalWindowDataset(Dataset):
    def __init__(
        self,
        signals: dict[str, np.ndarray],
        labels: dict[str, dict[str, np.ndarray]],
        family: str,
        record_ids: list[str],
        window_seconds: int,
        stride_seconds: int,
        *,
        permute_targets: bool = False,
        seed: int = 0,
    ):
        self.signals = signals
        self.labels = labels
        self.spec = FAMILY_SPECS[family]
        self.input_length = window_seconds * self.spec.input_rate_hz
        self.output_length = window_seconds * self.spec.output_rate_hz
        self.refs: list[WindowRef] = []
        self.permute_targets = permute_targets
        for record_id in record_ids:
            signal_seconds = signals[record_id].shape[-1] // self.spec.input_rate_hz
            label_seconds = min(
                len(labels[record_id][head + "_target"]) // self.spec.output_rate_hz
                for head in self.spec.output_heads
            )
            duration = min(signal_seconds, label_seconds)
            for start in range(0, duration - window_seconds + 1, stride_seconds):
                output_start = start * self.spec.output_rate_hz
                validity = np.stack(
                    [
                        labels[record_id][head + "_valid"][
                            output_start : output_start + self.output_length
                        ]
                        for head in self.spec.output_heads
                    ]
                )
                if validity.any():
                    self.refs.append(
                        WindowRef(
                            record_id,
                            start * self.spec.input_rate_hz,
                            output_start,
                        )
                    )
        if not self.refs:
            raise ValueError(f"No valid windows for {family}")
        self._permutation = np.random.default_rng(seed).permutation(len(self.refs))

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int):
        ref = self.refs[index]
        signal = self.signals[ref.record_id][
            :, ref.input_start : ref.input_start + self.input_length
        ]
        label_ref = self.refs[int(self._permutation[index])] if self.permute_targets else ref
        target = np.stack(
            [
                self.labels[label_ref.record_id][head + "_target"][
                    label_ref.output_start : label_ref.output_start + self.output_length
                ]
                for head in self.spec.output_heads
            ]
        )
        valid = np.stack(
            [
                self.labels[label_ref.record_id][head + "_valid"][
                    label_ref.output_start : label_ref.output_start + self.output_length
                ]
                for head in self.spec.output_heads
            ]
        )
        return (
            torch.from_numpy(np.ascontiguousarray(signal)).float(),
            torch.from_numpy(np.ascontiguousarray(target)).float(),
            torch.from_numpy(np.ascontiguousarray(valid)).float(),
            ref.record_id,
            ref.output_start,
        )


def head_prevalence(
    labels: dict[str, dict[str, np.ndarray]],
    record_ids: list[str],
    heads: tuple[str, ...],
) -> dict[str, dict[str, int | float]]:
    output = {}
    for head in heads:
        positives = 0
        valid = 0
        for record_id in record_ids:
            target = labels[record_id][head + "_target"].astype(bool)
            mask = labels[record_id][head + "_valid"].astype(bool)
            positives += int(np.sum(target & mask))
            valid += int(mask.sum())
        output[head] = {
            "positive_samples": positives,
            "valid_samples": valid,
            "prevalence": positives / valid if valid else 0.0,
        }
    return output
