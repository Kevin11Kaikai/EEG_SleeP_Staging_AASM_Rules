from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


WORKTREE = Path(__file__).resolve().parents[2]


def _configured_path(variable: str, default: Path) -> Path:
    """Resolve a private dependency without embedding a workstation path."""

    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else default.resolve()


RESULTS = _configured_path("AASM_V4_RESULTS_ROOT", WORKTREE / "results" / "local")
FROZEN_CORE = _configured_path(
    "AASM_V4_FROZEN_ROOT", WORKTREE / "frozen_dependencies" / "temporal_r1"
)
PUBLIC_ROOT = _configured_path(
    "AASM_V4_PUBLIC_RULE_ROOT", WORKTREE / "frozen_dependencies" / "aasm_rules_v2_public"
)
BASE_BANK_ROOT = RESULTS / "aasm_full_detector_integration_20261115T032300-0500" / "event_bank" / "B000"
R1_ROOT = FROZEN_CORE / "results" / "development_r1"
ALIGNED_REFERENCE = R1_ROOT / "evaluation" / "aligned_development.csv.gz"
ALLOWLIST_MANIFEST = RESULTS / "sumov2_spindle_v3_20261116T125515-0500" / "hmc_inference_inputs.json"

PHASE1C = RESULTS / "neural_handlers_phase1c_20261117T162815-0500"
PHASE1D = RESULTS / "sem_final_upgrade_v4_20261117T181322-0500"
SSSM_RUN = RESULTS / "multi_event_dl_audit_v3_20261116T181230-0500"
K08_RUN = RESULTS / "k08_r04_robustness_v3_20261116T225331-0500"
R04_RUN = RESULTS / "r04_rem_k08_final_v3_20261117T093354-0500"
SP_RUN = RESULTS / "sumov2_spindle_v3_20261116T125515-0500"
AR_RUN = RESULTS / "aasm_full_detector_integration_20261115T032300-0500"

_record_text = os.environ.get("AASM_V4_RECORDS", "")
RECORDS = tuple(item.strip() for item in _record_text.split(",") if item.strip())
if not RECORDS:
    # Public placeholder IDs preserve the ten-record protocol without exposing
    # the private development-set pseudonyms. Set AASM_V4_RECORDS for replay.
    RECORDS = tuple(f"development_record_{index:02d}" for index in range(1, 11))
HANDLERS = (
    "E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08",
    "E09", "E10", "E11", "E12", "E13", "SP", "AR",
)
EVENT_TYPE = {
    "E01": "slow_wave", "E02": "kcomplex", "E03": "vertex",
    "E04": "rem", "E05": "alpha", "E06": "blink", "E07": "sem",
    "E08": "lamf", "E09": "low_emg", "E10": "normal_emg",
    "E11": "high_emg", "E12": "slowing", "E13": "mbm",
    "SP": "spindle", "AR": "arousal",
}
DISPLAY_NAME = {
    "E01": "Slow-wave", "E02": "K-complex", "E03": "Vertex",
    "E04": "REM", "E05": "Alpha", "E06": "Blink", "E07": "SEM",
    "E08": "LAMF", "E09": "Low EMG", "E10": "Normal EMG",
    "E11": "High EMG", "E12": "Slowing", "E13": "MBM proxy",
    "SP": "Spindle", "AR": "Arousal",
}

GROUPS = {
    "G_N2": ("E02", "SP", "AR"),
    "G_OCULAR": ("E04", "E06", "E07"),
    "G_EEG": ("E01", "E03", "E05", "E08", "E12"),
    "G_EMG_MOV": ("E09", "E10", "E11", "E13"),
}


def build_arms() -> dict[str, tuple[str, ...]]:
    arms: dict[str, tuple[str, ...]] = {"B000": ()}
    arms.update({f"S_{handler}": (handler,) for handler in HANDLERS})
    arms.update(GROUPS)
    arms["F_ALL15"] = HANDLERS
    arms.update({f"L_{handler}": tuple(x for x in HANDLERS if x != handler) for handler in HANDLERS})
    if len(arms) != 36 or len(set(arms)) != 36:
        raise AssertionError("Phase 2 arm registry must contain exactly 36 unique arms")
    return arms


ARMS = build_arms()


@dataclass(frozen=True)
class FamilyConfig:
    family: str
    handlers: tuple[str, ...]
    checkpoint: Path
    expected_sha256: str
    channels: tuple[str, ...]
    kinds: tuple[str, ...]
    window_seconds: int
    batch_size: int


FAMILIES = {
    "eeg_background": FamilyConfig(
        "eeg_background", ("E05", "E08", "E12"),
        PHASE1C / "models" / "eeg_background" / "best_checkpoint.pt",
        "be1b0b34af81f8897b2d1f8ba7e78622db8b49d427c6e88e11b8d8d6e1da3045",
        ("EEG C4-M1", "EEG O2-M1"), ("eeg", "eeg"), 120, 24,
    ),
    "eog_events": FamilyConfig(
        "eog_events", ("E06",),
        PHASE1C / "models" / "eog_events" / "best_checkpoint.pt",
        "027c600c63e3098531c2cf7a4aff05a244480c0f0c6207c6ac47a2c5be2ca57a",
        ("EOG E1-M2", "EOG E2-M2"), ("eog", "eog"), 20, 64,
    ),
    "chin_emg": FamilyConfig(
        "chin_emg", ("E09", "E10", "E11"),
        PHASE1C / "models" / "chin_emg" / "best_checkpoint.pt",
        "c5c47a68527eaca3cf40f4f99303921f1c90a1552ceed5dc1313807078168974",
        ("EMG chin",), ("emg",), 120, 32,
    ),
    "major_movement": FamilyConfig(
        "major_movement", ("E13",),
        PHASE1C / "models" / "major_movement" / "best_checkpoint.pt",
        "078e49dd77d6e249b72f79a18291d59b46b6ab4e062844a77edd9e05f7f0d62f",
        ("EEG F4-M1", "EEG C4-M1", "EMG chin"), ("eeg", "eeg", "emg"), 120, 24,
    ),
}

PHASE1C_CONFIG = {
    "E05": (0.8, 1, {"minimum_seconds": 2.0, "merge_gap_seconds": 0.5}),
    "E06": (0.8, 10, {"minimum_seconds": 0.1, "merge_gap_seconds": 0.1}),
    "E08": (0.5, 1, {"minimum_seconds": 1.0, "merge_gap_seconds": 0.5}),
    "E09": (0.2, 2, {"minimum_seconds": 1.0, "merge_gap_seconds": 0.5}),
    "E10": (0.3, 2, {"minimum_seconds": 1.0, "merge_gap_seconds": 0.5}),
    "E11": (0.8, 2, {"minimum_seconds": 1.0, "merge_gap_seconds": 0.5}),
    "E12": (0.4, 1, {"minimum_seconds": 1.0, "merge_gap_seconds": 0.0}),
    "E13": (0.7, 1, {"minimum_seconds": 1.0, "merge_gap_seconds": 0.0}),
}

SSSM_CHECKPOINT = SSSM_RUN / "inputs" / "sssm_wheel" / "sssm" / "saved_models" / "model.pt"
SSSM_SHA256 = "2f85ec2c4d940a4f417b35137a49401e3e6e6df943aef9a7215a342a5b2ae7c6"
E07_CHECKPOINT = PHASE1D / "models" / "tcn" / "best_checkpoint.pt"
E07_SHA256 = "44bd9c2ca9966cd1f39e212d4dd6adc650ec3a4df189350c28cbd3fc8e8823d9"

PRECOMPUTED = {
    "E02": K08_RUN / "hmc_k08" / "detectors" / "{record}_k08.json",
    "E04": R04_RUN / "r04" / "detectors" / "{record}_r04.json",
    "SP": SP_RUN / "hmc_sumov2_events" / "{record}.json",
    "AR": AR_RUN / "detectors" / "{record}" / "unet_arousal.json",
}

EXPECTED_MODEL_SHA = {
    "E01": SSSM_SHA256, "E02": "356fbf13065ad18ba68171c12f49bb2193aecd07f9c0f3af777fc4224fd30630",
    "E03": SSSM_SHA256, "E04": "5abf1a5eddbfe9998301e5e7a2e95c4dd3ed977e0199d433a23b5cf8f69a4390",
    "E05": FAMILIES["eeg_background"].expected_sha256,
    "E06": FAMILIES["eog_events"].expected_sha256,
    "E07": E07_SHA256, "E08": FAMILIES["eeg_background"].expected_sha256,
    "E09": FAMILIES["chin_emg"].expected_sha256,
    "E10": FAMILIES["chin_emg"].expected_sha256,
    "E11": FAMILIES["chin_emg"].expected_sha256,
    "E12": FAMILIES["eeg_background"].expected_sha256,
    "E13": FAMILIES["major_movement"].expected_sha256,
    "SP": "2f3054ffd12c56e440c00548c38448d1f6e3333b360de50919b7015c02964e79",
    "AR": "6fbea2857567f95023e6dd34461fba92cab238c9c334dbe9d66494b258aa715f",
}
