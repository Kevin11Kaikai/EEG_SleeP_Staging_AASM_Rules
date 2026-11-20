from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any


WORKTREE = Path(__file__).resolve().parents[2]


def _configured_path(variable: str, default: Path) -> Path:
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else default.resolve()


FROZEN_CORE = _configured_path(
    "AASM_V4_FROZEN_ROOT", WORKTREE / "frozen_dependencies" / "temporal_r1"
)
FROZEN_SRC = FROZEN_CORE / "src"
RESULTS_ROOT = _configured_path("AASM_V4_RESULTS_ROOT", WORKTREE / "results" / "local")
SOURCE_RUN = _configured_path(
    "AASM_V4_CHAMPION_RUN",
    RESULTS_ROOT / "phase2_full_exhaustive_search_20261119T085226-0500",
)
CHAMPION_ROOT = SOURCE_RUN / "finalists" / "full_trace" / "M_6408"
ALIGNED_REFERENCE = FROZEN_CORE / "results" / "development_r1" / "evaluation" / "aligned_development.csv.gz"

_record_text = os.environ.get("AASM_V4_RECORDS", "")
RECORDS = tuple(item.strip() for item in _record_text.split(",") if item.strip())
if not RECORDS:
    RECORDS = tuple(f"development_record_{index:02d}" for index in range(1, 11))
_discovery_text = os.environ.get("AASM_V4_DISCOVERY_RECORDS", "")
DISCOVERY_RECORDS = tuple(
    item.strip() for item in _discovery_text.split(",") if item.strip()
) or RECORDS[: len(RECORDS) // 2]
CONFIRMATION_RECORDS = tuple(record for record in RECORDS if record not in DISCOVERY_RECORDS)
STAGES = ("W", "N1", "N2", "N3", "R", "UNDETERMINED")
STAGE_TO_CODE = {stage: index for index, stage in enumerate(STAGES)}
CODE_TO_STAGE = {value: key for key, value in STAGE_TO_CODE.items()}
PRIMARY_EPOCHS = 8902
FULL_VALID_EPOCHS = 9211
SEED = 2026


@dataclass(frozen=True)
class RuleConfig:
    config_id: str
    rem_backward_policy: str
    wake_ocular_policy: str
    n1_alpha_policy: str
    n1_nonalpha_policy: str
    n2_continuation_policy: str
    post_arousal_reentry_policy: str
    fallback_policy: str
    rem_second_half_policy: str
    source_ids: tuple[str, ...]
    claim_types: tuple[str, ...]
    changed_modules: tuple[str, ...]
    complexity: int
    parent_config: str = "temporal_r1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("source_ids", "claim_types", "changed_modules"):
            value[key] = list(value[key])
        return value


def candidate_universe() -> tuple[RuleConfig, ...]:
    """Return a frozen 2 x 3 x 2^6 source-grounded rule universe."""

    dimensions = (
        ("off", "aasm_v21_backward_bridge"),
        (
            "duration15",
            "irregular_presence_with_tone",
            "blink_count6_or_irregular_with_tone",
        ),
        ("adjacent_attenuation", "persistent_alpha_majority_anchor"),
        ("any_marker", "lamf_sem_vertex"),
        ("prior_epoch_structure", "physiological_anchor"),
        ("temporal_majority_first", "structure_reentry_first"),
        ("generic_inherit", "aasm_stage_specific"),
        ("all_structure_blocks", "second_half_structure_delayed"),
    )
    output: list[RuleConfig] = []
    for index, values in enumerate(product(*dimensions)):
        (
            rem_backward,
            wake_ocular,
            n1_alpha,
            n1_nonalpha,
            n2_continuation,
            post_arousal,
            fallback,
            rem_second_half,
        ) = values
        changed: list[str] = []
        sources = ["FROZEN_TEMPORAL_R1"]
        claims = ["AASM_OPERATIONALIZATION"]
        mappings = (
            (rem_backward != "off", "R.BACKWARD_BRIDGE", "AASM_V2_1_REM_BACKWARD", "AASM_PUBLIC_UPDATE"),
            (wake_ocular != "duration15", "W.2_W.3_OCULAR_SEMANTICS", "AASM_STAGE_W_EYE_CRITERIA", "AASM_COMPATIBLE_DETECTOR_CALIBRATION"),
            (n1_alpha != "adjacent_attenuation", "N1.1_ALPHA_ANCHOR", "AASM_V2_1_N1_MAJORITY_LAMF", "AASM_COMPATIBLE_ENGINEERING"),
            (n1_nonalpha != "any_marker", "N1.2_LAMF_SEM_VERTEX", "AASM_V2_1_N1_MAJORITY_LAMF", "AASM_PUBLIC_UPDATE"),
            (n2_continuation != "prior_epoch_structure", "N2.2_ANCHORED_CONTINUATION", "AASM_V2_1_N2_CONTINUATION", "AASM_PUBLIC_UPDATE"),
            (post_arousal != "temporal_majority_first", "N2_POST_AROUSAL_REENTRY", "AASM_V2_1_N2_POST_AROUSAL", "AASM_PUBLIC_UPDATE"),
            (fallback != "generic_inherit", "STAGE_SPECIFIC_CONTINUATION", "AASM_STAGE_CONTINUATION_TERMINATION", "AASM_COMPATIBLE_ENGINEERING"),
            (rem_second_half != "all_structure_blocks", "R.2_SECOND_HALF_STRUCTURE", "AASM_REM_TRANSITION_FAQ", "AASM_PUBLIC_FAQ_OPERATIONALIZATION"),
        )
        for active, module, source, claim in mappings:
            if active:
                changed.append(module)
                sources.append(source)
                claims.append(claim)
        output.append(
            RuleConfig(
                config_id=f"C{index:03d}",
                rem_backward_policy=rem_backward,
                wake_ocular_policy=wake_ocular,
                n1_alpha_policy=n1_alpha,
                n1_nonalpha_policy=n1_nonalpha,
                n2_continuation_policy=n2_continuation,
                post_arousal_reentry_policy=post_arousal,
                fallback_policy=fallback,
                rem_second_half_policy=rem_second_half,
                source_ids=tuple(dict.fromkeys(sources)),
                claim_types=tuple(dict.fromkeys(claims)),
                changed_modules=tuple(changed),
                complexity=len(changed),
            )
        )
    baseline = output[0]
    expected = {
        "config_id": "C000",
        "rem_backward_policy": "off",
        "wake_ocular_policy": "duration15",
        "n1_alpha_policy": "adjacent_attenuation",
        "n1_nonalpha_policy": "any_marker",
        "n2_continuation_policy": "prior_epoch_structure",
        "post_arousal_reentry_policy": "temporal_majority_first",
        "fallback_policy": "generic_inherit",
        "rem_second_half_policy": "all_structure_blocks",
    }
    if any(getattr(baseline, key) != value for key, value in expected.items()):
        raise AssertionError("C000 must remain exact Temporal R1")
    if len(output) != 384 or len({item.config_id for item in output}) != 384:
        raise AssertionError("candidate universe must contain exactly 384 configs")
    return tuple(output)


def validate_frozen_paths(include_reference: bool = False) -> None:
    required = [FROZEN_CORE, CHAMPION_ROOT]
    if include_reference:
        required.append(ALIGNED_REFERENCE)
    for record in RECORDS:
        required.extend(
            [
                CHAMPION_ROOT / "event_banks" / f"{record}.json.gz",
                CHAMPION_ROOT / "evidence" / f"{record}.json.gz",
                CHAMPION_ROOT / "predictions" / f"{record}.json.gz",
            ]
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen input(s): {missing}")
    if len(RECORDS) != 10 or len(DISCOVERY_RECORDS) != 5 or len(CONFIRMATION_RECORDS) != 5:
        raise RuntimeError("expected a frozen 5-record discovery / 5-record confirmation split")
    if set(DISCOVERY_RECORDS) & set(CONFIRMATION_RECORDS):
        raise RuntimeError("discovery and confirmation record groups overlap")
