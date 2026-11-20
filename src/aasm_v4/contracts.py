"""Public contracts for the V4 research release.

These contracts describe detector selections and aggregate stage metrics.  They
intentionally do not encode patient-level data, signal paths, or model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


STAGES: tuple[str, ...] = ("W", "N1", "N2", "N3", "R")
UNDETERMINED = "UNDETERMINED"


@dataclass(frozen=True)
class DetectorProfile:
    """A frozen set of event-source selections.

    ``enabled_handlers`` contains only handlers that differ from the frozen
    B000 event bank.  It is therefore an intervention description, not a claim
    that every listed detector has independent clinical validation.
    """

    profile_id: str
    parent_profile: str | None
    enabled_handlers: tuple[str, ...]
    selection_scope: str
    scientific_scope: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DetectorProfile":
        handlers = tuple(str(item) for item in value.get("enabled_handlers", ()))
        if len(handlers) != len(set(handlers)):
            raise ValueError("enabled_handlers must be unique")
        return cls(
            profile_id=str(value["profile_id"]),
            parent_profile=(
                None if value.get("parent_profile") is None else str(value["parent_profile"])
            ),
            enabled_handlers=handlers,
            selection_scope=str(value["selection_scope"]),
            scientific_scope=str(value["scientific_scope"]),
        )


@dataclass(frozen=True)
class RuleProfile:
    """A validated rule-policy profile layered over a frozen parent engine."""

    config_id: str
    parent_config: str
    policies: Mapping[str, str]
    changed_modules: tuple[str, ...]
    scientific_scope: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuleProfile":
        policy_keys = (
            "rem_backward_policy",
            "wake_ocular_policy",
            "n1_alpha_policy",
            "n1_nonalpha_policy",
            "n2_continuation_policy",
            "post_arousal_reentry_policy",
            "fallback_policy",
            "rem_second_half_policy",
        )
        missing = [key for key in policy_keys if key not in value]
        if missing:
            raise ValueError(f"rule profile missing policies: {missing}")
        return cls(
            config_id=str(value["config_id"]),
            parent_config=str(value["parent_config"]),
            policies={key: str(value[key]) for key in policy_keys},
            changed_modules=tuple(str(item) for item in value.get("changed_modules", ())),
            scientific_scope=str(value["scientific_scope"]),
        )


@dataclass(frozen=True)
class AggregateStageMetrics:
    """Fixed-mask aggregate metrics with explicit coverage."""

    n: int
    correct: int
    accuracy: float
    macro_f1: float
    coverage: float
    per_stage_f1: Mapping[str, float]

    def validate(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be positive")
        if not 0 <= self.correct <= self.n:
            raise ValueError("correct must be between zero and n")
        for name, number in (
            ("accuracy", self.accuracy),
            ("macro_f1", self.macro_f1),
            ("coverage", self.coverage),
        ):
            if not 0.0 <= number <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if set(self.per_stage_f1) != set(STAGES):
            raise ValueError(f"per_stage_f1 must contain exactly {STAGES}")


def validate_handler_registry(rows: Sequence[Mapping[str, Any]]) -> None:
    """Validate the public 15-handler registry without loading private assets."""

    expected = {*(f"E{index:02d}" for index in range(1, 14)), "SP", "AR"}
    identifiers = [str(row.get("handler_id")) for row in rows]
    if set(identifiers) != expected or len(identifiers) != len(expected):
        raise ValueError("registry must contain E01-E13, SP, and AR exactly once")
    allowed = {
        "expert_supported",
        "weak_supervision_only",
        "prototype_only",
        "movement_proxy",
    }
    invalid = [row for row in rows if row.get("evidence_level") not in allowed]
    if invalid:
        raise ValueError("registry contains an unsupported evidence_level")
