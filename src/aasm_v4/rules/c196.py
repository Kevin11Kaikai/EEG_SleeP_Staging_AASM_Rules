"""Public definition of the selected C196 research policy.

The executable discovery engine is retained in
``experiments/v4_rule_discovery/shadow_engine.py``.  This module exposes the
frozen selection as auditable metadata and deliberately does not silently
replace the public AASM V2 or Temporal R1 engines.
"""

from __future__ import annotations

from types import MappingProxyType


C196_POLICY = MappingProxyType(
    {
        "parent_config": "temporal_r1",
        "rem_backward_policy": "aasm_v21_backward_bridge",
        "post_arousal_reentry_policy": "structure_reentry_first",
        "unchanged_policies": (
            "wake_ocular_policy=duration15",
            "n1_alpha_policy=adjacent_attenuation",
            "n1_nonalpha_policy=any_marker",
            "n2_continuation_policy=prior_epoch_structure",
            "fallback_policy=generic_inherit",
            "rem_second_half_policy=all_structure_blocks",
        ),
        "scientific_scope": "WITHIN_PROJECT_POST_HOC_CONFIRMATION",
    }
)


def describe_c196() -> dict[str, object]:
    """Return a JSON-serializable copy of the frozen policy description."""

    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in C196_POLICY.items()
    }
