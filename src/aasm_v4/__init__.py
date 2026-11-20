"""Public V4 event-to-stage research interfaces.

The V4 layer keeps :mod:`aasm_v2` and :mod:`aasm_temporal_r1` frozen and adds
portable handler profiles, combination-search metadata, and the selected C196
decision-policy overlay.
"""

from .profiles import load_detector_profile, load_rule_profile

__all__ = ["load_detector_profile", "load_rule_profile"]
