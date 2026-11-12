"""Versioned operational rules; no implicit merger of incompatible definitions."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class RuleDefinition:
    rule_id: str
    source_document: str
    source_version: str
    required_events: tuple
    onset_or_continuation_or_termination: str
    thresholds: dict
    boundary_conventions: str


def registry() -> list[dict]:
    definitions = [
        ("W.1", ("alpha",), "onset", {"coverage_gt_s": 15}),
        ("W.2", ("blink",), "onset", {"coverage_gt_s": 15}),
        ("W.3", ("rem", "normal_emg"), "onset", {"concurrent_coverage_gt_s": 15}),
        ("N1.1", ("alpha", "lamf"), "onset", {"lamf_gt_s": 15, "alpha_attenuated": True}),
        ("N1.2", ("slowing", "vertex", "sem"), "onset", {"slowing_hz": 1}),
        ("N2.1", ("spindle", "kcomplex", "arousal"), "onset", {"onset_window_s": [-15, 15]}),
        ("N2.2", ("lamf", "arousal", "spindle", "kcomplex"), "continuation", {"lamf_gt_s": 15}),
        ("N2.3", ("slow_wave", "arousal"), "continuation", {"previous_stage": "N3"}),
        ("N2.4", ("arousal", "lamf", "mbm", "sem"), "termination", {"reentry": "new qualifying N2 event"}),
        ("N3.1", ("slow_wave",), "onset", {"coverage_ge_s": 6, "ptp_gt_uv": 75, "source_band_hz": [.5, 2]}),
        ("R.1", ("lamf", "rem", "low_emg", "spindle", "kcomplex"), "onset", {"low_emg_gt_s": 15, "overlap_gt_s": 0}),
        ("R.2", ("lamf", "low_emg", "arousal", "spindle", "kcomplex", "sem"), "continuation", {"definite_R_anchor": True}),
        ("R.3", ("high_emg", "lamf", "arousal", "sem", "spindle", "kcomplex"), "termination", {"target": "branch-specific"}),
        ("MBM.1", ("mbm", "alpha"), "movement", {"movement_gt_s": 15, "adjacent_W": True}),
        ("MBM.2", ("mbm",), "movement", {"lookahead": "next non-MBM algorithm prediction"}),
    ]
    result = [asdict(RuleDefinition(rid, "SleepVLM Table S2", "arXiv:2603.26738v4 / AASM3 operationalization",
                events, kind, thresholds,
                "Half-open [start,end); exactly 15s is not majority; N3 includes exactly6s; no GT; "
                "consecutive MBM solved as a block; clinical absence UNKNOWN; primary executes explicitly logged candidate-absence proxies"))
              for rid, events, kind, thresholds in definitions]
    for r in result:
        if r["rule_id"] == "N3.1":
            r["implementation_deviation"] = ("YASA shared original 0.3–1.5Hz SW candidate definition on F4+C4; "
                "not exact 0.5–2Hz frontal morphology. Held fixed to isolate requested detector packages.")
    result.append(asdict(RuleDefinition("ORIGINAL.PASS1_PASS2", "Frozen local paper-inspired specification",
        "full_v1 source_snapshot; arXiv:2605.22859v1 / AASM2.6 inspiration",
        ("slow_wave", "alpha", "lamf", "rem", "low_emg", "spindle", "kcomplex", "arousal"),
        "two_pass", {"N3_gt_s": 6, "low_emg_ge_s": 15, "R_lamf_ge_s": 1},
        "Full-night first pass; second sequential pass with immediate inheritance. Candidate negatives are legacy proxies.")))
    return result
