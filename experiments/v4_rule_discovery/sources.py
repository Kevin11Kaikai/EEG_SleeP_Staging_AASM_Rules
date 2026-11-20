from __future__ import annotations


RULE_SOURCES = {
    "FROZEN_TEMPORAL_R1": {
        "claim_type": "AASM_OPERATIONALIZATION",
        "locator": "frozen_dependencies/temporal_r1/src/aasm_temporal_r1/engine.py",
        "use": "Exact executable baseline and fixed detector/evidence inputs.",
    },
    "AASM_V2_1_REM_BACKWARD": {
        "claim_type": "AASM_PUBLIC_UPDATE",
        "url": "https://aasm.org/wp-content/uploads/2017/11/Summary-of-Updates-in-v2.1-FINAL.pdf",
        "use": "Definite-R context may determine contiguous preceding LAMF/low-tone segments without intervening termination evidence.",
    },
    "AASM_STAGE_W_EYE_CRITERIA": {
        "claim_type": "AASM_COMPATIBLE_DETECTOR_CALIBRATION",
        "url": "https://isr.aasm.org/help/TheAASMManualfortheScoringofSlee.html",
        "use": "Discrete Wake eye markers are tested as event count/presence instead of impossible 15-second event duration.",
    },
    "AASM_V2_1_N1_MAJORITY_LAMF": {
        "claim_type": "AASM_PUBLIC_UPDATE",
        "url": "https://aasm.org/wp-content/uploads/2017/11/Summary-of-Updates-in-v2.1-FINAL.pdf",
        "use": "N1 requires majority LAMF; SEM/vertex are supportive transition markers, not generic slowing substitutes.",
    },
    "AASM_V2_1_N2_CONTINUATION": {
        "claim_type": "AASM_PUBLIC_UPDATE",
        "url": "https://aasm.org/wp-content/uploads/2017/11/Summary-of-Updates-in-v2.1-FINAL.pdf",
        "use": "N2 context continues until a defined transition without an immediate prior-epoch structure requirement.",
    },
    "AASM_V2_1_N2_POST_AROUSAL": {
        "claim_type": "AASM_PUBLIC_UPDATE_PLUS_ENGINEERING_OPERATIONALIZATION",
        "url": "https://aasm.org/wp-content/uploads/2017/11/Summary-of-Updates-in-v2.1-FINAL.pdf",
        "use": (
            "AASM supports segment-level N2 re-entry after a qualifying post-arousal "
            "spindle/K-complex; whole-epoch priority over temporal majority is a tested "
            "project engineering operationalization, not AASM verbatim."
        ),
    },
    "AASM_STAGE_CONTINUATION_TERMINATION": {
        "claim_type": "AASM_COMPATIBLE_ENGINEERING",
        "url": "https://aasm.org/wp-content/uploads/2017/11/Summary-of-Updates-in-v2.1-FINAL.pdf",
        "use": "Stage-specific continuation replaces a generic previous-output fallback.",
    },
    "AASM_REM_TRANSITION_FAQ": {
        "claim_type": "AASM_PUBLIC_FAQ_OPERATIONALIZATION",
        "url": "https://aasm.org/resources/pdf/faqsscoringmanual.pdf",
        "use": "K-complex timing within R/N2 transitions is handled by half-epoch chronology rather than any-event blocking.",
    },
}


IMMUTABLE_RULE_CONTRACT = {
    "epoch_seconds": 30,
    "strict_majority_seconds": 15,
    "slow_wave_seconds_inclusive": 6,
    "interval_convention": "half_open",
    "truth_logic": "strong_kleene",
    "n2_onset_window": "[-15,+15)",
    "k_arousal_exclusion": True,
    "movement_resolution": "frozen_two_pass",
    "event_banks": "M_6408_frozen",
    "evidence_aggregation": "frozen",
    "detector_thresholds": "frozen",
    "discovery_records": "AASM_V4_DISCOVERY_RECORDS",
    "confirmation_records": "AASM_V4_RECORDS minus discovery records",
}
