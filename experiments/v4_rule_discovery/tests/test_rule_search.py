from __future__ import annotations

import os
from dataclasses import replace

import pytest

from experiments.v4_rule_discovery.config import RECORDS, candidate_universe
from experiments.v4_rule_discovery.replay import replay_record
from experiments.v4_rule_discovery.shadow_engine import engine

from aasm_v2.types import EpochEvidence, Evidence


KEYS = (
    "mbm", "slow_wave", "alpha", "blink", "irregular_rem_normal",
    "spindle", "kcomplex", "lamf", "rem", "low_emg",
    "rem_low_overlap", "n2_spindle", "n2_kcomplex", "arousal_then_sem",
    "first_half_n2", "sem", "slowing", "vertex", "arousal",
    "high_emg", "normal_emg",
)


def row(
    index: int,
    *,
    attributes: dict | None = None,
    event_ids: dict[str, list[str]] | None = None,
    **seconds: float,
) -> EpochEvidence:
    identifiers = event_ids or {}
    events = {
        key: Evidence(
            seconds=float(seconds.get(key, 0.0)),
            event_ids=list(identifiers.get(key, [])),
            detector_status="AVAILABLE",
            negative_validated=False,
        )
        for key in KEYS
    }
    row_attributes = {"alpha_generator": False, "n2_event_in_epoch": False}
    row_attributes.update(attributes or {})
    return EpochEvidence(
        index=index,
        signals_good=True,
        events=events,
        attributes=row_attributes,
    )


def packet(
    *,
    arousal: list[dict] | None = None,
    rem: list[dict] | None = None,
    lamf: list[dict] | None = None,
    structure: list[dict] | None = None,
    carry_structure: list[dict] | None = None,
) -> dict:
    return {
        "arousal": list(arousal or []),
        "lamf": list(lamf or []),
        "structure": list(structure or []),
        "carry_structure": list(carry_structure or []),
        "rem": list(rem or []),
    }


BASELINE = candidate_universe()[0]


def config(**changes):
    return replace(BASELINE, **changes)


def test_candidate_universe_and_c000_contract():
    values = candidate_universe()
    assert len(values) == 384
    assert len({value.config_id for value in values}) == 384
    assert len(
        {
            (
                value.rem_backward_policy,
                value.wake_ocular_policy,
                value.n1_alpha_policy,
                value.n1_nonalpha_policy,
                value.n2_continuation_policy,
                value.post_arousal_reentry_policy,
                value.fallback_policy,
                value.rem_second_half_policy,
            )
            for value in values
        }
    ) == 384
    assert BASELINE.config_id == "C000"
    assert BASELINE.changed_modules == ()
    assert BASELINE.complexity == 0


@pytest.mark.parametrize("record_id", RECORDS)
def test_c000_exact_frozen_regression(record_id: str):
    if "AASM_V4_CHAMPION_RUN" not in os.environ:
        pytest.skip("private frozen champion replay is not configured")
    result = replay_record(BASELINE, record_id)
    assert result["baseline_exact"] is True
    assert len(result["decisions"]) == result["summary"]["epochs"]


def test_rem_backward_bridge_relabels_only_eligible_prior_epoch():
    rows = [
        row(0, lamf=20, low_emg=20),
        row(1, lamf=20, rem=1, low_emg=20, rem_low_overlap=1),
    ]
    packets = [
        packet(),
        packet(rem=[{"event_id": "r1", "onset": 35.0, "offset": 36.0}]),
    ]
    baseline, _ = engine(rows, packets, BASELINE)
    bridged, diagnostics = engine(
        rows, packets, config(rem_backward_policy="aasm_v21_backward_bridge")
    )
    assert baseline[0]["stage"] == "UNDETERMINED"
    assert baseline[1]["rule_id"] == "R.1"
    assert bridged[0]["stage"] == "R"
    assert bridged[0]["rule_id"] == "R.BACKWARD"
    assert bridged[0]["previous_context"]["future_definite_r_epoch"] == 1
    assert diagnostics[0]["rem_backward_relabelled"] is True


def test_rem_backward_bridge_stops_at_competing_wake_evidence():
    rows = [
        row(0, alpha=20, lamf=20, low_emg=20),
        row(1, lamf=20, rem=1, low_emg=20, rem_low_overlap=1),
    ]
    decisions, diagnostics = engine(
        rows,
        [packet(), packet()],
        config(rem_backward_policy="aasm_v21_backward_bridge"),
    )
    assert decisions[0]["stage"] == "W"
    assert decisions[0]["rule_id"] == "W.1"
    assert diagnostics[0]["rem_backward_relabelled"] is False


def test_wake_blink_count_and_tone_replaces_impossible_duration_test():
    blink_ids = [f"b{index}" for index in range(6)]
    ocular = row(0, blink=0.6, normal_emg=20, event_ids={"blink": blink_ids})
    baseline, _ = engine([ocular], [packet()], BASELINE)
    revised, _ = engine(
        [ocular],
        [packet()],
        config(wake_ocular_policy="blink_count6_or_irregular_with_tone"),
    )
    assert baseline[0]["stage"] == "UNDETERMINED"
    assert revised[0]["stage"] == "W"
    assert revised[0]["rule_id"] == "W.2"


def test_wake_irregular_eye_presence_requires_normal_or_high_tone():
    with_tone = row(0, irregular_rem_normal=1, normal_emg=20)
    without_tone = row(0, irregular_rem_normal=1)
    revised = config(wake_ocular_policy="irregular_presence_with_tone")
    positive, _ = engine([with_tone], [packet()], revised)
    negative, _ = engine([without_tone], [packet()], revised)
    assert positive[0]["rule_id"] == "W.3"
    assert negative[0]["stage"] == "UNDETERMINED"


def test_n1_nonalpha_path_requires_majority_lamf_and_sem_or_vertex():
    slowing_only = row(0, slowing=1)
    short_lamf_sem = row(0, lamf=10, sem=1)
    majority_lamf_sem = row(0, lamf=20, sem=1)
    revised = config(n1_nonalpha_policy="lamf_sem_vertex")

    baseline, _ = engine([slowing_only], [packet()], BASELINE)
    revised_slowing, _ = engine([slowing_only], [packet()], revised)
    revised_short, _ = engine([short_lamf_sem], [packet()], revised)
    revised_positive, _ = engine([majority_lamf_sem], [packet()], revised)

    assert baseline[0]["rule_id"] == "N1.2"
    assert revised_slowing[0]["stage"] == "UNDETERMINED"
    assert revised_short[0]["stage"] == "UNDETERMINED"
    assert revised_positive[0]["stage"] == "N1"
    assert revised_positive[0]["rule_id"] == "N1.2"


def test_n2_physiological_anchor_continues_without_prior_epoch_structure():
    rows = [row(0, n2_spindle=1), row(1, lamf=20)]
    packets = [packet(), packet()]
    baseline, _ = engine(rows, packets, BASELINE)
    revised, _ = engine(
        rows,
        packets,
        config(n2_continuation_policy="physiological_anchor"),
    )
    assert baseline[0]["rule_id"] == "N2.1"
    assert baseline[1]["rule_id"] == "FALLBACK.INHERIT_PREVIOUS"
    assert revised[1]["stage"] == "N2"
    assert revised[1]["rule_id"] == "N2.2"


def test_post_arousal_structure_reentry_precedes_n1_temporal_majority():
    arousal = {
        "event_id": "a1", "event_type": "arousal", "onset": 31.0, "offset": 35.0,
    }
    lamf = {
        "event_id": "l1", "event_type": "lamf", "onset": 35.0, "offset": 51.0,
    }
    spindle = {
        "event_id": "s1", "event_type": "spindle", "onset": 51.0, "offset": 52.0,
    }
    rows = [
        row(0, n2_spindle=1),
        row(
            1,
            lamf=16,
            arousal=4,
            n2_spindle=1,
            attributes={"n2_after_arousal_ids": ["s1"]},
            event_ids={"arousal": ["a1"], "lamf": ["l1"], "n2_spindle": ["s1"]},
        ),
    ]
    packets = [packet(), packet(arousal=[arousal], lamf=[lamf], structure=[spindle])]
    baseline, _ = engine(rows, packets, BASELINE)
    revised, _ = engine(
        rows,
        packets,
        config(post_arousal_reentry_policy="structure_reentry_first"),
    )
    assert baseline[1]["stage"] == "N1"
    assert baseline[1]["rule_id"] == "N2.4.POST_AROUSAL_MAJORITY"
    assert revised[1]["stage"] == "N2"
    assert revised[1]["rule_id"] == "N2.1.POST_AROUSAL_REENTRY"


def test_stage_specific_fallback_does_not_inherit_wake_without_wake_evidence():
    rows = [row(0, alpha=20), row(1)]
    baseline, _ = engine(rows, [packet(), packet()], BASELINE)
    revised, _ = engine(
        rows,
        [packet(), packet()],
        config(fallback_policy="aasm_stage_specific"),
    )
    assert [item["stage"] for item in baseline] == ["W", "W"]
    assert baseline[1]["rule_id"] == "FALLBACK.INHERIT_PREVIOUS"
    assert revised[1]["stage"] == "UNDETERMINED"
    assert "no AASM-supported" in revised[1]["abstention_reason"]


def test_stage_specific_fallback_retains_supported_n1_continuation():
    rows = [row(0, lamf=20, sem=1), row(1, lamf=20)]
    revised, _ = engine(
        rows,
        [packet(), packet()],
        config(fallback_policy="aasm_stage_specific"),
    )
    assert revised[0]["stage"] == "N1"
    assert revised[1]["stage"] == "N1"
    assert revised[1]["rule_id"] == "N1.CONTINUE"


def _rem_continuation_case(structure_onset: float):
    structure = {
        "event_id": "s1",
        "event_type": "spindle",
        "onset": structure_onset,
        "offset": structure_onset + 1,
    }
    rows = [
        row(0, lamf=20, rem=1, low_emg=20, rem_low_overlap=1),
        row(1, lamf=20, low_emg=20, spindle=1, event_ids={"spindle": ["s1"]}),
    ]
    return rows, [packet(), packet(structure=[structure])]


def test_rem_second_half_structure_does_not_block_r2_until_next_epoch():
    rows, packets = _rem_continuation_case(50.0)
    baseline, _ = engine(rows, packets, BASELINE)
    revised, _ = engine(
        rows,
        packets,
        config(rem_second_half_policy="second_half_structure_delayed"),
    )
    assert baseline[1]["rule_id"] == "FALLBACK.INHERIT_PREVIOUS"
    assert revised[1]["stage"] == "R"
    assert revised[1]["rule_id"] == "R.2"


def test_rem_first_half_structure_still_blocks_r2():
    rows, packets = _rem_continuation_case(40.0)
    revised, _ = engine(
        rows,
        packets,
        config(rem_second_half_policy="second_half_structure_delayed"),
    )
    assert revised[1]["stage"] == "R"
    assert revised[1]["rule_id"] == "FALLBACK.INHERIT_PREVIOUS"


def test_invalid_signal_never_creates_a_stage():
    invalid = row(0, alpha=20, slow_wave=10)
    invalid.signals_good = False
    decisions, _ = engine([invalid], [packet()], BASELINE)
    assert decisions[0]["stage"] == "UNDETERMINED"
    assert decisions[0]["quality_status"] == "INVALID_SIGNAL"
