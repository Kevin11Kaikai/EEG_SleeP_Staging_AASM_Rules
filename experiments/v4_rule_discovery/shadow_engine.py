from __future__ import annotations

import sys
from dataclasses import asdict
from typing import Any

from .config import FROZEN_SRC, RuleConfig


if str(FROZEN_SRC) not in sys.path:
    sys.path.insert(0, str(FROZEN_SRC))

from aasm_temporal_r1.timeline import EndContext, finalize_context, index_packets, sweep  # noqa: E402
from aasm_v2.rules import STAGES, assign, inherit, resolve_mbm  # noqa: E402
from aasm_v2.types import StageDecision, Truth as T, truth  # noqa: E402


def build_packets(bank: dict, rows: list) -> list[dict]:
    packets = index_packets(bank, rows)
    events = {event["event_id"]: event for event in bank["events"]["enhanced"]}
    for row, packet in zip(rows, packets):
        packet["rem"] = [events[event_id] for event_id in row.get("rem").event_ids if event_id in events]
    return packets


def _validate_config(config: RuleConfig) -> None:
    allowed = {
        "rem_backward_policy": {"off", "aasm_v21_backward_bridge"},
        "wake_ocular_policy": {"duration15", "irregular_presence_with_tone", "blink_count6_or_irregular_with_tone"},
        "n1_alpha_policy": {"adjacent_attenuation", "persistent_alpha_majority_anchor"},
        "n1_nonalpha_policy": {"any_marker", "lamf_sem_vertex"},
        "n2_continuation_policy": {"prior_epoch_structure", "physiological_anchor"},
        "post_arousal_reentry_policy": {"temporal_majority_first", "structure_reentry_first"},
        "fallback_policy": {"generic_inherit", "aasm_stage_specific"},
        "rem_second_half_policy": {"all_structure_blocks", "second_half_structure_delayed"},
    }
    for field, values in allowed.items():
        value = getattr(config, field)
        if value not in values:
            raise ValueError(f"{field}={value}")


def _apply_rem_backward_bridge(
    output: list[StageDecision], rows: list, diagnostics: list[dict[str, Any]]
) -> None:
    """Apply the public AASM v2.1 definite-R backward bridge.

    The bridge never changes event evidence. It relabels only contiguous prior
    epochs with majority LAMF, majority low chin tone, and no competing
    structure, arousal, SEM, high tone, strong alpha, N3, Wake, or movement.
    """

    relabelled: set[int] = set()
    anchors = [index for index, decision in enumerate(output) if decision.rule_id == "R.1"]
    for anchor in anchors:
        index = anchor - 1
        while index >= 0:
            row = rows[index]
            decision = output[index]
            if decision.rule_id.startswith("MBM") or decision.stage in ("W", "N3"):
                break
            if not row.signals_good:
                break

            def has(key: str):
                return row.get(key).present(False)

            def ge(key: str, value: float, inclusive: bool = False):
                return row.get(key).threshold(value, inclusive)

            qualifies = (
                ge("lamf", 15)
                & ge("low_emg", 15)
                & ~ge("high_emg", 15)
                & ~ge("alpha", 15)
                & ~has("spindle")
                & ~has("kcomplex")
                & ~has("arousal")
                & ~has("arousal_then_sem")
                & ~has("sem")
                & ~ge("mbm", 15)
            )
            if qualifies is not T.TRUE:
                break
            if decision.stage != "R":
                decision.stage = "R"
                decision.rule_id = "R.BACKWARD"
                decision.decision_source = "CONTINUATION"
                decision.abstention_reason = None
                decision.anchor = anchor
                decision.chain_length = anchor - index
                decision.evidence_event_ids = sorted(
                    set(row.get("lamf").event_ids + row.get("low_emg").event_ids)
                )
                decision.previous_context.update(
                    future_definite_r_epoch=anchor,
                    backward_bridge=True,
                    original_stage=diagnostics[index].get("pre_backward_stage"),
                )
                decision.trace.append(
                    {
                        "rule_id": "R.BACKWARD",
                        "value": "TRUE",
                        "source": "AASM_V2_1_REM_BACKWARD",
                        "future_definite_r_epoch": anchor,
                    }
                )
                relabelled.add(index)
            index -= 1
    for index in relabelled:
        diagnostics[index]["rem_backward_relabelled"] = True


def engine(
    rows: list,
    packets: list[dict],
    config: RuleConfig,
) -> tuple[list[dict], list[dict[str, Any]]]:
    """Run a frozen-input shadow engine; C000 exactly reproduces Temporal R1."""

    _validate_config(config)
    output: list[StageDecision] = []
    temporal_rows: list[dict] = []
    diagnostics: list[dict[str, Any]] = []
    context, rem_anchor = EndContext(), None
    last_alpha_majority_epoch: int | None = None
    last_alpha_majority_seconds: float | None = None

    for i, (row, packet) in enumerate(zip(rows, packets)):
        previous = output[-1] if output else None
        prev_row = rows[i - 1] if i else None
        prev_stage = previous.stage if previous else None
        if prev_stage in ("N2", "N3", "R"):
            last_alpha_majority_epoch = None
            last_alpha_majority_seconds = None
        decision = StageDecision(i, quality_status="AVAILABLE" if row.signals_good else "INVALID_SIGNAL")
        temporal = sweep(row, packet, context)
        if row.get("mbm").threshold(15) is T.TRUE:
            decision.abstention_reason = "movement pending independent lookahead"
            decision.previous_context["pending_mbm"] = True
            decision.previous_context["pre_movement_stage"] = (
                previous.previous_context.get("pre_movement_stage", previous.stage) if previous else None
            )
            output.append(decision)
            context, rem_anchor = EndContext(), None
            temporal["after"] = asdict(context)
            temporal_rows.append(temporal)
            diagnostics.append({"epoch_index": i, "pre_backward_stage": decision.stage, "rem_backward_relabelled": False})
            continue

        def has(key: str):
            return row.get(key).present(False)

        def ge(key: str, value: float, inclusive: bool = False):
            return row.get(key).threshold(value, inclusive)

        n3 = ge("slow_wave", 6, True)
        tone = ge("normal_emg", 15) | ge("high_emg", 15)
        if config.wake_ocular_policy == "duration15":
            blink_wake = ge("blink", 15)
            irregular_wake = ge("irregular_rem_normal", 15)
            blink_keys = ["blink"]
            irregular_keys = ["rem", "normal_emg"]
        else:
            blink_wake = ge("blink", 15)
            if config.wake_ocular_policy == "blink_count6_or_irregular_with_tone":
                blink_wake = truth(len(row.get("blink").event_ids) >= 6) & tone
            irregular_wake = ge("irregular_rem_normal", 0) & tone
            blink_keys = ["blink", "normal_emg", "high_emg"]
            irregular_keys = ["rem", "normal_emg", "high_emg"]
        wakes = [
            ("W.1", ge("alpha", 15), ["alpha"]),
            ("W.2", blink_wake, blink_keys),
            ("W.3", irregular_wake, irregular_keys),
        ]
        wake = wakes[0][1] | wakes[1][1] | wakes[2][1]
        no_structure = ~has("spindle") & ~has("kcomplex")
        lamf = ge("lamf", 15)
        rem = lamf & has("rem") & ge("low_emg", 15) & ge("rem_low_overlap", 0) & no_structure
        n2 = has("n2_spindle") | has("n2_kcomplex")
        if temporal["engaged"]:
            n2 = T.FALSE

        if config.n1_alpha_policy == "adjacent_attenuation":
            alpha_path = truth(row.attributes.get("alpha_generator")) & lamf & ~ge("alpha", 15)
            if prev_row is not None:
                now_alpha, prior_alpha = row.get("alpha").seconds, prev_row.get("alpha").seconds
                alpha_path &= truth(None if now_alpha is None or prior_alpha is None else now_alpha < prior_alpha)
            else:
                alpha_path = T.UNKNOWN
        else:
            alpha_path = (
                truth(last_alpha_majority_epoch is not None)
                & lamf
                & ~ge("alpha", 15)
                & truth(prev_stage in ("W", "N1"))
            )

        if config.n1_nonalpha_policy == "any_marker":
            nonalpha = ~truth(row.attributes.get("alpha_generator")) & (
                has("slowing") | has("vertex") | has("sem")
            )
            nonalpha_keys = ["slowing", "vertex", "sem"]
        else:
            nonalpha = ~truth(row.attributes.get("alpha_generator")) & lamf & (
                has("vertex") | has("sem")
            )
            nonalpha_keys = ["vertex", "sem", "lamf"]

        n2_reset = temporal["candidate_end"].phase == "POST_AROUSAL"
        stage_tests = [("N3.1", "N3", n3, ["slow_wave"], "DEFINITE")]
        stage_tests += [(rid, "W", value, keys, "DEFINITE") for rid, value, keys in wakes]
        if prev_stage == "R":
            stage_tests.extend(
                [
                    ("R.3.EMG_N1", "N1", ge("high_emg", 15) & lamf, ["high_emg", "lamf"], "TERMINATION"),
                    ("R.3.AROUSAL_SEM", "N1", has("arousal_then_sem"), ["arousal_then_sem"], "TERMINATION"),
                    ("R.3.N2", "N2", has("first_half_n2") & ~has("rem"), ["first_half_n2"], "TERMINATION"),
                ]
            )
        if prev_stage == "N2":
            stage_tests.append(
                ("N2.4.MBM_SEM", "N1", (prev_row.get("mbm").threshold(15) if prev_row else T.FALSE) & has("sem"), ["sem"], "TERMINATION")
            )
        if previous and previous.previous_context.get("pre_movement_stage") == "N2":
            stage_tests.append(("N2.4.MBM_SEM", "N1", has("sem"), ["sem"], "TERMINATION"))
        stage_tests.append(("R.1", "R", rem, ["lamf", "rem", "low_emg", "rem_low_overlap"], "DEFINITE"))

        post_arousal_ids = list(row.attributes.get("n2_after_arousal_ids", []))
        if temporal["engaged"]:
            if config.post_arousal_reentry_policy == "structure_reentry_first":
                stage_tests.append(
                    ("N2.1.POST_AROUSAL_REENTRY", "N2", truth(bool(post_arousal_ids)), ["n2_spindle", "n2_kcomplex"], "DEFINITE")
                )
            stage_tests.extend(
                [
                    ("N2.TEMPORAL_MAJORITY", "N2", truth(temporal["seconds"]["N2"] > 15), ["arousal", "lamf", "n2_spindle", "n2_kcomplex"], "CONTINUATION"),
                    ("N2.4.POST_AROUSAL_MAJORITY", "N1", truth(temporal["seconds"]["N1"] > 15), ["arousal", "lamf"], "TERMINATION"),
                ]
            )
        stage_tests.extend(
            [
                ("N2.1", "N2", n2, ["n2_spindle", "n2_kcomplex"], "DEFINITE"),
                ("N1.1", "N1", alpha_path, ["alpha", "lamf"], "DEFINITE"),
                ("N1.2", "N1", nonalpha, nonalpha_keys, "DEFINITE"),
            ]
        )

        previous_n2_structure = truth(prev_row.attributes.get("n2_event_in_epoch", False)) if prev_row else T.FALSE
        physiological_n2 = context.stage == "N2" and context.physiological_anchor_valid
        n2_continuation = truth(physiological_n2 and not n2_reset) & lamf & no_structure & ~has("arousal")
        if config.n2_continuation_policy == "prior_epoch_structure":
            n2_continuation &= previous_n2_structure
        if config.rem_second_half_policy == "all_structure_blocks":
            rem_continuation_clear = no_structure
        else:
            boundary = i * 30 + 15
            first_half = any(event["onset"] < boundary for event in packet.get("structure", []))
            rem_continuation_clear = truth(not first_half and not packet.get("carry_structure"))
        stage_tests.extend(
            [
                ("N2.3", "N2", truth(prev_stage == "N3") & ~n3 & ~has("arousal") & ~wake & ~rem, ["slow_wave", "arousal"], "CONTINUATION"),
                ("N2.2", "N2", n2_continuation, ["lamf", "arousal"], "CONTINUATION"),
                ("R.2", "R", truth(prev_stage == "R" and rem_anchor is not None) & lamf & ge("low_emg", 15) & rem_continuation_clear & ~has("arousal") & ~has("arousal_then_sem"), ["lamf", "low_emg", "arousal"], "CONTINUATION"),
            ]
        )

        for rid, label, value, keys, source in stage_tests:
            if not row.signals_good:
                value &= T.UNKNOWN
            decision.trace.append(
                {
                    "rule_id": rid,
                    "value": value.value,
                    "input_status": {key: row.get(key).detector_status for key in keys},
                    "n2_reentry_required": n2_reset,
                    "evidence_policy": "candidate_proxy",
                    "clinical_absence": "UNKNOWN_UNVALIDATED",
                    "negative_proposition": "no_detected_candidate",
                }
            )
            if value is T.TRUE:
                interrupted_n2 = any(item.get("previous_phase") == "N2" for item in temporal["transitions"])
                if rid == "N2.4.POST_AROUSAL_MAJORITY" and not interrupted_n2:
                    source = "CONTINUATION"
                assign(decision, row, rid, label, keys, source, previous if source != "DEFINITE" else None)
                if rid in ("N2.TEMPORAL_MAJORITY", "N2.4.POST_AROUSAL_MAJORITY"):
                    extra = set(context.origin_arousal_ids + context.physiological_anchor_event_ids)
                    extra.update(item["event_id"] for item in temporal["transitions"])
                    decision.evidence_event_ids = sorted(set(decision.evidence_event_ids) | extra)
                    if previous is None or previous.stage != label:
                        decision.anchor, decision.chain_length = i, 0
                if rid == "N1.1" and prev_row:
                    decision.previous_context.update(
                        epoch=i - 1,
                        stage=prev_stage,
                        alpha_seconds=prev_row.get("alpha").seconds,
                        current_alpha_seconds=row.get("alpha").seconds,
                    )
                    if config.n1_alpha_policy == "persistent_alpha_majority_anchor":
                        decision.previous_context.update(
                            alpha_anchor_epoch=last_alpha_majority_epoch,
                            alpha_anchor_seconds=last_alpha_majority_seconds,
                        )
                    decision.evidence_event_ids += prev_row.get("alpha").event_ids
                if rid == "R.1":
                    rem_anchor = i
                if prev_stage in ("N2", "R") and label in ("W", "N3"):
                    decision.previous_context["terminated_rule"] = "R.3" if prev_stage == "R" else "N2.4"
                break

        incompatible = has("arousal") is T.TRUE or n2_reset or has("arousal_then_sem") is T.TRUE
        if decision.stage not in STAGES and previous:
            if config.fallback_policy == "generic_inherit":
                inherit(decision, previous, row, "INHERIT_PREVIOUS", incompatible, 2)
            else:
                if prev_stage == "N1" and lamf is T.TRUE and wake is not T.TRUE and rem is not T.TRUE and n3 is not T.TRUE:
                    assign(decision, row, "N1.CONTINUE", "N1", ["lamf"], "CONTINUATION", previous)
                elif prev_stage == "N2" and n2_continuation is T.TRUE:
                    assign(decision, row, "N2.CONTINUE", "N2", ["lamf", "arousal"], "CONTINUATION", previous)
                else:
                    decision.abstention_reason = "no AASM-supported stage-specific continuation"
        if decision.stage != "R":
            rem_anchor = None
        context = finalize_context(decision, temporal, row, packet, context)
        temporal["after"] = asdict(context)
        temporal["epoch_label"] = decision.stage
        temporal["prediction_anchor"] = decision.anchor
        temporal["label_state_disagree"] = decision.stage != context.stage
        output.append(decision)
        temporal_rows.append(temporal)
        diagnostics.append({"epoch_index": i, "pre_backward_stage": decision.stage, "rem_backward_relabelled": False})
        if ge("alpha", 15) is T.TRUE:
            last_alpha_majority_epoch = i
            last_alpha_majority_seconds = row.get("alpha").seconds

    resolve_mbm(output, rows)
    if config.rem_backward_policy == "aasm_v21_backward_bridge":
        _apply_rem_backward_bridge(output, rows, diagnostics)

    result = []
    for decision, temporal in zip(output, temporal_rows):
        item = decision.to_dict()
        item["temporal"] = {key: value for key, value in temporal.items() if key != "candidate_end"}
        result.append(item)
    return result, diagnostics
