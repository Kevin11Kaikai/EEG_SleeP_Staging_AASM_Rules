"""Deterministic rule engines. Only EpochEvidence enters this API."""
from copy import deepcopy
from ..types import EpochEvidence, StageDecision, Truth as T, truth

STAGES = ("W", "N1", "N2", "N3", "R")


def original_engine(rows: list[EpochEvidence], fallback: str = "INHERIT_PREVIOUS") -> list[StageDecision]:
    first = []
    for i, row in enumerate(rows):
        prior = rows[i - 1] if i else None
        prior_stage = first[i - 1].stage if i else None
        def has(key):
            return row.get(key).present(conservative=False)
        def limit(key, value, inclusive=False):
            return row.get(key).threshold(value, inclusive)
        tests = [
            ("ORIGINAL.N3", "N3", limit("slow_wave", 6), ["slow_wave"]),
            ("ORIGINAL.W", "W", limit("alpha", 15), ["alpha"]),
            ("ORIGINAL.R", "R", limit("lamf", 1, True) & has("rem") &
             limit("low_emg", 15, True) & limit("rem_low_overlap", .01, True) &
             ~has("spindle") & ~has("kcomplex"), ["lamf", "rem", "low_emg"]),
            ("ORIGINAL.N2", "N2", has("n2_spindle") | has("n2_kcomplex"), ["n2_spindle", "n2_kcomplex"]),
            ("ORIGINAL.N1", "N1", truth(row.attributes.get("alpha_generator")) &
             truth(prior_stage == "W") & limit("lamf", 15) &
             (has("alpha") | (prior.get("alpha").present(False) if prior else T.FALSE)), ["alpha", "lamf"])
        ]
        decision = StageDecision(i, quality_status="AVAILABLE" if row.signals_good else "INVALID_SIGNAL")
        for rid, stage, value, keys in tests:
            value = value & (T.TRUE if row.signals_good else T.UNKNOWN)
            decision.trace.append({"rule_id": rid, "value": value.value,
                                   "candidate_absence_is_clinical_negative": False})
            if value is T.TRUE:
                assign(decision, row, rid, stage, keys, "DEFINITE")
                if rid == "ORIGINAL.N1" and prior:
                    decision.previous_context.update(epoch=i-1, stage=prior_stage,
                        alpha_seconds=prior.get("alpha").seconds)
                    decision.evidence_event_ids += prior.get("alpha").event_ids
                break
        first.append(decision)
    final = deepcopy(first)
    for i, row in enumerate(rows):
        decision = final[i]
        if decision.stage in STAGES or not row.signals_good or i == 0:
            continue
        previous = final[i - 1]
        def has(key):
            return row.get(key).present(False)
        common = ~has("arousal") & ~row.get("alpha").threshold(15) & ~row.get("slow_wave").threshold(6)
        rem = (truth(previous.stage == "R") & common & row.get("lamf").threshold(1, True)
               & row.get("low_emg").threshold(15, True) & ~has("spindle") & ~has("kcomplex"))
        n2 = truth(previous.stage == "N2") & common & ~(has("rem") & row.get("low_emg").threshold(15, True))
        for rid, stage, value in (("ORIGINAL.R_CONT", "R", rem), ("ORIGINAL.N2_CONT", "N2", n2)):
            decision.trace.append({"rule_id": rid, "value": value.value})
            if value is T.TRUE:
                assign(decision, row, rid, stage, ["lamf", "low_emg", "arousal"], "CONTINUATION", previous)
                break
        if decision.stage not in STAGES:
            inherit(decision, previous, row, fallback, False, 2)
    return final


def assign(d, row, rule, stage, keys, source, previous=None):
    d.stage, d.rule_id, d.decision_source, d.abstention_reason = stage, rule, source, None
    d.evidence_event_ids = sorted(set(e for key in keys for e in row.get(key).event_ids))
    d.anchor = d.epoch_index if source in ("DEFINITE", "TERMINATION") else (previous.anchor if previous else d.epoch_index)
    d.chain_length = 0 if source in ("DEFINITE", "TERMINATION") else (previous.chain_length + 1 if previous else 0)
    if previous:
        d.previous_context.update(epoch=previous.epoch_index, stage=previous.stage,
                                  rule=previous.rule_id, anchor=previous.anchor)


def inherit(d, previous, row, policy, incompatible, maximum):
    d.previous_context.update(epoch=previous.epoch_index, stage=previous.stage,
        anchor=previous.anchor, previous_chain_length=previous.chain_length,
        incompatible_evidence=bool(incompatible), evidence_available=row.signals_good,
        fallback_policy=policy)
    can = row.signals_good and previous.stage in STAGES and previous.quality_status != "INVALID_SIGNAL"
    if policy == "ABSTAIN":
        d.abstention_reason = "fallback=ABSTAIN; no supported definite/temporal rule"
        return
    if policy == "CONDITIONAL_INHERIT":
        can &= not incompatible and previous.chain_length < maximum
    if can:
        assign(d, row, "FALLBACK." + policy, previous.stage, [], "INHERITANCE", previous)
        d.previous_context["fallback_reason"] = "no TRUE rule; not new physiological evidence"
    else:
        d.abstention_reason = "invalid/gap/unknown history or conditional inheritance limit"


def enhanced_engine(rows: list[EpochEvidence], fallback: str = "INHERIT_PREVIOUS",
                    maximum_chain: int = 2, negative_policy: str = "candidate_proxy") -> list[StageDecision]:
    if negative_policy not in ("candidate_proxy", "clinical_strict"):
        raise ValueError("Unknown evidence policy")
    output = []
    n2_reset = False
    last_arousal_end = None
    rem_anchor = None
    for i, row in enumerate(rows):
        previous = output[-1] if output else None
        prev_row = rows[i - 1] if i else None
        prev_stage = previous.stage if previous else None
        decision = StageDecision(i, quality_status="AVAILABLE" if row.signals_good else "INVALID_SIGNAL")
        if row.get("mbm").threshold(15) is T.TRUE:
            # Do not allow a movement epoch to seed its own future resolution.
            decision.abstention_reason = "movement pending independent lookahead"
            decision.previous_context["pending_mbm"] = True
            decision.previous_context["pre_movement_stage"] = (
                previous.previous_context.get("pre_movement_stage", previous.stage) if previous else None)
            output.append(decision)
            n2_reset, rem_anchor = True, None
            continue
        def has(key):
            return row.get(key).present(negative_policy == "clinical_strict")
        def ge(key, threshold, inclusive=False):
            return row.get(key).threshold(threshold, inclusive)
        n3 = ge("slow_wave", 6, True)
        wakes = [("W.1", ge("alpha", 15), ["alpha"]),
                 ("W.2", ge("blink", 15), ["blink"]),
                 ("W.3", ge("irregular_rem_normal", 15), ["rem", "normal_emg"])]
        wake = wakes[0][1] | wakes[1][1] | wakes[2][1]
        no_structure = ~has("spindle") & ~has("kcomplex")
        lamf = ge("lamf", 15)
        rem = lamf & has("rem") & ge("low_emg", 15) & ge("rem_low_overlap", 0) & no_structure
        n2 = has("n2_spindle") | (has("n2_kcomplex") & ~has("arousal"))
        # K unassociated absence remains unknown without validated arousal negatives.
        alpha_path = truth(row.attributes.get("alpha_generator")) & lamf & ~ge("alpha", 15)
        if prev_row is not None:
            now_alpha, prior_alpha = row.get("alpha").seconds, prev_row.get("alpha").seconds
            attenuation = None if now_alpha is None or prior_alpha is None else now_alpha < prior_alpha
            alpha_path = alpha_path & truth(attenuation)
        else:
            alpha_path = T.UNKNOWN
        nonalpha = ~truth(row.attributes.get("alpha_generator")) & (has("slowing") | has("vertex") | has("sem"))
        # Positive arousal invalidates N2 continuation until a new structural anchor.
        aroused = has("arousal") is T.TRUE
        after_ids = row.attributes.get("n2_after_arousal_ids", [])
        event_end = row.attributes.get("last_arousal_end")
        if event_end is not None:
            last_arousal_end = max(last_arousal_end or 0, event_end)
        if last_arousal_end is not None and "n2_window_onsets" in row.attributes:
            after_ids = [eid for eid, onset in row.attributes["n2_window_onsets"] if onset >= last_arousal_end]
        if aroused and prev_stage == "N2":
            n2_reset = True
        if n2_reset and not after_ids:
            n2 = T.FALSE
        tests = [("N3.1", "N3", n3, ["slow_wave"], "DEFINITE")]
        tests += [(rid, "W", value, keys, "DEFINITE") for rid, value, keys in wakes]
        # Termination precedes compatible onset and all continuation.
        if prev_stage == "R":
            tests.extend([
                ("R.3.EMG_N1", "N1", ge("high_emg", 15) & lamf, ["high_emg", "lamf"], "TERMINATION"),
                ("R.3.AROUSAL_SEM", "N1", has("arousal_then_sem"), ["arousal_then_sem"], "TERMINATION"),
                ("R.3.N2", "N2", has("first_half_n2") & ~has("rem"), ["first_half_n2"], "TERMINATION")])
        if prev_stage == "N2":
            tests.extend([
                ("N2.4.AROUSAL", "N1", has("arousal") & lamf & truth(not after_ids), ["arousal", "lamf"], "TERMINATION"),
                ("N2.4.MBM_SEM", "N1", (prev_row.get("mbm").threshold(15) if prev_row else T.FALSE) & has("sem"), ["sem"], "TERMINATION")])
        if previous and previous.previous_context.get("pre_movement_stage") == "N2":
            tests.append(("N2.4.MBM_SEM", "N1", has("sem"), ["sem"], "TERMINATION"))
        tests.append(("R.1", "R", rem, ["lamf", "rem", "low_emg", "rem_low_overlap"], "DEFINITE"))
        tests.extend([
            ("N2.1", "N2", n2, ["n2_spindle", "n2_kcomplex"], "DEFINITE"),
            ("N1.1", "N1", alpha_path, ["alpha", "lamf"], "DEFINITE"),
            ("N1.2", "N1", nonalpha, ["slowing", "vertex", "sem"], "DEFINITE")])
        no_arousal = ~has("arousal")
        previous_n2_structure = truth(prev_row.attributes.get("n2_event_in_epoch", False)) if prev_row else T.FALSE
        tests.extend([
            ("N2.3", "N2", truth(prev_stage == "N3") & ~n3 & no_arousal & ~wake & ~rem,
             ["slow_wave", "arousal"], "CONTINUATION"),
            ("N2.2", "N2", truth(prev_stage == "N2" and not n2_reset) & previous_n2_structure &
             lamf & no_structure & no_arousal, ["lamf", "arousal"], "CONTINUATION"),
            ("R.2", "R", truth(prev_stage == "R" and rem_anchor is not None) & lamf &
             ge("low_emg", 15) & no_structure & no_arousal & ~has("arousal_then_sem"),
             ["lamf", "low_emg", "arousal"], "CONTINUATION")])
        for rid, stage, value, keys, source in tests:
            if not row.signals_good:
                value = value & T.UNKNOWN
            decision.trace.append({"rule_id": rid, "value": value.value,
                "input_status": {k: row.get(k).detector_status for k in keys},
                "n2_reentry_required": n2_reset, "evidence_policy": negative_policy,
                "clinical_absence": "UNKNOWN_UNVALIDATED",
                "negative_proposition": "no_detected_candidate" if negative_policy == "candidate_proxy" else "clinical_absence"})
            if value is T.TRUE:
                assign(decision, row, rid, stage, keys, source, previous if source != "DEFINITE" else None)
                if rid == "N1.1" and prev_row:
                    decision.previous_context.update(epoch=i-1, stage=prev_stage,
                        alpha_seconds=prev_row.get("alpha").seconds,
                        current_alpha_seconds=row.get("alpha").seconds)
                    decision.evidence_event_ids += prev_row.get("alpha").event_ids
                if rid == "R.1":
                    rem_anchor = i
                if rid == "N2.1" or rid == "R.3.N2":
                    n2_reset = False
                if prev_stage in ("N2", "R") and stage in ("W", "N3"):
                    decision.previous_context["terminated_rule"] = "R.3" if prev_stage == "R" else "N2.4"
                break
        incompatible = aroused or n2_reset or has("arousal_then_sem") is T.TRUE
        if decision.stage not in STAGES and previous:
            inherit(decision, previous, row, fallback, incompatible, maximum_chain)
        if decision.stage != "R":
            rem_anchor = None
        if not row.signals_good:
            n2_reset, rem_anchor = False, None
        output.append(decision)
    resolve_mbm(output, rows)
    return output


def resolve_mbm(decisions: list[StageDecision], rows: list[EpochEvidence]) -> None:
    """Resolve each contiguous movement block once from nonmovement base predictions.

    Future context is used ONLY here. Tail/unknown destination -> abstain, no cycle.
    """
    flags = [row.get("mbm").threshold(15) is T.TRUE for row in rows]
    i = 0
    while i < len(rows):
        if not flags[i]:
            i += 1
            continue
        j = i + 1
        while j < len(rows) and flags[j]:
            j += 1
        left = decisions[i - 1].stage if i else None
        right = decisions[j].stage if j < len(rows) else None
        alpha = any(rows[k].get("alpha").threshold(0) is T.TRUE for k in range(i, j))
        wake = alpha or left == "W" or right == "W"
        for k in range(i, j):
            d = decisions[k]
            d.previous_context.update(mbm_block=[i, j], future_prediction_epoch=j if j < len(rows) else None,
                                       future_prediction_stage=right, base_stage=d.stage)
            d.trace.append({"rule_id": "MBM.1" if wake else "MBM.2", "value": "TRUE",
                            "future_source": "algorithm_nonmovement_base_prediction"})
            if wake or right in STAGES:
                assign(d, rows[k], "MBM.1" if wake else "MBM.2", "W" if wake else right,
                       ["mbm", "alpha"] if wake else ["mbm"], "MOVEMENT_LOOKAHEAD")
            else:
                d.stage, d.rule_id, d.decision_source = "UNDETERMINED", "MBM.2.UNKNOWN", "ABSTENTION"
                d.abstention_reason = "no known future nonmovement algorithm prediction"
        i = j
