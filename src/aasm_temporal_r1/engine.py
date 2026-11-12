"""R1: narrow temporal repair of the frozen enhanced engine.

Frozen assign/inherit/resolve_mbm are imported without alteration. The event
bank is label-free; no annotation/evaluation module is imported by this package.
"""
from dataclasses import asdict
from aasm_v2.evidence import aggregate
from aasm_v2.rules import assign, inherit, resolve_mbm, STAGES
from aasm_v2.types import StageDecision, Truth as T, truth
from .timeline import EndContext, index_packets, sweep, finalize_context


def stage(bank: dict) -> list[dict]:
    rows = aggregate(bank, "enhanced")
    return engine(rows, index_packets(bank, rows))


def engine(rows, packets) -> list[dict]:
    output, temporal_rows = [], []
    context, rem_anchor = EndContext(), None
    for i, (row, packet) in enumerate(zip(rows, packets)):
        previous = output[-1] if output else None
        prev_row = rows[i-1] if i else None
        prev_stage = previous.stage if previous else None
        decision = StageDecision(i, quality_status="AVAILABLE" if row.signals_good else "INVALID_SIGNAL")
        temporal = sweep(row, packet, context)
        if row.get("mbm").threshold(15) is T.TRUE:
            decision.abstention_reason = "movement pending independent lookahead"
            decision.previous_context["pending_mbm"] = True
            decision.previous_context["pre_movement_stage"] = (
                previous.previous_context.get("pre_movement_stage", previous.stage) if previous else None)
            output.append(decision)
            context, rem_anchor = EndContext(), None
            temporal["after"] = asdict(context)
            temporal_rows.append(temporal)
            continue
        def has(key): return row.get(key).present(False)
        def ge(key, val, inclusive=False): return row.get(key).threshold(val, inclusive)
        n3 = ge("slow_wave", 6, True)
        wakes = [("W.1",ge("alpha",15),["alpha"]),
                 ("W.2",ge("blink",15),["blink"]),
                 ("W.3",ge("irregular_rem_normal",15),["rem","normal_emg"])]
        wake = wakes[0][1] | wakes[1][1] | wakes[2][1]
        no_structure = ~has("spindle") & ~has("kcomplex")
        lamf = ge("lamf",15)
        rem = lamf & has("rem") & ge("low_emg",15) & ge("rem_low_overlap",0) & no_structure
        # Existing aggregate already excludes locally arousal-associated K.
        n2 = has("n2_spindle") | has("n2_kcomplex")
        # Arousal timing, when active, supersedes the event-only epoch shortcut.
        if temporal["engaged"]:
            n2 = T.FALSE
        alpha_path = truth(row.attributes.get("alpha_generator")) & lamf & ~ge("alpha",15)
        if prev_row is not None:
            now_alpha, prior_alpha = row.get("alpha").seconds, prev_row.get("alpha").seconds
            alpha_path &= truth(None if now_alpha is None or prior_alpha is None else now_alpha < prior_alpha)
        else:
            alpha_path = T.UNKNOWN
        nonalpha = ~truth(row.attributes.get("alpha_generator")) & (has("slowing")|has("vertex")|has("sem"))
        n2_reset = temporal["candidate_end"].phase == "POST_AROUSAL"
        tests = [("N3.1","N3",n3,["slow_wave"],"DEFINITE")]
        tests += [(rid,"W",v,keys,"DEFINITE") for rid,v,keys in wakes]
        if prev_stage == "R":
            tests.extend([
                ("R.3.EMG_N1","N1",ge("high_emg",15)&lamf,["high_emg","lamf"],"TERMINATION"),
                ("R.3.AROUSAL_SEM","N1",has("arousal_then_sem"),["arousal_then_sem"],"TERMINATION"),
                ("R.3.N2","N2",has("first_half_n2")&~has("rem"),["first_half_n2"],"TERMINATION")])
        if prev_stage == "N2":
            tests.append(("N2.4.MBM_SEM","N1",(prev_row.get("mbm").threshold(15) if prev_row else T.FALSE)&has("sem"),["sem"],"TERMINATION"))
        if previous and previous.previous_context.get("pre_movement_stage") == "N2":
            tests.append(("N2.4.MBM_SEM","N1",has("sem"),["sem"],"TERMINATION"))
        # R.1 now wins over the N2 arousal fragment assignment; its predicate is unchanged.
        tests.append(("R.1","R",rem,["lamf","rem","low_emg","rem_low_overlap"],"DEFINITE"))
        if temporal["engaged"]:
            tests.extend([
                ("N2.TEMPORAL_MAJORITY","N2",truth(temporal["seconds"]["N2"]>15),["arousal","lamf","n2_spindle","n2_kcomplex"],"CONTINUATION"),
                ("N2.4.POST_AROUSAL_MAJORITY","N1",truth(temporal["seconds"]["N1"]>15),["arousal","lamf"],"TERMINATION")])
        tests.extend([
            ("N2.1","N2",n2,["n2_spindle","n2_kcomplex"],"DEFINITE"),
            ("N1.1","N1",alpha_path,["alpha","lamf"],"DEFINITE"),
            ("N1.2","N1",nonalpha,["slowing","vertex","sem"],"DEFINITE")])
        previous_n2_structure = truth(prev_row.attributes.get("n2_event_in_epoch",False)) if prev_row else T.FALSE
        physiological_n2 = context.stage == "N2" and context.physiological_anchor_valid
        tests.extend([
            ("N2.3","N2",truth(prev_stage=="N3")&~n3&~has("arousal")&~wake&~rem,["slow_wave","arousal"],"CONTINUATION"),
            ("N2.2","N2",truth(physiological_n2 and not n2_reset)&previous_n2_structure&lamf&no_structure&~has("arousal"),["lamf","arousal"],"CONTINUATION"),
            ("R.2","R",truth(prev_stage=="R" and rem_anchor is not None)&lamf&ge("low_emg",15)&no_structure&~has("arousal")&~has("arousal_then_sem"),["lamf","low_emg","arousal"],"CONTINUATION")])
        for rid,label,value,keys,source in tests:
            if not row.signals_good: value &= T.UNKNOWN
            decision.trace.append({"rule_id":rid,"value":value.value,
                "input_status":{k:row.get(k).detector_status for k in keys},
                "n2_reentry_required":n2_reset,"evidence_policy":"candidate_proxy",
                "clinical_absence":"UNKNOWN_UNVALIDATED","negative_proposition":"no_detected_candidate"})
            if value is T.TRUE:
                interrupted_n2 = any(t.get("previous_phase") == "N2" for t in temporal["transitions"])
                if rid == "N2.4.POST_AROUSAL_MAJORITY" and not interrupted_n2:
                    source = "CONTINUATION"
                assign(decision,row,rid,label,keys,source,previous if source!="DEFINITE" else None)
                if rid in ("N2.TEMPORAL_MAJORITY","N2.4.POST_AROUSAL_MAJORITY"):
                    extra = set(context.origin_arousal_ids + context.physiological_anchor_event_ids)
                    extra.update(t["event_id"] for t in temporal["transitions"])
                    decision.evidence_event_ids = sorted(set(decision.evidence_event_ids)|extra)
                    # Output-label lineage is distinct from the physiological event anchor.
                    if previous is None or previous.stage != label:
                        decision.anchor,decision.chain_length = i,0
                if rid == "N1.1" and prev_row:
                    decision.previous_context.update(epoch=i-1,stage=prev_stage,
                        alpha_seconds=prev_row.get("alpha").seconds,current_alpha_seconds=row.get("alpha").seconds)
                    decision.evidence_event_ids += prev_row.get("alpha").event_ids
                if rid == "R.1": rem_anchor = i
                if prev_stage in ("N2","R") and label in ("W","N3"):
                    decision.previous_context["terminated_rule"] = "R.3" if prev_stage=="R" else "N2.4"
                break
        incompatible = has("arousal") is T.TRUE or n2_reset or has("arousal_then_sem") is T.TRUE
        if decision.stage not in STAGES and previous:
            # Exactly the frozen function, policy, chain setting and previous OUTPUT label.
            inherit(decision,previous,row,"INHERIT_PREVIOUS",incompatible,2)
        if decision.stage != "R": rem_anchor = None
        context = finalize_context(decision,temporal,row,packet,context)
        temporal["after"] = asdict(context)
        temporal["epoch_label"] = decision.stage
        temporal["prediction_anchor"] = decision.anchor
        temporal["label_state_disagree"] = decision.stage != context.stage
        output.append(decision)
        temporal_rows.append(temporal)
    resolve_mbm(output,rows)
    result=[]
    for d,t in zip(output,temporal_rows):
        item=d.to_dict()
        t={k:v for k,v in t.items() if k!="candidate_end"}
        item["temporal"]=t
        result.append(item)
    return result
