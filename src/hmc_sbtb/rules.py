"""Three-valued, ordered, immutable pass-1 and chronological pass-2 rules."""
from copy import deepcopy
import math
from .temporal import Tri, evidence

STAGES = ["W", "N1", "N2", "N3", "R"]
UNKNOWN = "UNDETERMINED"
ORDER = ["N3", "W", "R", "N2", "N1"]


def adjacent(previous, current):
    return (previous is not None and previous["record_id"] == current["record_id"] and
            previous["subject_id"] == current["subject_id"] and
            previous["epoch_index"] + 1 == current["epoch_index"] and
            previous["end_sec"] == current["start_sec"])


def condition(row, name, predicate):
    value = row.get(name + "_sec")
    if value is None or not math.isfinite(float(value)):
        return Tri.UNKNOWN
    return evidence(predicate(value), row.get(name + "_available", False))


def definite(features, cfg):
    outputs, traces = [], []
    for i, f in enumerate(features):
        executed = []
        prediction, rule_id, ids = UNKNOWN, "NONE", []
        previous = outputs[i - 1] if i else None
        if not adjacent(previous, f):
            previous = None
        for stage in ORDER:
            if prediction != UNKNOWN:
                executed.append({"stage": stage, "status": "NOT_EVALUATED"})
                continue
            if stage == "N3":
                terms = {"swa_gt_6": condition(f, "slow_wave", lambda x: x > cfg["rules"]["sw_coverage_gt_sec"])}
                event_ids = f["slow_wave_ids"]
            elif stage == "W":
                terms = {"alpha_gt_half": condition(f, "alpha", lambda x: x > cfg["rules"]["wake_alpha_gt_sec"])}
                event_ids = f["alpha_ids"]
            elif stage == "R":
                terms = {
                    "lamf": condition(f, "lamf", lambda x: x >= cfg["rules"]["rem_lamf_min_sec"]),
                    "eye_movements": condition(f, "rem_candidate", lambda x: x > 0),
                    "low_emg": condition(f, "low_emg", lambda x: x >= cfg["rules"]["rem_low_emg_min_sec"]),
                    "eye_low_coincidence": evidence(f["rem_low_overlap_sec"] >= cfg["rules"]["rem_eye_low_overlap_min_sec"],
                        f["rem_candidate_available"] and f["low_emg_available"]),
                    "no_spindle": ~condition(f, "spindle", lambda x: x > 0),
                    "no_kcomplex": ~condition(f, "kcomplex", lambda x: x > 0),
                }
                event_ids = f["lamf_ids"] + f["rem_candidate_ids"] + f["low_emg_ids"]
            elif stage == "N2":
                spindle = evidence(bool(f["n2_spindle_ids"]), f["n2_window_available"])
                k = evidence(bool(f["n2_kcomplex_ids"]), f["n2_kcomplex_available"])
                terms = {"spindle_or_unassociated_k": spindle | k}
                event_ids = f["n2_spindle_ids"] + f["n2_kcomplex_ids"]
            else:
                prev_alpha = (condition(features[i - 1], "alpha", lambda x: x > 0) if previous else Tri.FALSE)
                terms = {
                    "alpha_generator": evidence(bool(f["generates_alpha_rhythm"]), f["generates_alpha_rhythm"] is not None),
                    "previous_actual_pass1_wake": evidence(previous is not None and previous["pass1_prediction"] == "W"),
                    "lamf_gt_half": condition(f, "lamf", lambda x: x > cfg["rules"]["n1_lamf_gt_sec"]),
                    "alpha_current_or_previous": condition(f, "alpha", lambda x: x > 0) | prev_alpha,
                }
                event_ids = f["lamf_ids"] + f["alpha_ids"]
            result = Tri.TRUE
            for value in terms.values():
                result = result & value
            if f["quality_barrier"]:
                result = result & Tri.UNKNOWN
            executed.append({"stage": stage, "status": "EVALUATED", "conditions": terms,
                             "result": result, "event_ids": event_ids,
                             "missing_evidence": [k for k, v in terms.items() if v == Tri.UNKNOWN]})
            if result == Tri.TRUE:
                prediction, rule_id, ids = stage, "P1_" + stage, event_ids
        outputs.append({k: f[k] for k in ["record_id", "subject_id", "epoch_index", "start_sec", "end_sec", "quality_status"]} |
                       {"pass1_prediction": prediction, "final_prediction": prediction,
                        "decision_source": "DEFINITE" if prediction != UNKNOWN else UNKNOWN,
                        "rule_id": rule_id, "event_ids": ids, "parent_epoch": None,
                        "anchor_epoch": i if prediction != UNKNOWN else None,
                        "chain_length": 0, "original_rule_id": rule_id})
        traces.append({"record_id": f["record_id"], "epoch_index": i, "actual_execution": executed,
                       "features": f, "thresholds": cfg["rules"], "pass1_prediction": prediction})
    return outputs, traces


def run_rules(features, cfg, variant="C"):
    outputs, traces = definite(features, cfg)
    # deepcopy prevents A/B/C state and pass-1 traces from sharing mutable arrays.
    outputs, traces = deepcopy(outputs), deepcopy(traces)
    for i, (out, f, trace) in enumerate(zip(outputs, features, traces)):
        checks = []
        previous = outputs[i - 1] if i else None
        if not adjacent(previous, f):
            previous = None
        if variant != "A" and out["pass1_prediction"] == UNKNOWN and not f["quality_barrier"]:
            for stage in cfg["rules"]["continuity_order"]:
                if out["final_prediction"] != UNKNOWN:
                    checks.append({"stage": stage, "status": "NOT_EVALUATED"})
                    continue
                terms = {"previous_stage": evidence(previous is not None and previous["final_prediction"] == stage),
                         "no_arousal_candidate": ~condition(f, "arousal_candidate", lambda x: x > 0),
                         "no_wake_alpha": ~condition(f, "alpha", lambda x: x > cfg["rules"]["wake_alpha_gt_sec"]),
                         "no_n3": ~condition(f, "slow_wave", lambda x: x > cfg["rules"]["sw_coverage_gt_sec"])}
                if stage == "R":
                    terms.update({
                        "lamf": condition(f, "lamf", lambda x: x >= cfg["rules"]["rem_lamf_min_sec"]),
                        "low_emg": condition(f, "low_emg", lambda x: x >= cfg["rules"]["rem_low_emg_min_sec"]),
                        "no_spindle": ~condition(f, "spindle", lambda x: x > 0),
                        "no_kcomplex": ~condition(f, "kcomplex", lambda x: x > 0)})
                else:
                    rem = (condition(f, "rem_candidate", lambda x: x > 0) &
                           condition(f, "low_emg", lambda x: x >= cfg["rules"]["rem_low_emg_min_sec"]))
                    terms["no_rem_eye_low_emg"] = ~rem
                result = Tri.TRUE
                for value in terms.values():
                    result = result & value
                checks.append({"stage": stage, "status": "EVALUATED", "conditions": terms, "result": result})
                if result == Tri.TRUE:
                    out.update({"final_prediction": stage, "decision_source": "CONTINUATION", "rule_id": "P2_" + stage,
                                "parent_epoch": i - 1, "anchor_epoch": previous["anchor_epoch"],
                                "chain_length": previous["chain_length"] + 1,
                                "original_rule_id": previous["original_rule_id"]})
            if (variant == "C" and cfg["rules"]["inheritance"] and out["final_prediction"] == UNKNOWN and
                    previous is not None and previous["final_prediction"] != UNKNOWN and
                    previous["quality_status"] != "INVALID_SIGNAL"):
                out.update({"final_prediction": previous["final_prediction"], "decision_source": "INHERITANCE",
                            "rule_id": "PREVIOUS_STAGE", "parent_epoch": i - 1,
                            "anchor_epoch": previous["anchor_epoch"], "chain_length": previous["chain_length"] + 1,
                            "original_rule_id": previous["original_rule_id"]})
        trace["pass2_execution"] = checks
        trace["previous_final_prediction"] = previous["final_prediction"] if previous else None
        trace["decision"] = deepcopy(out)
    return outputs, traces
