"""N2-arousal fragment chronology, separate from 30-second output labels.

No reference annotations or signal thresholds enter this module. Only the
frozen event intervals and their existing availability/association flags enter.
"""
from copy import deepcopy
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from aasm_v2.types import EpochEvidence


@dataclass
class EndContext:
    stage: str = "UNDETERMINED"
    phase: str = "OTHER"  # OTHER, N2, POST_AROUSAL
    repair_active: bool = False
    pending_arousal_end: float | None = None
    physiological_anchor_valid: bool = False
    physiological_anchor_epoch: int | None = None
    physiological_anchor_event_ids: list[str] = field(default_factory=list)
    origin_arousal_ids: list[str] = field(default_factory=list)
    last_arousal_ids: list[str] = field(default_factory=list)


def index_packets(bank: dict, rows: list[EpochEvidence]) -> list[dict]:
    """Add continuous times without changing frozen per-epoch evidence."""
    events = {e["event_id"]: e for e in bank["events"]["enhanced"]}
    structural = defaultdict(list)
    for e in events.values():
        if e["event_type"] in ("spindle", "kcomplex"):
            structural[int(e["onset"] // 30)].append(e)
    packets = []
    for i, row in enumerate(rows):
        def selected(key):
            if row.get(key).seconds is None:
                return []
            return [events[e] for e in row.get(key).event_ids if e in events]
        # Preserve the frozen two-epoch channel-validity restriction.
        window_good = all(bank["goods"][r][j] for r in "fc"
                          for j in range(max(0, i - 1), i + 1))
        qualified = []
        for e in structural[i]:
            kind = e["event_type"]
            known = row.get(kind).seconds is not None and window_good
            if kind == "kcomplex":
                known &= row.get("arousal").seconds is not None
            if known and not e["measurements"].get("associated_arousal_ids"):
                qualified.append(e)
        # The frozen N2.1 window includes the previous epoch's last 15 seconds.
        # Recover its physiological anchor only if no later/overlapping arousal
        # has invalidated it. This does not add an ordinary continuation rule.
        prior_ar = ([events[e] for e in rows[i-1].get("arousal").event_ids
                     if e in events] if i else [])
        carry = []
        for key in ("n2_spindle", "n2_kcomplex"):
            if row.get(key).seconds is None:
                continue
            for eid in row.get(key).event_ids:
                e = events[eid]
                if e["onset"] >= i * 30:
                    continue
                if any(a["offset"] > e["onset"] and a["onset"] < i*30 for a in prior_ar):
                    continue
                carry.append(e)
        packets.append({"arousal": selected("arousal"), "lamf": selected("lamf"),
                        "structure": qualified, "carry_structure": carry})
    return packets


def sweep(row: EpochEvidence, packet: dict, incoming: EndContext) -> dict:
    """Half-open sweep. Arousal seconds remain UNKNOWN in the 30s denominator."""
    start, end = row.index * 30., (row.index + 1) * 30.
    state = deepcopy(incoming)
    engaged = incoming.repair_active
    transitions, segments, consumed = [], [], set()
    if packet.get("carry_structure"):
        e = max(packet["carry_structure"], key=lambda v: v["onset"])
        if state.pending_arousal_end is None or e["onset"] >= state.pending_arousal_end:
            state.phase, state.stage = "N2", "N2"
            state.physiological_anchor_valid = True
            state.physiological_anchor_epoch = int(e["onset"] // 30)
            state.physiological_anchor_event_ids = [e["event_id"]]
            state.pending_arousal_end = None
            transitions.append({"time": start, "kind": "PRIOR_HALF_N2_ANCHOR",
                                "event_id": e["event_id"], "event_onset": e["onset"]})
    boundaries = {start, end}
    for kind in ("arousal", "lamf"):
        for e in packet[kind]:
            boundaries.update((max(start, e["onset"]), min(end, e["offset"])))
    for e in packet["structure"]:
        boundaries.add(e["onset"])
    boundaries = sorted(t for t in boundaries if start <= t <= end)
    for a, b in zip(boundaries, boundaries[1:]):
        active_ar = [e for e in packet["arousal"] if e["onset"] <= a < e["offset"]]
        for e in active_ar:
            if e["event_id"] in consumed:
                continue
            consumed.add(e["event_id"])
            if state.phase in ("N2", "POST_AROUSAL"):
                engaged = True
                previous_phase = state.phase
                old_anchor = state.physiological_anchor_event_ids[:]
                state.phase = "POST_AROUSAL"
                state.pending_arousal_end = max(state.pending_arousal_end or 0., e["offset"])
                state.physiological_anchor_valid = False
                state.physiological_anchor_epoch = None
                state.physiological_anchor_event_ids = []
                if previous_phase == "N2" or not state.origin_arousal_ids:
                    state.origin_arousal_ids = [e["event_id"]]
                state.last_arousal_ids = [e["event_id"]]
                transitions.append({"time": a, "kind": "N2_ANCHOR_INVALIDATED",
                    "event_id": e["event_id"], "old_anchor_ids": old_anchor,
                    "previous_phase": previous_phase,
                    "arousal_offset": e["offset"]})
        # A spindle during an active arousal is not a post-arousal reentry event.
        for e in packet["structure"]:
            if e["onset"] != a:
                continue
            if active_ar or (state.pending_arousal_end is not None and a < state.pending_arousal_end):
                transitions.append({"time": a, "kind": "STRUCTURE_BEFORE_AROUSAL_END",
                                    "event_id": e["event_id"]})
                continue
            was_post = state.phase == "POST_AROUSAL"
            state.phase = "N2"
            state.physiological_anchor_valid = True
            state.physiological_anchor_epoch = row.index
            state.physiological_anchor_event_ids = [e["event_id"]]
            state.pending_arousal_end = None
            transitions.append({"time": a, "kind": "N2_REENTRY" if was_post else "N2_STRUCTURE",
                                "event_id": e["event_id"], "event_type": e["event_type"]})
        lamf_ids = [e["event_id"] for e in packet["lamf"] if e["onset"] <= a < e["offset"]]
        ongoing = state.pending_arousal_end is not None and a < state.pending_arousal_end
        if not row.signals_good:
            value, reason = "UNDETERMINED", "invalid_signal"
        elif state.phase == "POST_AROUSAL":
            if ongoing or active_ar:
                value, reason = "UNDETERMINED", "arousal_interval"
            elif lamf_ids:
                value, reason = "N1", "post_arousal_lamf"
            else:
                value, reason = "UNDETERMINED", "post_arousal_lamf_not_observed"
        elif state.phase == "N2" and state.physiological_anchor_valid:
            value, reason = "N2", "valid_n2_context"
        elif state.stage == "N1" and lamf_ids:
            value, reason = "N1", "prior_n1_lamf"
        else:
            value, reason = "UNDETERMINED", "outside_n2_arousal_context"
        state.stage = value
        segments.append({"onset": a, "offset": b, "stage": value, "reason": reason,
                         "lamf_event_ids": lamf_ids,
                         "physiological_anchor_ids": state.physiological_anchor_event_ids[:],
                         "arousal_ids": state.last_arousal_ids[:]})
    seconds = {s: sum(x["offset"] - x["onset"] for x in segments if x["stage"] == s)
               for s in ("N1", "N2", "UNDETERMINED")}
    assert abs(sum(seconds.values()) - 30.) < 1e-6
    return {"engaged": engaged, "before": asdict(incoming), "candidate_end": state,
            "segments": segments, "seconds": seconds, "transitions": transitions,
            "convention": "fixed30; arousal UNKNOWN; temporal majority strictly >15"}


def finalize_context(decision, temporal: dict, row: EpochEvidence, packet: dict,
                     previous: EndContext) -> EndContext:
    """Fallback labels never rewrite a live, interrupted physiological context."""
    if not row.signals_good:
        return EndContext()
    override = decision.stage in ("W", "N3", "R") and decision.decision_source != "INHERITANCE"
    # R.3.N2 fixes this epoch's label but cannot undo a later N2 arousal.
    override |= (decision.rule_id.startswith("R.3") and decision.stage != "N2")
    override |= "MBM" in decision.rule_id
    if temporal["engaged"] and not override:
        end = deepcopy(temporal["candidate_end"])
        end.repair_active = end.phase == "POST_AROUSAL" or end.stage != decision.stage
        if not end.repair_active:
            end.origin_arousal_ids = []
            end.last_arousal_ids = []
        return end
    # Outside the repair path retain frozen epoch-driven behavior.
    end = EndContext(stage=decision.stage)
    if decision.stage == "N2":
        end.phase = "N2"
        end.physiological_anchor_valid = True
        ids = [e["event_id"] for e in packet["structure"]]
        if ids:
            end.physiological_anchor_event_ids = ids
            end.physiological_anchor_epoch = row.index
        elif previous.phase == "N2":
            end.physiological_anchor_event_ids = previous.physiological_anchor_event_ids[:]
            end.physiological_anchor_epoch = previous.physiological_anchor_epoch
        else:
            end.physiological_anchor_epoch = decision.anchor
    return end
