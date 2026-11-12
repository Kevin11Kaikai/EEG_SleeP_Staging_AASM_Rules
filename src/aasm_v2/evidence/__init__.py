"""Event-to-epoch aggregation without any reference annotation access."""
from collections import defaultdict
import numpy as np
from ..temporal import duration, merge, intersect
from ..types import Evidence, EpochEvidence

ROLES = {"slow_wave": "fc", "spindle": "fc", "alpha": "o", "lamf": "c",
         "rem": "lr", "blink": "lr", "sem": "lr", "vertex": "fc",
         "low_emg": "m", "high_emg": "m", "normal_emg": "m", "arousal": "c",
         "slowing": "co", "mbm": ""}
PARTIAL_ABSENCE = {"arousal", "kcomplex", "blink", "sem", "vertex", "mbm", "slowing"}


def aggregate(bank: dict, mode: str) -> list[EpochEvidence]:
    groups = defaultdict(list)
    for event in bank["events"][mode]:
        groups[event["event_type"]].append(event)
    intervals = {k: merge([(e["onset"], e["offset"]) for e in ev]) for k, ev in groups.items()}
    status = bank["availability"][mode]
    n = int(bank["info"]["duration"] // 30)
    goods = bank["goods"]
    alpha_total = duration(intervals.get("alpha", []), 0, n * 30)
    alpha_generator = alpha_total >= 30 if status.get("alpha") == "AVAILABLE" else None
    overlap = intersect(intervals.get("rem", []), intervals.get("low_emg", []))
    wake_eye_tone = intersect(intervals.get("rem", []), intervals.get("normal_emg", []))
    # Index event references locally to avoid a quadratic full-night join.
    bins = defaultdict(lambda: defaultdict(list))
    onset_bins = defaultdict(lambda: defaultdict(list))
    for kind, events in groups.items():
        for e in events:
            for j in range(max(0, int(e["onset"] // 30)), min(n, int(np.ceil(e["offset"] / 30)))):
                bins[j][kind].append(e)
            onset_bins[int(e["onset"] // 30)][kind].append(e)
    result = []
    for i in range(n):
        start, end = i * 30, (i + 1) * 30
        values = {}
        keys = set(ROLES) | {"kcomplex"}
        for key in keys:
            roles = ("c" if mode == "original" else "f") if key == "kcomplex" else ROLES[key]
            state = status.get(key, "UNAVAILABLE")
            if not all(goods[r][i] for r in roles):
                state = "INVALID_SIGNAL" if all(bank["info"]["roles"][r] for r in roles) else "UNAVAILABLE"
            events = bins[i][key]
            secs = duration([(e["onset"], e["offset"]) for e in events], start, end)
            values[key] = Evidence(secs if state in ("AVAILABLE", "PARTIALLY_SUPPORTED") else None,
                                   [e["event_id"] for e in events], state,
                                   negative_validated=False)
        eye_emg_known = all(values[k].seconds is not None for k in ("rem", "low_emg"))
        values["rem_low_overlap"] = Evidence(duration(overlap, start, end) if eye_emg_known else None,
                                             values["rem"].event_ids + values["low_emg"].event_ids)
        rem_onsets = sorted(e["onset"] for e in bins[i]["rem"])
        spacing = np.diff(rem_onsets)
        irregular = len(spacing) >= 2 and np.std(spacing) / max(np.mean(spacing), 1e-9) > .2
        wake_known = all(values[k].seconds is not None for k in ("rem", "normal_emg"))
        values["irregular_rem_normal"] = Evidence(
            duration(wake_eye_tone, start, end) if wake_known and irregular else (0.0 if wake_known else None),
            values["rem"].event_ids + values["normal_emg"].event_ids, "PARTIALLY_SUPPORTED")
        window_good = all(goods[r][j] for r in "fc" for j in range(max(0, i - 1), i + 1))
        for kind in ("spindle", "kcomplex"):
            candidates = onset_bins[i][kind] + onset_bins[i - 1][kind]
            keep = [e for e in candidates if max(0, start - 15) <= e["onset"] < start + 15
                    and (kind != "kcomplex" or not e["measurements"].get("associated_arousal_ids"))]
            known = window_good and values[kind].seconds is not None
            if kind == "kcomplex":
                known &= values["arousal"].seconds is not None
            # Qualified K means no associated candidate, NOT a validated absent arousal.
            values["n2_" + kind] = Evidence(float(len(keep)) if known else None,
                                             [e["event_id"] for e in keep],
                                             "PARTIALLY_SUPPORTED" if kind == "kcomplex" else "AVAILABLE",
                                             negative_validated=False)
        first = [e for e in onset_bins[i]["spindle"] + onset_bins[i]["kcomplex"]
                 if start <= e["onset"] < start + 15
                 and (e["event_type"] != "kcomplex" or not e["measurements"].get("associated_arousal_ids"))]
        values["first_half_n2"] = Evidence(float(len(first)) if window_good else None,
                                            [e["event_id"] for e in first])
        ar = bins[i]["arousal"]
        recent_ar = bins[i-1]["arousal"] + ar
        sem = bins[i]["sem"]
        followed = [(a, s) for a in recent_ar for s in sem
                    if a["offset"] <= s["onset"] <= a["offset"] + 30]
        values["arousal_then_sem"] = Evidence(float(len(followed)) if all(
            values[k].seconds is not None for k in ("arousal", "sem")) else None,
            [e["event_id"] for pair in followed for e in pair], "PARTIALLY_SUPPORTED")
        last_arousal_end = max((e["offset"] for e in ar), default=None)
        window_events = [e for kind in ("spindle", "kcomplex")
                         for e in onset_bins[i][kind] + onset_bins[i - 1][kind]
                         if max(0,start-15) <= e["onset"] < start+15
                         and (kind != "kcomplex" or not e["measurements"].get("associated_arousal_ids"))]
        n2_after = [e for e in window_events if last_arousal_end is None or e["onset"] >= last_arousal_end]
        result.append(EpochEvidence(i, bool(goods["f"][i] and goods["c"][i]), values,
            {"alpha_generator": alpha_generator, "alpha_total_seconds": alpha_total,
             "n2_event_in_epoch": bool(bins[i]["spindle"] or any(
                 not e["measurements"].get("associated_arousal_ids") for e in bins[i]["kcomplex"])),
             "n2_after_arousal_ids": [e["event_id"] for e in n2_after],
             "n2_window_onsets": [[e["event_id"],e["onset"]] for e in window_events],
             "last_arousal_end": last_arousal_end,
             "k_nonassociated_status": "UNKNOWN_CLINICAL_ABSENCE"}))
    return result
