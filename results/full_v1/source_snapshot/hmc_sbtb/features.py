"""Aggregate label-free physiological events using real interval unions."""
import json
import numpy as np
from .temporal import coverage, onset_in, pair_overlap, intersection


def aggregate(events, goods, availability, info, cfg):
    n = int(info["duration_sec"] // 30)
    by_type = {}
    for event in events:
        if event["qualified"]:
            by_type.setdefault(event["event_type"], []).append(event)
    intervals = {k: [(e["onset"], e["offset"]) for e in v] for k, v in by_type.items()}
    alpha_total = coverage(intervals.get("alpha", []), 0, n * 30)
    alpha_generator = (alpha_total >= cfg["detectors"]["alpha"]["generator_min_total_sec"]
                       if availability["alpha"] == "AVAILABLE" else None)
    out = []
    for i in range(n):
        a, b = i * 30, (i + 1) * 30
        row = {"record_id": info["subject_id"], "subject_id": info["subject_id"],
               "epoch_index": i, "start_sec": a, "end_sec": b,
               "generates_alpha_rhythm": alpha_generator}
        eeg_ok = bool(goods.get("frontal", np.zeros(n, bool))[i] and
                      goods.get("central", np.zeros(n, bool))[i])
        row["quality_barrier"] = not eeg_ok
        for kind in ["slow_wave", "spindle", "alpha", "lamf", "low_emg", "high_emg",
                     "rem_candidate", "kcomplex", "arousal_candidate"]:
            detector_key = "emg_tone" if kind.endswith("emg") else kind
            roles = {"slow_wave": ["frontal", "central"], "spindle": ["frontal", "central"],
                     "alpha": ["occipital"], "lamf": ["central"], "low_emg": ["emg"],
                     "high_emg": ["emg"], "rem_candidate": ["eog_left", "eog_right"],
                     "kcomplex": ["central"], "arousal_candidate": ["central"]}[kind]
            available = availability.get(detector_key) in ["AVAILABLE", "PARTIALLY_SUPPORTED"]
            available &= all(bool(goods.get(r, np.zeros(n, bool))[i]) for r in roles)
            row[kind + "_available"] = bool(available)
            row[kind + "_sec"] = coverage(intervals.get(kind, []), a, b) if available else None
            row[kind + "_ids"] = [e["event_id"] for e in by_type.get(kind, [])
                                  if intersection((e["onset"], e["offset"]), (a, b)) > 0]
        start = max(0, a - cfg["rules"]["n2_onset_half_window_sec"])
        end = a + cfg["rules"]["n2_onset_half_window_sec"]
        previous_good = i == 0 or all(goods[r][i - 1] for r in ["frontal", "central"])
        row["n2_window_quality"] = bool(eeg_ok and previous_good)
        row["n2_window_available"] = bool(row["n2_window_quality"] and row["spindle_available"])
        row["n2_spindle_ids"] = [e["event_id"] for e in by_type.get("spindle", [])
                                  if onset_in(e["onset"], start, end)]
        row["n2_kcomplex_ids"] = [e["event_id"] for e in by_type.get("kcomplex", [])
            if onset_in(e["onset"], start, end) and not json.loads(e["features"])["arousal_candidate_ids"]]
        row["n2_kcomplex_available"] = (row["n2_window_quality"] and row["kcomplex_available"]
                                          and row["arousal_candidate_available"])
        row["rem_low_overlap_sec"] = pair_overlap(intervals.get("rem_candidate", []),
                                                    intervals.get("low_emg", []), a, b)
        row["quality_status"] = "INVALID_SIGNAL" if row["quality_barrier"] else "AVAILABLE"
        out.append(row)
    return out
