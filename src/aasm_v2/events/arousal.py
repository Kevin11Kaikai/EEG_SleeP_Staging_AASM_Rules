"""Frozen pre-onset reference, multi-band frequency-shift candidate detector.

Sleep context is an explicitly imperfect signal-derived surrogate, never labels.
No candidate is treated as a clinically certified event or certified absence.
"""
import numpy as np
from ..temporal import runs


def legacy_arousal(ratio, valid) -> tuple:
    marked = np.zeros(len(ratio), bool)
    counts = {"windows": len(ratio), "history_valid": 0, "stable_cv": 0, "ratio_pass": 0}
    for i in range(10, len(ratio)):
        if not np.all(valid[i - 10:i + 1]):
            continue
        counts["history_valid"] += 1
        prior = ratio[i - 10:i]
        if np.mean(prior) <= 0 or np.std(prior) / np.mean(prior) > .5:
            continue
        counts["stable_cv"] += 1
        if ratio[i] > 3 * np.median(prior):
            counts["ratio_pass"] += 1
            marked[i] = True
    intervals = runs(marked, minimum=3)
    counts.update({"one_second_candidates": int(marked.sum()),
                   "short_runs_rejected": len(runs(marked)) - len(intervals),
                   "accepted": len(intervals)})
    return [(a, b, {"legacy_frequency_shift": True}) for a, b in intervals], counts


def enhanced_arousal(spec: dict, sleep_like, spindle, artifact, rem_context,
                     emg_rms, emg_valid, cfg: dict) -> tuple:
    n = len(spec["valid"])
    matrix = np.column_stack([spec[k] for k in ("theta", "alpha", "beta")])
    counts = {k: 0 for k in ("tested_onsets", "valid_history", "stable_sleep_surrogate",
                             "abrupt_shift", "duration_3s", "artifact_spindle_clear",
                             "rem_context", "rem_emg_pass", "unknown_context_rejected", "accepted")}
    accepted = []
    i, history, minimum = cfg["baseline_s"], cfg["baseline_s"], cfg["minimum_s"]
    while i + minimum <= n:
        counts["tested_onsets"] += 1
        if not spec["valid"][i - history:i + minimum].all():
            i += 1
            continue
        counts["valid_history"] += 1
        background = matrix[i - history:i]
        ref = np.median(background, axis=0)
        stable = np.max(np.sum(np.abs(background - ref), axis=1)) <= cfg["stability_l1_max"]
        if not stable or not np.all(sleep_like[i - history:i]):
            i += 1
            continue
        counts["stable_sleep_surrogate"] += 1
        frequency = float(np.median(spec["frequency"][i - history:i]))
        base_power = np.median(background * spec["total"][i - history:i, None], axis=0)

        def changed(j):
            increased = ((matrix[j] - ref >= cfg["relative_increase"]) &
                         (matrix[j] * spec["total"][j] >= cfg["power_ratio"] * base_power))
            return bool(np.any(increased) or
                        spec["frequency"][j] >= frequency + cfg["frequency_increase_hz"])

        if not changed(i):
            i += 1
            continue
        counts["abrupt_shift"] += 1
        # Freeze this reference for the whole event. Candidate seconds never enter it.
        end = i
        while end < n and spec["valid"][end] and changed(end):
            end += 1
        if end - i < minimum:
            i += 1
            continue
        counts["duration_3s"] += 1
        if np.any(artifact[i:end]) or np.mean(spindle[i:end]) > .25:
            i = end
            continue
        counts["artifact_spindle_clear"] += 1
        contexts = rem_context[max(0, i - history):i]
        context = bool(np.any(contexts == 1))
        if not context and np.any(contexts == -1):
            counts["unknown_context_rejected"] += 1
            i = end
            continue
        mseconds = 0
        if context:
            counts["rem_context"] += 1
            if np.all(emg_valid[i - history:end]):
                tone = float(np.median(emg_rms[i - history:i]))
                above = emg_rms[i:end] >= max(cfg["emg_ratio"] * tone,
                                             tone + cfg["emg_absolute_increase_uv"])
                mseconds = sum(b - a for a, b in runs(above, minimum=cfg["emg_increase_s"]))
            if mseconds < cfg["emg_increase_s"]:
                i = end
                continue
            counts["rem_emg_pass"] += 1
        accepted.append((float(i), float(end), {"reference_start": i - history,
                         "reference_end": i, "reference_frozen": True,
                         "sleep_context": "signal_surrogate_not_expert_confirmed",
                         "rem_context": context, "emg_increase_seconds": mseconds}))
        counts["accepted"] += 1
        # Another candidate needs ten fresh seconds of signal-derived stable sleep.
        i = end + history
    return accepted, counts
