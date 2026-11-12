"""Signal algorithms only. No imports from annotations/evaluation or label inputs."""
import gc
import json
import numpy as np
from scipy.signal import find_peaks, resample_poly
from .edf_loader import load_channel
from .preprocessing import (bandpass, quality_epochs, sample_mask, segments,
                            spectral_windows, psd_ratio, emg_envelope, filter_valid_blocks)
from .temporal import intersection

EVENT_COLUMNS = ["record_id", "subject_id", "event_id", "event_type", "onset", "offset",
                 "duration", "channel", "features", "qualified", "rejection_reason",
                 "quality_status", "detector_version"]


def detect_record(path, info, cfg):
    import yasa
    sf = cfg["sampling_hz"]
    n_epochs = int(info["duration_sec"] // 30)
    all_events, quality, availability, backgrounds = [], [], {}, {}
    goods, retained = {}, {}

    def add(kind, onset, offset, channel, features=None, good=True, reason="", version="heuristic_v1"):
        all_events.append({"record_id": info["subject_id"], "subject_id": info["subject_id"],
                           "event_type": kind, "onset": float(onset), "offset": float(offset),
                           "duration": float(offset - onset), "channel": channel,
                           "features": json.dumps(features or {}, sort_keys=True), "qualified": bool(good),
                           "rejection_reason": reason, "quality_status": "AVAILABLE" if good else "INVALID_SIGNAL",
                           "detector_version": version})

    def valid_interval(good, a, b):
        first, last = max(0, int(a // 30)), min(n_epochs, int(np.ceil(b / 30)))
        return first < last and bool(np.all(good[first:last]))

    for role in ["frontal", "central", "occipital", "eog_left", "eog_right", "emg"]:
        name = info["selected_channels"][role]
        kind = "emg" if role == "emg" else ("eog" if role.startswith("eog") else "eeg")
        if name is None:
            goods[role] = np.zeros(n_epochs, dtype=bool)
            availability[role] = "UNAVAILABLE"
            continue
        x, native_sf = load_channel(path, name, None)
        # Quality is checked on native samples before anti-aliasing/digital filtering.
        bounds = next((c for c in info["channels"] if c["channel"] == name), {})
        good, rows = quality_epochs(x, native_sf, kind, cfg["quality"],
                                    (bounds["physical_min_uv"], bounds["physical_max_uv"]))
        goods[role] = good
        for row in rows:
            row.update({"channel": name, "role": role, "record_id": info["subject_id"],
                        "native_sampling_hz": native_sf})
        quality.extend(rows)
        availability[role] = "AVAILABLE" if good.any() else "INVALID_SIGNAL"
        if native_sf != sf:
            from fractions import Fraction
            ratio = Fraction(sf / native_sf).limit_denominator(1000)
            x = resample_poly(x, ratio.numerator, ratio.denominator)
        mask = sample_mask(good, len(x), sf)
        clean = np.where(np.isfinite(x), x, 0.0)
        x = filter_valid_blocks(clean, sf, mask, cfg["filter"][kind], cfg["filter"]["order"])
        del clean
        if role in ["frontal", "central"]:
            for detector, event_type in [(yasa.sw_detect, "slow_wave"),
                                          (yasa.spindles_detect, "spindle")]:
                params = dict(cfg["detectors"]["sw" if event_type == "slow_wave" else "spindle"])
                status = "AVAILABLE" if good.any() else "INVALID_SIGNAL"
                chunk, context = cfg["detectors"]["chunk_sec"], cfg["detectors"]["context_sec"]
                failures = []
                blocks = [(a, b, core) for a, b in segments(mask, sf)
                          for core in np.arange(a, b, chunk)]
                for block_start, block_end, core in blocks:
                    lo, hi = max(block_start, core - context), min(block_end, core + chunk + context)
                    a, b = int(lo * sf), int(hi * sf)
                    if b - a < sf * 10 or not mask[a:b].any():
                        continue
                    try:
                        result = detector(x[a:b], sf=sf, ch_names=[name], hypno=None,
                                          remove_outliers=False, verbose=False, **params)
                        if result is not None:
                            for row in result.summary().to_dict("records"):
                                start, end = float(row["Start"]) + lo, float(row["End"]) + lo
                                if not (core <= start < min(core + chunk, info["duration_sec"])):
                                    continue
                                valid = valid_interval(good, start, end)
                                add(event_type, start, end, name, {
                                    k: (v.item() if isinstance(v, np.generic) else v)
                                    for k, v in row.items() if k != "Channel"}, valid,
                                    "quality_overlap" if not valid else "", f"yasa_{yasa.__version__}")
                        del result
                    except Exception as exc:
                        failures.append({"interval": [lo, hi], "error": repr(exc)})
                # A detector exception invalidates this channel detector, not a zero count.
                availability[f"{event_type}_{role}"] = "INVALID_SIGNAL" if failures else status
                backgrounds[f"{event_type}_{role}_failures"] = failures
                gc.collect()
        if role == "central":
            lamf = cfg["detectors"]["lamf"]
            threshold = float(np.mean(np.abs(x[mask])) - lamf["std_multiplier"] * np.std(x[mask])) if mask.any() else None
            backgrounds["lamf_amplitude_threshold_uv"] = threshold
            if threshold is not None and threshold > 0:
                for a, b in segments((np.abs(x) < threshold) & mask, sf, lamf["minimum_sec"]):
                    ratio = psd_ratio(x[int(a * sf):int(b * sf)], sf, lamf["band_hz"], lamf["psd_total_hz"])
                    qualified = ratio >= lamf["relative_threshold"]
                    add("lamf", a, b, name, {"theta_relative_power": ratio, "amplitude_threshold_uv": threshold},
                        qualified, "theta_power_below_threshold" if not qualified else "", "paper_literal_lamf_v1")
                availability["lamf"] = availability[role]
            else:
                availability["lamf"] = "INVALID_SIGNAL"
            # Explicit frequency-shift candidate, not validated clinical arousal.
            ac = cfg["detectors"]["arousal"]
            spectral = spectral_windows(x, sf, mask, ac["window_sec"], ac["step_sec"], ac["band_hz"], [0.3, 35])
            candidates = []
            for j in range(ac["baseline_sec"], len(spectral)):
                previous = spectral[j - ac["baseline_sec"]:j]
                if spectral[j][0] - previous[0][0] > ac["baseline_sec"] + 0.01:
                    continue
                p = np.array([z[2] for z in previous])
                if np.mean(p) > 0 and np.std(p) / np.mean(p) <= ac["baseline_cv_max"]:
                    if spectral[j][2] > ac["shift_ratio"] * np.median(p):
                        candidates.append(spectral[j][:2])
            from .temporal import union
            for a, b in union(candidates):
                if b - a >= ac["minimum_sec"]:
                    add("arousal_candidate", a, b, name, {"clinical_event_validation": False})
            availability["arousal_candidate"] = "PARTIALLY_SUPPORTED" if good.any() else "INVALID_SIGNAL"
            kc = cfg["detectors"]["kcomplex"]
            slow = filter_valid_blocks(x, sf, mask, kc["band_hz"])
            peaks, _ = find_peaks(-slow, height=kc["negative_peak_uv"], distance=int(kc["min_distance_sec"] * sf))
            for peak in peaks:
                a = peak + int(kc["peak_interval_sec"][0] * sf)
                b = min(len(slow), peak + int(kc["peak_interval_sec"][1] * sf) + 1)
                if a >= b:
                    continue
                positive = a + int(np.argmax(slow[a:b]))
                start, end = max(0, (peak - int(0.25 * sf)) / sf), min(len(x) / sf, (positive + int(.25 * sf)) / sf)
                ptp = float(slow[positive] - slow[peak])
                valid = (slow[positive] >= kc["positive_peak_uv"] and
                         kc["peak_to_peak_uv"] <= ptp <= kc["max_ptp_uv"] and
                         valid_interval(good, start, end))
                add("kcomplex", start, end, name, {"negative_peak_uv": float(slow[peak]),
                    "positive_peak_uv": float(slow[positive]), "peak_to_peak_uv": ptp,
                    "negative_peak_sec": float(peak / sf), "positive_peak_sec": float(positive / sf),
                    "arousal_association": "computed_from_candidates_after_detection"}, valid,
                    "morphology_or_quality" if not valid else "")
            del slow
            availability["kcomplex"] = "PARTIALLY_SUPPORTED" if good.any() else "INVALID_SIGNAL"
        elif role == "occipital":
            ac = cfg["detectors"]["alpha"]
            for a, b, ratio in spectral_windows(x, sf, mask, ac["window_sec"], ac["step_sec"], ac["band_hz"], ac["psd_total_hz"]):
                if ratio >= ac["relative_threshold"]:
                    add("alpha", a, b, name, {"alpha_relative_power": ratio}, version="welch_alpha_v1")
            availability["alpha"] = availability[role]
        elif role.startswith("eog"):
            retained[role] = x
        elif role == "emg":
            events, rms, baseline = emg_envelope(x, sf, mask, cfg["detectors"]["emg"])
            backgrounds["emg"] = baseline
            availability["emg_tone"] = baseline["status"]
            for a, b, value, event_type in events:
                add(event_type, a, b, name, {"rms_uv": value}, version="quantile_emg_v1")
        del x
        gc.collect()
    if len(retained) == 2 and all(goods[k].any() for k in retained):
        chunk, context = cfg["detectors"]["chunk_sec"], cfg["detectors"]["context_sec"]
        failures = []
        combined = sample_mask(goods["eog_left"] & goods["eog_right"], len(retained["eog_left"]), sf)
        blocks = [(a, b, core) for a, b in segments(combined, sf) for core in np.arange(a, b, chunk)]
        for block_start, block_end, core in blocks:
            lo, hi = max(block_start, core - context), min(block_end, core + chunk + context)
            a, b = int(lo * sf), int(hi * sf)
            try:
                result = yasa.rem_detect(retained["eog_left"][a:b], retained["eog_right"][a:b], sf,
                                         hypno=None, verbose=False, **cfg["detectors"]["rem"])
                if result is not None:
                    for row in result.summary().to_dict("records"):
                        start, end = float(row["Start"]) + lo, float(row["End"]) + lo
                        if core <= start < min(core + chunk, info["duration_sec"]):
                            good = valid_interval(goods["eog_left"] & goods["eog_right"], start, end)
                            add("rem_candidate", start, end, "E1/M2+E2/M2", row, good,
                                "quality_overlap" if not good else "", f"yasa_{yasa.__version__}")
                del result
            except Exception as exc:
                failures.append({"interval": [lo, hi], "error": repr(exc)})
        availability["rem_candidate"] = "INVALID_SIGNAL" if failures else "AVAILABLE"
        backgrounds["rem_failures"] = failures
    else:
        availability["rem_candidate"] = "UNAVAILABLE"
    for kind in ["alpha", "lamf", "emg_tone", "kcomplex", "arousal_candidate"]:
        availability.setdefault(kind, "UNAVAILABLE")
    for kind in ["eye_blink", "slow_eye_movement", "vertex_wave", "background_slowing"]:
        availability[kind] = "UNAVAILABLE"
    for event_type in ["slow_wave", "spindle"]:
        statuses = [availability.get(f"{event_type}_{r}", "UNAVAILABLE") for r in ["frontal", "central"]]
        availability[event_type] = "AVAILABLE" if all(x == "AVAILABLE" for x in statuses) else "UNAVAILABLE"
    all_events.sort(key=lambda e: (e["onset"], e["offset"], e["event_type"], e["channel"]))
    for i, event in enumerate(all_events):
        event["event_id"] = f"{info['subject_id']}_{i:07d}"
    arousals = [e for e in all_events if e["event_type"] == "arousal_candidate" and e["qualified"]]
    for event in all_events:
        if event["event_type"] == "kcomplex":
            features = json.loads(event["features"])
            features["arousal_candidate_ids"] = [a["event_id"] for a in arousals
                if intersection((event["onset"], event["offset"] + cfg["detectors"]["kcomplex"]["arousal_after_sec"]),
                                (a["onset"], a["offset"])) > 0]
            event["features"] = json.dumps(features, sort_keys=True)
    return all_events, quality, availability, backgrounds, goods
