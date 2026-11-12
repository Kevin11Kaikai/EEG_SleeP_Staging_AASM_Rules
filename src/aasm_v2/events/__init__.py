"""Compute original-specification and enhanced candidate banks from the same PSG."""
from dataclasses import asdict
import logging
import numpy as np
from scipy.signal import welch
from scipy.stats import trim_mean
from ..data import header, read_channel
from ..preprocessing import quality, prepare, spectra, rms_windows
from ..temporal import runs, merge, raster, duration
from ..types import Event
from .morphology import biphasic_candidates, ocular_candidates, vertex_candidates
from .arousal import legacy_arousal, enhanced_arousal

PARTIAL = {"kcomplex", "arousal", "blink", "sem", "vertex", "mbm", "slowing"}


def measured_windows(values: dict, selected) -> list:
    return merge(zip(values["start"][selected & values["valid"]],
                     values["end"][selected & values["valid"]]))


def yasa_candidates(signals: list, sf: float, good, detector: str, cfg: dict) -> tuple:
    import yasa
    from yasa.others import trimbothstd
    logging.getLogger("yasa").setLevel(logging.ERROR)
    result, log = [], []
    func = {"slow_wave": yasa.sw_detect, "spindle": yasa.spindles_detect,
            "rem": yasa.rem_detect}[detector]
    for left, right in runs(good, 1 / 30):
        for core in np.arange(left, right, cfg["chunk_seconds"]):
            a = max(left, core - cfg["context_seconds"])
            b = min(right, core + cfg["chunk_seconds"] + cfg["context_seconds"])
            chunks = [x[round(a * sf):round(b * sf)] for x in signals]
            # Check YASA's trimmed-amplitude eligibility BEFORE interpreting None.
            spreads = [float(trimbothstd(x, cut=.05)) for x in chunks]
            entry = {"detector": detector, "start": a, "end": b, "core": float(core),
                     "robust_spread_uv": spreads}
            if b - a < 10 or any(not .1 < v < 1000 for v in spreads):
                entry["status"] = "INVALID_SIGNAL"
                log.append(entry)
                continue
            try:
                params = dict(cfg[detector])
                common = dict(sf=sf, hypno=None, verbose="ERROR", remove_outliers=False)
                found = (func(*chunks, **common, **params) if detector == "rem" else
                         func(chunks[0], ch_names=["physical_channel"], **common, **params))
                entry["status"] = "NO_CANDIDATE" if found is None else "EVENTS_RETURNED"
                entry["count"] = 0
                if found is not None:
                    for row in found.summary().to_dict("records"):
                        onset, offset = float(row["Start"]) + a, float(row["End"]) + a
                        if core <= onset < min(right, core + cfg["chunk_seconds"]):
                            measurements = {k: float(v) for k, v in row.items()
                                            if isinstance(v, (int, float, np.number)) and np.isfinite(v)}
                            result.append((onset, offset, measurements))
                            entry["count"] += 1
            except Exception as exc:
                entry.update(status="DETECTOR_FAILURE", error=repr(exc))
            log.append(entry)
    return result, log


def detect(path, cfg: dict) -> dict:
    info = header(path)
    sf = cfg["sampling_hz"]
    n = int(info["duration"] // 30)
    signals, goods, raw_features = {}, {}, {}
    for role in ("f", "c", "o", "l", "r", "m"):
        x, native = read_channel(path, role, info)
        if x is None:
            goods[role] = np.zeros(n, bool)
            signals[role] = np.zeros(n * 30 * sf)
            continue
        detail = next(c for c in info["channels"] if c["name"] == info["roles"][role])
        kind = "emg" if role == "m" else ("eog" if role in "lr" else "eeg")
        good = quality(x, native, kind, (detail["min_uv"], detail["max_uv"]), cfg["qc"])
        goods[role] = good
        # Independent raw movement features survive ordinary channel QC rejection.
        if role in "fcm":
            raw_features[role] = spectra(x[:n * 30 * round(native)], native, np.ones(n, bool))
        signals[role] = prepare(x, native, sf, good, cfg["filters"][kind], cfg["filters"]["order"])
        del x
    banks = {mode: [] for mode in ("original", "enhanced")}
    stats = {"yasa": [], "backgrounds": {}, "arousal": {}, "units": "uV",
             "event_validation": "no expert event gold standard"}
    available = {mode: {} for mode in banks}

    def add(mode, kind, rows, roles, params, version):
        status = "PARTIALLY_SUPPORTED" if kind in PARTIAL else "AVAILABLE"
        if not all(goods[r].any() for r in roles):
            status = "UNAVAILABLE" if any(info["roles"][r] is None for r in roles) else "INVALID_SIGNAL"
        available[mode][kind] = status
        channel = "+".join(info["roles"][r] or f"missing:{r}" for r in roles)
        for row in rows:
            a, b = row[:2]
            measurements = row[2] if len(row) > 2 else {}
            banks[mode].append(Event(event_id="", event_type=kind, onset=float(a),
                                     offset=float(b), channel=channel, detector_version=version,
                                     parameters=params, quality_status=status,
                                     evidence_status="TRUE", measurements=measurements))

    # Same YASA definition and calls shared by both banks, not duplicate computation.
    for kind, roles_list in (("slow_wave", [["f"], ["c"]]),
                              ("spindle", [["f"], ["c"]]), ("rem", [["l", "r"]])):
        failed = False
        for roles in roles_list:
            joint = np.logical_and.reduce([goods[r] for r in roles])
            rows, logs = yasa_candidates([signals[r] for r in roles], sf, joint, kind, cfg["yasa"])
            for log in logs:
                log["roles"] = roles
            stats["yasa"].extend(logs)
            failed |= any(x["status"] in ("INVALID_SIGNAL", "DETECTOR_FAILURE") for x in logs)
            for mode in banks:
                enhanced_rows = rows
                if mode == "enhanced" and kind == "rem":
                    enhanced_rows = []
                    for a, b, detail in rows:
                        ix, jx = round(a * sf), round(b * sf)
                        corr = np.corrcoef(signals["l"][ix:jx], signals["r"][ix:jx])[0, 1]
                        if np.isfinite(corr) and corr < -.3:
                            enhanced_rows.append((a, b, dict(detail, dual_eog_correlation=float(corr))))
                add(mode, kind, enhanced_rows, roles, cfg["yasa"][kind], "yasa-0.6.5")
        if failed:
            for mode in banks:
                available[mode][kind] = "INVALID_SIGNAL"
    c = spectra(signals["c"], sf, goods["c"])
    o = spectra(signals["o"], sf, goods["o"], 2, .5)
    alpha = measured_windows(o, o["alpha"] >= .5)
    for mode in banks:
        add(mode, "alpha", alpha, ["o"], {"window_s": 2, "step_s": .5, "relative_power": .5}, "welch-8-12Hz")
    mask = np.repeat(goods["c"], 30 * sf)
    x = signals["c"]
    threshold = float(np.mean(np.abs(x[mask])) - .01 * np.std(x[mask])) if mask.any() else 0
    lamf = []
    for a, b in runs((np.abs(x) < threshold) & mask, sf, 1):
        f, p = welch(x[round(a * sf):round(b * sf)], fs=sf, nperseg=min(round(2 * sf), round((b - a) * sf)))
        denom = p[(f >= .3) & (f <= 35)].sum()
        ratio = p[(f >= 4) & (f <= 7)].sum() / max(denom, 1e-30)
        if ratio >= .01:
            lamf.append((a, b, {"theta_fraction": float(ratio)}))
    add("original", "lamf", lamf, ["c"], {"sample_threshold_uv": threshold}, "literal-amplitude-v0")
    lw = spectra(x, sf, goods["c"], 1, .5)
    lc = cfg["enhanced"]["lamf"]
    add("enhanced", "lamf", measured_windows(lw, (lw["rms"] <= lc["rms_max_uv"]) &
        (lw["theta"] >= lc["theta_fraction_min"])), ["c"], lc, "local-rms-lamf-v2")
    m = rms_windows(signals["m"], sf, goods["m"])
    valid_rms = m["rms"][m["valid"]]
    if len(valid_rms):
        q20, q80 = map(float, np.quantile(valid_rms, [.2, .8]))
        epoch_tones = []
        for i in np.flatnonzero(goods["m"]):
            samples = m["rms"][(m["start"] >= i * 30) & (m["end"] <= (i + 1) * 30)]
            if np.mean(samples) > 0 and np.std(samples) / np.mean(samples) <= cfg["enhanced"]["emg"]["stable_cv_max"]:
                epoch_tones.append(float(np.median(samples)))
        floor = float(np.quantile(epoch_tones, cfg["enhanced"]["emg"]["floor_quantile"])) if epoch_tones else 0.0
        low = cfg["enhanced"]["emg"]["low_multiplier"] * floor
        high = max(cfg["enhanced"]["emg"]["high_multiplier"] * floor, float(np.median(valid_rms)))
    else:
        q20 = q80 = low = high = floor = 0.0
    stats["backgrounds"].update(lamf_original_uv=threshold, emg_q20_uv=q20, emg_q80_uv=q80,
                                emg_stable_epochs=len(epoch_tones) if len(valid_rms) else 0,
                                emg_floor_uv=floor, enhanced_low_uv=low, enhanced_high_uv=high)
    for mode, lower, upper in (("original", q20, q80), ("enhanced", low, high)):
        for kind, selected in (("low_emg", m["rms"] <= lower),
                               ("high_emg", m["rms"] >= upper),
                               ("normal_emg", m["rms"] > lower)):
            add(mode, kind, measured_windows(m, selected), ["m"],
                {"lower_rms_uv": lower, "upper_rms_uv": upper}, "quantile-v0" if mode == "original" else "stable-floor-v2-experimental")
            if lower <= 1e-8 or upper <= lower:
                available[mode][kind] = "INVALID_SIGNAL"
    ocular = ocular_candidates(signals["l"], signals["r"], sf, goods["l"] & goods["r"], cfg["enhanced"])
    blink_kept=[]
    bc=cfg["enhanced"]["blink"]
    for a,b,detail in ocular["blink"]:
        lo=max(0,round((a-bc["artifact_context_s"])*sf))
        hi=min(len(signals["m"]),round((b+bc["artifact_context_s"])*sf))
        loud_eog=max(np.abs(signals["l"][lo:hi]).max(initial=0),
                     np.abs(signals["r"][lo:hi]).max(initial=0))>bc["maximum_uv"]
        loud_emg=np.sqrt(np.mean(signals["m"][lo:hi]**2))>bc["artifact_emg_rms_uv"]
        if not (loud_eog and loud_emg):
            blink_kept.append((a,b,detail))
    stats["blink_artifact_rejected"]=len(ocular["blink"])-len(blink_kept)
    ocular["blink"]=blink_kept
    for kind in ("blink", "sem"):
        add("enhanced", kind, ocular[kind], ["l", "r"], cfg["enhanced"][kind], "dual-eog-morphology-v2-experimental")
    add("enhanced", "vertex", vertex_candidates(signals["c"], signals["f"], sf, goods["c"] & goods["f"],
        cfg["enhanced"]["vertex"]), ["c", "f"], cfg["enhanced"]["vertex"], "central-sharp-v2-experimental")
    # Background slowing uses algorithmic W-alpha segments as reference, never stage annotations.
    alpha_mask = raster(alpha, len(c["valid"]))
    wake_freqs = c["frequency"][alpha_mask & c["valid"]]
    base_freq = float(np.median(wake_freqs)) if len(wake_freqs) >= 30 else None
    slowing = np.zeros(len(c["valid"]), bool) if base_freq is None else (
        (c["frequency"] <= base_freq - 1) & (c["frequency"] >= 4) & (c["frequency"] <= 7))
    add("enhanced", "slowing", measured_windows(c, slowing), ["c", "o"],
        {"reference_frequency_hz": base_freq}, "wake-surrogate-slowing-v2-experimental")
    if base_freq is None:
        available["enhanced"]["slowing"] = "UNAVAILABLE"
    for mode in banks:
        role = "c" if mode == "original" else "f"
        kc = cfg["enhanced"]["kcomplex"]
        add(mode, "kcomplex", biphasic_candidates(signals[role], sf, goods[role], kc, mode == "original"),
            [role], dict(kc, association_seconds=3 if mode == "original" else 1),
            "legacy-morphology-spec" if mode == "original" else "crossing-morphology-v2-experimental")
    # Clinical movement candidate: concurrent broad EEG disturbance AND muscle activity.
    mbm = np.zeros(len(c["valid"]), bool)
    if all(role in raw_features for role in "fcm"):
        mc = cfg["enhanced"]["mbm"]
        mbm = ((raw_features["f"]["rms"] > mc["eeg_rms_uv"]) &
               (raw_features["c"]["rms"] > mc["eeg_rms_uv"]) &
               (raw_features["c"]["beta"] > mc["eeg_beta_fraction"]) &
               (raw_features["m"]["rms"] > mc["emg_rms_uv"]))
    add("enhanced", "mbm", runs(mbm), ["f", "c", "m"], cfg["enhanced"]["mbm"], "multimodal-movement-v2-experimental")
    old_rows, old_counts = legacy_arousal(c["fast"], c["valid"])
    add("original", "arousal", old_rows, ["c"], {"ratio": 3, "cv": .5, "duration": 3}, "legacy-rolling-spec")
    sec = len(c["valid"])
    def intervals(kind):
        return [(e.onset, e.offset) for e in banks["enhanced"] if e.event_type == kind]
    sp = raster(intervals("spindle"), sec)
    lowmask = raster(intervals("low_emg"), sec)
    lamfmask = raster(intervals("lamf"), sec)
    remmask = raster([(max(0, a - 10), b + 30) for a, b in intervals("rem")], sec)
    sleep_like = c["valid"] & ~alpha_mask & ~mbm & ((c["delta"] + c["theta"] >= .6) | sp)
    # Sleep is a surrogate: its limitations are explicit in every accepted event.
    rm = rms_windows(signals["m"], sf, goods["m"], 1, 1)
    context = (remmask & lowmask & lamfmask).astype(np.int8)
    context[~np.repeat(goods["l"] & goods["r"] & goods["m"],30)] = -1
    rows, counts = enhanced_arousal(c, sleep_like, sp, mbm, context,
                                    rm["rms"], rm["valid"], cfg["enhanced"]["arousal"])
    add("enhanced", "arousal", rows, ["c", "m"], cfg["enhanced"]["arousal"], "fixed-reference-shift-v2-experimental")
    stats["arousal"] = {"original": old_counts, "enhanced": counts}
    for mode, events in banks.items():
        events.sort(key=lambda e: (e.onset, e.offset, e.event_type, e.channel))
        for i, event in enumerate(events):
            event.event_id = f"{info['record_id']}:{mode}:{i:07d}"
        arousals = [e for e in events if e.event_type == "arousal"]
        extra = 3 if mode == "original" else 1
        for e in events:
            if e.event_type == "kcomplex":
                e.measurements["associated_arousal_ids"] = [a.event_id for a in arousals
                    if a.onset <= e.offset + extra and a.offset > e.onset]
        stats[mode + "_counts"] = {kind: sum(e.event_type == kind for e in events)
                                   for kind in sorted(available[mode])}
    return {"info": info, "events": {k: [asdict(e) for e in v] for k, v in banks.items()},
            "availability": available, "goods": {k: v.tolist() for k, v in goods.items()},
            "diagnostics": stats}
