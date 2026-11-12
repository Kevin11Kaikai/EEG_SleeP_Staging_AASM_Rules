"""Experimental waveform candidates. Boundaries come from signal crossings/bases."""
import numpy as np
from scipy.signal import find_peaks, peak_widths
from ..preprocessing import band_filter
from ..temporal import runs


def biphasic_candidates(x, sf: float, good, cfg: dict, legacy: bool = False) -> list:
    results = []
    for start, end in runs(good, 1 / 30):
        base = round(start * sf)
        slow = band_filter(x[base:round(end * sf)], sf, cfg["band"])
        negative, _ = find_peaks(-slow, height=cfg["negative_uv"], distance=round(.5 * sf))
        crossing = np.flatnonzero(np.diff(np.signbit(slow)))
        for neg in negative:
            left, right = neg + round(.2 * sf), min(len(slow), neg + round(sf) + 1)
            if right <= left:
                continue
            pos = left + int(np.argmax(slow[left:right]))
            amplitude = slow[pos] - slow[neg]
            if not (slow[pos] >= cfg["positive_uv"] and
                    cfg["ptp_min_uv"] <= amplitude <= cfg["ptp_max_uv"]):
                continue
            if legacy:
                a, b = max(0, neg - round(.25 * sf)), min(len(slow), pos + round(.25 * sf))
            else:
                before = crossing[crossing < neg]
                after = crossing[crossing > pos]
                if not len(before) or not len(after):
                    continue
                a, b = int(before[-1]), int(after[0] + 1)
                if not cfg["duration_min_s"] <= (b - a) / sf <= cfg["duration_max_s"]:
                    continue
                # A large rapid discontinuity is not a physiological K-complex.
                if np.abs(np.diff(x[base + a:base + b])).max(initial=0) > 100:
                    continue
            results.append((start + a / sf, start + b / sf,
                            {"negative_uv": float(slow[neg]), "positive_uv": float(slow[pos]),
                             "ptp_uv": float(amplitude), "negative_time": start + neg / sf,
                             "positive_time": start + pos / sf,
                             "boundary": "fixed_legacy_padding" if legacy else "zero_crossings"}))
    return results


def ocular_candidates(left, right, sf: float, good, cfg: dict) -> dict:
    """Same-polarity short transients vs slow anti-phase deflections.

    HMC oblique EOG cannot prove eyelid motion. Blink/SEM remain PARTIALLY_SUPPORTED.
    """
    out = {"blink": [], "sem": []}
    for lo, hi in runs(good, 1 / 30):
        origin = round(lo * sf)
        l = band_filter(left[origin:round(hi * sf)], sf, [.3, 5])
        r = band_filter(right[origin:round(hi * sf)], sf, [.3, 5])
        for kind, signal in (("blink", (l + r) / 2), ("sem", (l - r) / 2)):
            settings = cfg[kind]
            peaks, props = find_peaks(np.abs(signal), height=settings["minimum_uv"],
                                     prominence=settings["minimum_uv"] / 2,
                                     distance=round(settings["duration_s"][0] * sf))
            if not len(peaks):
                continue
            # Width at 90% prominence is a reproducible signal-dependent boundary.
            widths = peak_widths(np.abs(signal), peaks, rel_height=.9)
            for j, peak in enumerate(peaks):
                a, b = int(np.floor(widths[2][j])), int(np.ceil(widths[3][j])) + 1
                d = (b - a) / sf
                if not settings["duration_s"][0] <= d <= settings["duration_s"][1]:
                    continue
                corr = float(np.corrcoef(l[a:b], r[a:b])[0, 1])
                if not np.isfinite(corr):
                    continue
                if kind == "blink":
                    accepted = corr >= settings["same_polarity_correlation"]
                    accepted &= props["peak_heights"][j] <= settings["maximum_uv"]
                else:
                    accepted = corr <= settings["negative_correlation"]
                    accepted &= (peak-widths[2][j])/sf >= settings["minimum_initial_deflection_s"]
                if accepted:
                    out[kind].append((lo + a / sf, lo + b / sf,
                                      {"correlation": corr, "peak_uv": float(abs(signal[peak])),
                                       "duration_s": d, "initial_deflection_s":float((peak-widths[2][j])/sf),
                                       "boundary": "90_percent_prominence"}))
    return out


def vertex_candidates(central, frontal, sf: float, good, cfg: dict) -> list:
    out = []
    for lo, hi in runs(good, 1 / 30):
        a0 = round(lo * sf)
        x = band_filter(central[a0:round(hi * sf)], sf, [1, 15])
        f = band_filter(frontal[a0:round(hi * sf)], sf, [1, 15])
        peaks, _ = find_peaks(-x, height=cfg["negative_uv"], prominence=cfg["negative_uv"],
                              distance=round(.5 * sf))
        if not len(peaks):
            continue
        width = peak_widths(-x, peaks, rel_height=.5)
        for j, peak in enumerate(peaks):
            d = width[0][j] / sf
            if cfg["duration_s"][0] <= d <= cfg["duration_s"][1] and abs(x[peak]) > abs(f[peak]):
                out.append((lo + width[2][j] / sf, lo + width[3][j] / sf,
                            {"central_peak_uv": float(x[peak]), "frontal_uv": float(f[peak]),
                             "limitation": "C4 proxy; Cz not recorded"}))
    return out
