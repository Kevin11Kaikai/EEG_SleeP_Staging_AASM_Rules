"""Physical-unit preprocessing with no resampling/filtering across invalid epochs."""
from fractions import Fraction
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt, welch
from ..temporal import runs


def quality(x, sf: float, kind: str, bounds: tuple, cfg: dict) -> np.ndarray:
    width = round(sf * 30)
    chunks = x[:len(x) // width * width].reshape(-1, width)
    finite = np.isfinite(chunks).all(axis=1)
    clean = np.nan_to_num(chunks)
    repeated = (np.diff(clean, axis=1) == 0).mean(axis=1)
    saturated = ((clean <= bounds[0] * .999) | (clean >= bounds[1] * .999)).mean(axis=1)
    return (finite & (np.ptp(clean, axis=1) >= cfg["min_ptp_uv"])
            & (np.abs(clean).max(axis=1) <= cfg["max_abs_uv"][kind])
            & (repeated <= cfg["repeated_fraction"])
            & (saturated <= cfg["saturation_fraction"]))


def band_filter(x, sf: float, band, order: int = 4) -> np.ndarray:
    return sosfiltfilt(butter(order, band, btype="bandpass", fs=sf, output="sos"), x)


def prepare(x, sf: float, target: float, good, band, order: int = 4) -> np.ndarray:
    result = np.zeros(len(good) * round(target * 30))
    ratio = Fraction(target / sf).limit_denominator(10000)
    for a, b in runs(good, 1 / 30):
        segment = x[round(a * sf):round(b * sf)]
        if sf != target:
            segment = resample_poly(segment, ratio.numerator, ratio.denominator)
        result[round(a * target):round(b * target)] = band_filter(segment, target, band, order)
    return result


def spectra(x, sf: float, good, window: float = 1, step: float = 1) -> dict:
    """Batch Welch; each row represents its actual complete measurement window."""
    width, stride = round(window * sf), round(step * sf)
    starts = np.arange(0, len(x) - width + 1, stride)
    windows = np.lib.stride_tricks.sliding_window_view(x, width)[::stride]
    mask = np.repeat(good, round(30 * sf))
    valid = np.lib.stride_tricks.sliding_window_view(mask, width)[::stride].all(axis=1)
    f, p = welch(windows, fs=sf, nperseg=min(width, round(2 * sf)), axis=1)
    total = p[:, (f >= .3) & (f <= 35)].sum(axis=1)
    total = np.maximum(total, 1e-30)
    result = {"start": starts / sf, "end": (starts + width) / sf,
              "valid": valid, "rms": np.sqrt(np.mean(windows ** 2, axis=1)),
              "ptp": np.ptp(windows, axis=1), "total": total}
    for name, lo, hi in (("delta", .3, 4), ("theta", 4, 7), ("alpha", 8, 12),
                          ("sigma", 12, 16), ("beta", 16.01, 30), ("fast", 8, 30)):
        result[name] = p[:, (f >= lo) & (f <= hi)].sum(axis=1) / total
    selected = (f >= .3) & (f <= 35)
    result["frequency"] = (p[:, selected] * f[selected]).sum(axis=1) / total
    return result


def rms_windows(x, sf: float, good, window: float = 1, step: float = .5) -> dict:
    width, stride = round(window * sf), round(step * sf)
    starts = np.arange(0, len(x) - width + 1, stride)
    view = np.lib.stride_tricks.sliding_window_view(x, width)[::stride]
    mask = np.repeat(good, round(30 * sf))
    valid = np.lib.stride_tricks.sliding_window_view(mask, width)[::stride].all(axis=1)
    return {"start": starts / sf, "end": (starts + width) / sf, "valid": valid,
            "rms": np.sqrt(np.mean(view ** 2, axis=1))}
