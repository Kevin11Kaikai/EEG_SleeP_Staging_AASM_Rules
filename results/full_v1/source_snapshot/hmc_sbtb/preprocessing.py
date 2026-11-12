"""Physical-unit filters, unlabeled quality masks, spectra, and EMG envelope."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch


def bandpass(x, sf, band, order=4):
    return sosfiltfilt(butter(order, band, btype="bandpass", fs=sf, output="sos"), x)


def quality_epochs(x, sf, kind, cfg, physical_bounds=None):
    n = len(x) // int(sf * 30)
    x = x[:n * int(sf * 30)].reshape(n, -1)
    finite = np.isfinite(x).all(axis=1)
    clean = np.nan_to_num(x)
    ptp = np.ptp(clean, axis=1)
    maximum = np.max(np.abs(clean), axis=1)
    limit = cfg[f"max_abs_{kind}_uv"]
    flat = ptp < cfg["min_ptp_uv"]
    # A long repeated exact sample level can signal a clipped/disconnected electrode.
    repeated = (np.diff(clean, axis=1) == 0).mean(axis=1)
    saturation = repeated > 0.2
    if physical_bounds is not None:
        lo, hi = physical_bounds
        saturation |= ((clean <= lo * 0.999) | (clean >= hi * 0.999)).mean(axis=1) > cfg["saturation_fraction"]
    good = finite & ~flat & (maximum <= limit) & ~saturation
    f, p = welch(clean, fs=sf, nperseg=min(int(sf * 2), clean.shape[1]), axis=1)
    denominator = p[:, (f >= 0.3) & (f <= 55)].sum(axis=1)
    line = np.divide(p[:, (f >= 49) & (f <= 51)].sum(axis=1), denominator,
                     out=np.zeros(n), where=denominator > 0)
    return good, [{"epoch_index": i, "finite": bool(finite[i]), "ptp_uv": float(ptp[i]),
                   "max_abs_uv": float(maximum[i]), "flat": bool(flat[i]),
                   "line_50hz_relative_power": float(line[i]),
                   "line_50hz_warning": bool(line[i] > cfg["line_ratio_warn"]),
                   "saturation": bool(saturation[i]), "valid": bool(good[i])} for i in range(n)]


def filter_valid_blocks(x, sf, mask, band, order=4):
    out = np.zeros_like(x)
    for a, b in segments(mask, sf):
        first, last = round(a * sf), round(b * sf)
        if last - first > sf:
            out[first:last] = bandpass(x[first:last], sf, band, order)
    return out


def sample_mask(good, n, sf):
    mask = np.repeat(good, int(sf * 30))
    return np.pad(mask, (0, max(0, n - len(mask))), constant_values=False)[:n]


def segments(mask, sf, minimum=0):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [(a / sf, b / sf) for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0])
            if (b - a) / sf >= minimum]


def psd_ratio(x, sf, band, total, nperseg=None):
    f, p = welch(x, fs=sf, nperseg=nperseg or min(len(x), int(sf * 2)), detrend="constant")
    # Bin-sum integration avoids undercounting 1-Hz-wide bands in 1-second windows.
    denominator = p[(f >= total[0]) & (f <= total[1])].sum()
    return float(p[(f >= band[0]) & (f <= band[1])].sum() / denominator) if denominator > 0 else 0.0


def spectral_windows(x, sf, mask, window, step, band, total):
    width, advance = int(window * sf), int(step * sf)
    out = []
    for start in range(0, len(x) - width + 1, advance):
        if mask[start:start + width].all():
            out.append((start / sf, (start + width) / sf,
                        psd_ratio(x[start:start + width], sf, band, total)))
    return out


def emg_envelope(x, sf, mask, cfg):
    rows = []
    width, step = int(cfg["rms_window_sec"] * sf), int(cfg["step_sec"] * sf)
    for a in range(0, len(x) - width + 1, step):
        if mask[a:a + width].all():
            rows.append((a / sf, (a + width) / sf, float(np.sqrt(np.mean(x[a:a + width] ** 2)))))
    if not rows:
        return [], [], {"status": "INVALID_SIGNAL"}
    rms = np.array([r[2] for r in rows])
    lo, hi = np.quantile(rms, [cfg["low_quantile"], cfg["high_quantile"]])
    if lo <= 1e-8 or hi <= lo:
        return [], rows, {"status": "INVALID_SIGNAL", "reason": "degenerate tone baseline"}
    events = [(a, b, value, "low_emg" if value <= lo else "high_emg")
              for a, b, value in rows if value <= lo or value >= hi]
    return events, rows, {"status": "AVAILABLE", "low_rms_uv": float(lo), "high_rms_uv": float(hi)}
