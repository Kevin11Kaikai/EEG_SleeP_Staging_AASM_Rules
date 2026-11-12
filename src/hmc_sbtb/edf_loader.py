"""PSG-only header inspection and physical signal access; never read hypnograms."""
from pathlib import Path
import mne
import numpy as np
from scipy.signal import resample_poly

# HMC official montage defines M1/M2; EDF uses the older equivalent A1/A2 names.
# Alias resolution does not construct new references or change EOG polarity.
ALIASES = {
    "frontal": ["EEG F4-M1", "EEG F4-A1", "F4:A1", "F4-M1", "F4-A1"],
    "central": ["EEG C4-M1", "EEG C4-A1", "C4:A1", "C4-M1", "C4-A1"],
    "occipital": ["EEG O2-M1", "EEG O2-A1", "O2:A1", "O2-M1", "O2-A1"],
    "auxiliary": ["EEG C3-M2", "EEG C3-A2", "C3:A2", "C3-M2", "C3-A2"],
    "eog_left": ["EOG E1-M2", "EOG E1-A2", "E1:A2", "EOG E1", "E1-M2"],
    "eog_right": ["EOG E2-M2", "EOG E2-A2", "E2:A2", "EOG E2", "E2-M2"],
    "emg": ["EMG chin", "Chin EMG", "EMG", "EMG chin1-chin2", "EMG 1-2"],
}


def volts_to_uv(signal):
    return np.asarray(signal, dtype=float) * 1e6


def read_signal_header(path):
    with Path(path).open("rb") as f:
        fixed = f.read(256)
        count = int(fixed[252:256])
        blob = f.read(count * 256)
    offset, result = 0, {}
    for key, width in [("label", 16), ("transducer", 80), ("dimension", 8),
                       ("physical_min", 8), ("physical_max", 8), ("digital_min", 8),
                       ("digital_max", 8), ("prefilter", 80), ("samples", 8), ("reserved", 32)]:
        result[key] = [blob[offset + i * width:offset + (i + 1) * width].decode("latin1").strip()
                       for i in range(count)]
        offset += count * width
    return result


def inspect_psg(path, config):
    path = Path(path)
    if "scoring" in path.name.lower():
        raise ValueError("prediction accepts PSG only")
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    extra = raw._raw_extras[0]
    header = read_signal_header(path)
    rows, selected = [], {}
    for role, aliases in ALIASES.items():
        requested = config["channel_map"][role]
        name = next((x for x in [requested] + aliases if x in raw.ch_names), None)
        selected[role] = name
    for idx, name in enumerate(raw.ch_names):
        rows.append({
            "channel": name, "role": next((k for k, v in selected.items() if v == name), "unused"),
            "native_sampling_hz": float(extra["n_samps"][idx] / extra["record_length"][0]),
            "original_unit": raw._orig_units.get(name), "mne_unit": "V",
            "detector_unit": "uV", "polarity_multiplier": 1,
            "reference_fidelity": "ADAPTED_A1_A2_legacy_mastoid_names" if "A1" in name or "A2" in name else "official_montage",
            "physical_min_uv": float(header["physical_min"][idx]),
            "physical_max_uv": float(header["physical_max"][idx]),
            "hardware_highpass_hz": float(extra["highpass"][idx]),
            "hardware_lowpass_hz": float(extra["lowpass"][idx]),
        })
    info = {"duration_sec": raw.n_times / raw.info["sfreq"],
            "start_datetime": raw.info["meas_date"].isoformat() if raw.info["meas_date"] else None,
            "subject_id": path.stem, "subject_identity": "record_proxy_not_verified_person_identifier",
            "selected_channels": selected, "channels": rows}
    raw.close()
    return info


def load_channel(path, name, target_hz):
    raw = mne.io.read_raw_edf(path, include=[name], preload=False, verbose=False)
    sf = float(raw.info["sfreq"])
    x = volts_to_uv(raw.get_data(picks=[name])[0])
    raw.close()
    if target_hz is not None and sf != target_hz:
        from fractions import Fraction
        ratio = Fraction(target_hz / sf).limit_denominator(1000)
        x = resample_poly(x, ratio.numerator, ratio.denominator)
    return x, sf
