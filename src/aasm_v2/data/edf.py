"""Minimal native EDF header inspection; MNE physical-value signal reader."""
from pathlib import Path

CHANNELS = {
    "f": ("EEG F4-M1", "EEG F4-A1"),
    "c": ("EEG C4-M1", "EEG C4-A1"),
    "o": ("EEG O2-M1", "EEG O2-A1"),
    "l": ("EOG E1-M2", "EOG E1-A2"),
    "r": ("EOG E2-M2", "EOG E2-A2"),
    "m": ("EMG chin", "Chin EMG"),
}


def unit_factor(unit: str) -> float:
    normalized = unit.replace("µ", "u").replace("μ", "u").lower().strip()
    factors = {"uv": 1.0, "mv": 1e3, "v": 1e6}
    if normalized not in factors:
        raise ValueError(f"Unsupported physical unit: {unit!r}")
    return factors[normalized]


def header(path: str | Path) -> dict:
    path = Path(path)
    if "scoring" in path.name.lower() or path.suffix.lower() != ".edf":
        raise ValueError("Prediction API accepts a signal PSG EDF only")
    with path.open("rb") as f:
        fixed = f.read(256)
        n = int(fixed[252:256])
        raw = f.read(256 * n)
    fields = {}
    position = 0
    for key, width in (("name", 16), ("transducer", 80), ("unit", 8),
                       ("pmin", 8), ("pmax", 8), ("dmin", 8), ("dmax", 8),
                       ("prefilter", 80), ("samples", 8), ("reserved", 32)):
        fields[key] = [raw[position + j * width:position + (j + 1) * width]
                       .decode("latin1").strip() for j in range(n)]
        position += n * width
    period = float(fixed[244:252])
    records = int(fixed[236:244])
    size = int(fixed[184:192]) + records * sum(map(int, fields["samples"])) * 2
    if records < 0 or size != path.stat().st_size:
        raise ValueError(f"EDF byte-count mismatch: {path.name}")
    if fixed[192:236].startswith(b"EDF+D"):
        raise ValueError("Discontinuous EDF requires explicit discontinuity handling")
    channels = []
    for j in range(n):
        if "Annotations" in fields["name"][j]:
            continue
        factor = unit_factor(fields["unit"][j])
        channels.append({"name": fields["name"][j], "unit": fields["unit"][j],
                         "sf": int(fields["samples"][j]) / period,
                         "min_uv": float(fields["pmin"][j]) * factor,
                         "max_uv": float(fields["pmax"][j]) * factor,
                         "prefilter": fields["prefilter"][j], "polarity": 1})
    names = [x["name"] for x in channels]
    return {"record_id": path.stem, "duration": records * period,
            "channels": channels, "size": size,
            "roles": {role: next((v for v in variants if v in names), None)
                      for role, variants in CHANNELS.items()},
            "identity": "recording_proxy; real-person linkage unavailable"}


def read_channel(path: str | Path, role: str, info: dict) -> tuple:
    import mne
    name = info["roles"][role]
    if name is None:
        return None, None
    raw = mne.io.read_raw_edf(str(path), include=[name], preload=False, verbose="ERROR")
    try:
        values = raw.get_data(picks=[name])[0] * 1e6
        return values, float(raw.info["sfreq"])
    finally:
        raw.close()
