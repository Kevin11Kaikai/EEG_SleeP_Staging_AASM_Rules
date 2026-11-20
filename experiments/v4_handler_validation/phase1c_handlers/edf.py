from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SignalHeader:
    index: int
    name: str
    transducer: str
    unit: str
    physical_min: float
    physical_max: float
    digital_min: int
    digital_max: int
    prefilter: str
    samples_per_record: int
    sampling_rate: float


@dataclass(frozen=True)
class EdfHeader:
    path: str
    header_bytes: int
    record_count: int
    record_duration_seconds: float
    duration_seconds: float
    signals: tuple[SignalHeader, ...]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["signals"] = [asdict(item) for item in self.signals]
        return value


def _decode_many(payload: bytes, count: int, width: int, offset: int) -> tuple[list[str], int]:
    values = [
        payload[offset + i * width : offset + (i + 1) * width]
        .decode("latin1", errors="replace")
        .strip()
        for i in range(count)
    ]
    return values, offset + count * width


def read_header(path: str | Path) -> EdfHeader:
    source = Path(path)
    with source.open("rb") as stream:
        fixed = stream.read(256)
        if len(fixed) != 256:
            raise ValueError(f"Truncated EDF fixed header: {source}")
        signal_count = int(fixed[252:256].decode("ascii").strip())
        signal_header = stream.read(signal_count * 256)
    header_bytes = int(fixed[184:192].decode("ascii").strip())
    record_count = int(fixed[236:244].decode("ascii").strip())
    record_duration = float(fixed[244:252].decode("ascii").strip())
    if record_count <= 0 or record_duration <= 0:
        raise ValueError(f"Unsupported EDF record layout: {source}")
    offset = 0
    labels, offset = _decode_many(signal_header, signal_count, 16, offset)
    transducers, offset = _decode_many(signal_header, signal_count, 80, offset)
    units, offset = _decode_many(signal_header, signal_count, 8, offset)
    physical_min, offset = _decode_many(signal_header, signal_count, 8, offset)
    physical_max, offset = _decode_many(signal_header, signal_count, 8, offset)
    digital_min, offset = _decode_many(signal_header, signal_count, 8, offset)
    digital_max, offset = _decode_many(signal_header, signal_count, 8, offset)
    prefilters, offset = _decode_many(signal_header, signal_count, 80, offset)
    samples, offset = _decode_many(signal_header, signal_count, 8, offset)
    _, offset = _decode_many(signal_header, signal_count, 32, offset)
    if offset != len(signal_header):
        raise AssertionError("EDF per-signal header parser did not consume the header")
    signals = []
    for index in range(signal_count):
        samples_per_record = int(samples[index])
        signals.append(
            SignalHeader(
                index=index,
                name=labels[index],
                transducer=transducers[index],
                unit=units[index],
                physical_min=float(physical_min[index]),
                physical_max=float(physical_max[index]),
                digital_min=int(digital_min[index]),
                digital_max=int(digital_max[index]),
                prefilter=prefilters[index],
                samples_per_record=samples_per_record,
                sampling_rate=samples_per_record / record_duration,
            )
        )
    expected = header_bytes + record_count * sum(item.samples_per_record for item in signals) * 2
    if source.stat().st_size != expected:
        raise ValueError(
            f"EDF byte-count mismatch for {source.name}: {source.stat().st_size} != {expected}"
        )
    return EdfHeader(
        path=str(source.resolve()),
        header_bytes=header_bytes,
        record_count=record_count,
        record_duration_seconds=record_duration,
        duration_seconds=record_count * record_duration,
        signals=tuple(signals),
    )


def unit_to_microvolts(unit: str) -> float:
    normalized = unit.replace("µ", "u").replace("μ", "u").strip().lower()
    factors = {"uv": 1.0, "mv": 1_000.0, "v": 1_000_000.0}
    if normalized not in factors:
        raise ValueError(f"Unsupported voltage unit: {unit!r}")
    return factors[normalized]


def read_channels(
    path: str | Path,
    names: Iterable[str],
    *,
    header: EdfHeader | None = None,
) -> dict[str, tuple[np.ndarray, float]]:
    source = Path(path)
    info = header or read_header(source)
    requested = list(names)
    by_name = {signal.name: signal for signal in info.signals}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise KeyError(f"Missing EDF channels in {source.name}: {missing}")
    samples_per_record = [signal.samples_per_record for signal in info.signals]
    total_samples = sum(samples_per_record)
    raw = np.memmap(
        source,
        dtype="<i2",
        mode="r",
        offset=info.header_bytes,
        shape=(info.record_count, total_samples),
    )
    starts = np.cumsum([0, *samples_per_record[:-1]])
    output: dict[str, tuple[np.ndarray, float]] = {}
    for name in requested:
        signal = by_name[name]
        left = int(starts[signal.index])
        right = left + signal.samples_per_record
        digital = np.asarray(raw[:, left:right], dtype=np.float64).reshape(-1)
        scale = (signal.physical_max - signal.physical_min) / (
            signal.digital_max - signal.digital_min
        )
        physical = signal.physical_min + (digital - signal.digital_min) * scale
        physical *= unit_to_microvolts(signal.unit)
        output[name] = (physical.astype(np.float32), signal.sampling_rate)
    del raw
    return output


def read_channel(
    path: str | Path, name: str, *, header: EdfHeader | None = None
) -> tuple[np.ndarray, float]:
    return read_channels(path, [name], header=header)[name]
