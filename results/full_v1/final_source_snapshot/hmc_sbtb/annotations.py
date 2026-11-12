"""Evaluation-only parsing, raw-label preservation and exact half-open alignment."""
import csv
from datetime import datetime, timezone
from pathlib import Path
import re
import mne
from .temporal import intersection, union

LABELS = {"Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
          "Sleep stage 3": "N3", "Sleep stage R": "R", "W": "W", "N1": "N1",
          "Sleep stage N1": "N1", "Sleep stage N2": "N2", "Sleep stage N3": "N3",
          "N2": "N2", "N3": "N3", "R": "R", "REM": "R"}


def edf_annotations(path, psg_start):
    annotations = mne.read_annotations(path)
    offset = 0.0
    if annotations.orig_time and psg_start:
        start = datetime.fromisoformat(psg_start)
        offset = (annotations.orig_time - start).total_seconds()
    return [{"onset": float(a), "duration": float(b), "raw_label": str(c),
             "label": LABELS.get(str(c).strip()), "origin_offset_sec": offset,
             "start": float(a + offset), "end": float(a + b + offset)}
            for a, b, c in zip(annotations.onset, annotations.duration, annotations.description)]


def parse_number(value):
    value = value.strip().strip('"')
    if ":" in value:
        sign = -1 if value.startswith("-") else 1
        pieces = [float(x) for x in value.lstrip("+-").split(":")]
        return sign * sum(x * 60 ** i for i, x in enumerate(reversed(pieces)))
    return float(value)


def txt_annotations(path, psg_start):
    text = Path(path).read_text(encoding="utf-8-sig", errors="strict")
    lines = text.splitlines()
    header_index = next((i for i, line in enumerate(lines) if "Recording onset" in line and "Annotation" in line), None)
    if header_index is None:
        raise ValueError("TXT header not recognized; no row-number inference allowed")
    dialect = csv.Sniffer().sniff(lines[header_index], delimiters=",;\t")
    reader = csv.DictReader(lines[header_index:], dialect=dialect, skipinitialspace=True)
    result = []
    for raw in reader:
        row = {k.strip(): (v.strip() if v else "") for k, v in raw.items() if k}
        if not row.get("Annotation"):
            continue
        onset, duration = parse_number(row["Recording onset"]), parse_number(row["Duration"])
        absolute_offset = None
        if row.get("Date") and row.get("Time") and psg_start:
            clock = row["Time"]
            if ":" not in clock:
                pieces = clock.split(".")
                clock = ":".join(pieces[:3]) + ("." + pieces[3] if len(pieces) > 3 else "")
            absolute_text = row["Date"] + " " + clock
            formats = ["%d.%m.%Y %H:%M:%S", "%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%d.%m.%y %H:%M:%S"]
            for fmt in formats + [f + ".%f" for f in formats]:
                try:
                    absolute = datetime.strptime(absolute_text, fmt).replace(tzinfo=timezone.utc)
                    absolute_offset = (absolute - datetime.fromisoformat(psg_start)).total_seconds()
                    break
                except ValueError:
                    pass
        label = row["Annotation"]
        result.append({"onset": onset, "duration": duration, "raw_label": label,
                       "label": LABELS.get(label), "start": onset, "end": onset + duration,
                       "absolute_onset_sec": absolute_offset,
                       "absolute_relative_discrepancy_sec": absolute_offset - onset if absolute_offset is not None else None})
    return result


def align(rows, epochs):
    result = []
    for i in range(epochs):
        a, b = i * 30, (i + 1) * 30
        overlapping = [r for r in rows if intersection((r["start"], r["end"]), (a, b)) > 1e-8]
        stages = [r for r in overlapping if r["label"] is not None]
        unknown_stages = [r for r in overlapping if r["label"] is None and "sleep stage" in r["raw_label"].lower()]
        labels = set(r["label"] for r in stages)
        covered = sum(y - x for x, y in union([(max(a, r["start"]), min(b, r["end"])) for r in stages]))
        overlaps = sum(intersection((x["start"], x["end"]), (y["start"], y["end"]))
                       for j, x in enumerate(stages) for y in stages[j + 1:]
                       if x["label"] != y["label"])
        label = next(iter(labels)) if len(labels) == 1 and abs(covered - 30) < 1e-6 and not unknown_stages and overlaps == 0 else None
        reason = "ALIGNED" if label else "UNKNOWN_LABEL" if unknown_stages else "CONFLICT" if len(labels) > 1 else "GAP_OR_PARTIAL"
        result.append({"epoch_index": i, "true_stage": label or "UNALIGNED", "alignment_status": reason,
                       "covered_sec": covered, "raw_labels": sorted(set(r["raw_label"] for r in overlapping))})
    return result
