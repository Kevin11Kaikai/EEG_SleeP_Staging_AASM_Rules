"""Continuous-time half-open interval operations, independent of epoch boundaries."""
import numpy as np


def runs(mask, rate: float = 1.0, minimum: float = 0.0) -> list[tuple]:
    changes = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return [(a / rate, b / rate) for a, b in changes.reshape(-1, 2)
            if (b - a) / rate >= minimum]


def merge(intervals) -> list[tuple]:
    out = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def intersect(left, right) -> list[tuple]:
    left, right = merge(left), merge(right)
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        a, b = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if b > a:
            result.append((a, b))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def duration(intervals, start: float, end: float) -> float:
    return sum(b - a for a, b in intersect(intervals, [(start, end)]))


def raster(intervals, n: int, rate: float = 1.0) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    for a, b in intervals:
        out[max(0, int(a * rate)):min(n, int(np.ceil(b * rate)))] = True
    return out
