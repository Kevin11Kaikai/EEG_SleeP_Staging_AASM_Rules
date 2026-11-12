"""Half-open time intervals and explicit Kleene evidence."""
from enum import Enum


class Tri(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"

    def __and__(self, other):
        if self == Tri.FALSE or other == Tri.FALSE:
            return Tri.FALSE
        return Tri.TRUE if self == other == Tri.TRUE else Tri.UNKNOWN

    def __or__(self, other):
        if self == Tri.TRUE or other == Tri.TRUE:
            return Tri.TRUE
        return Tri.FALSE if self == other == Tri.FALSE else Tri.UNKNOWN

    def __invert__(self):
        return {Tri.TRUE: Tri.FALSE, Tri.FALSE: Tri.TRUE}.get(self, Tri.UNKNOWN)


def evidence(condition, available=True):
    return Tri.UNKNOWN if not available else (Tri.TRUE if condition else Tri.FALSE)


def intersection(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((float(start), float(end)))
    return result


def coverage(intervals, start, end):
    return sum(intersection(x, (start, end)) for x in union(intervals))


def onset_in(onset, start, end):
    return start <= onset < end


def pair_overlap(first, second, start, end):
    left, right = union(first), union(second)
    i, j, result = 0, 0, 0.0
    while i < len(left) and j < len(right):
        a, b = left[i], right[j]
        result += intersection((max(a[0], b[0]), min(a[1], b[1])), (start, end))
        if a[1] <= b[1]:
            i += 1
        else:
            j += 1
    return result
