"""Small, explicit, serializable interfaces between independent modules."""
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Truth(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"

    def __bool__(self):
        raise TypeError("Three-state evidence requires explicit `is Truth.TRUE`")

    def __invert__(self):
        return {Truth.TRUE: Truth.FALSE, Truth.FALSE: Truth.TRUE,
                Truth.UNKNOWN: Truth.UNKNOWN}[self]

    def __and__(self, other):
        if self is Truth.FALSE or other is Truth.FALSE:
            return Truth.FALSE
        return Truth.TRUE if self is other is Truth.TRUE else Truth.UNKNOWN

    def __or__(self, other):
        if self is Truth.TRUE or other is Truth.TRUE:
            return Truth.TRUE
        return Truth.FALSE if self is other is Truth.FALSE else Truth.UNKNOWN


def truth(value: bool | None) -> Truth:
    return Truth.UNKNOWN if value is None else (Truth.TRUE if value else Truth.FALSE)


@dataclass
class Event:
    event_id: str
    event_type: str
    onset: float
    offset: float
    channel: str
    detector_version: str
    parameters: dict
    quality_status: str = "PARTIALLY_SUPPORTED"
    evidence_status: str = "TRUE"
    measurements: dict = field(default_factory=dict)

    def __post_init__(self):
        if not 0 <= self.onset < self.offset:
            raise ValueError("Event intervals must be nonempty, half-open and nonnegative")


@dataclass
class Evidence:
    seconds: float | None
    event_ids: list[str] = field(default_factory=list)
    detector_status: str = "AVAILABLE"
    negative_validated: bool = False

    def threshold(self, value: float, inclusive: bool = False) -> Truth:
        if self.seconds is None or self.detector_status in ("UNAVAILABLE", "INVALID_SIGNAL"):
            return Truth.UNKNOWN
        return truth(self.seconds >= value if inclusive else self.seconds > value)

    def present(self, conservative: bool = True) -> Truth:
        """Strict clinical evidence vs explicitly requested observable candidate count.

        conservative=False tests a DIFFERENT proposition: whether this detector
        returned candidates. It never certifies clinical absence.
        """
        result = self.threshold(0)
        if conservative and result is Truth.FALSE and not self.negative_validated:
            return Truth.UNKNOWN
        return result


@dataclass
class EpochEvidence:
    index: int
    signals_good: bool
    events: dict[str, Evidence]
    attributes: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str) -> Evidence:
        return self.events.get(key, Evidence(None, detector_status="UNAVAILABLE"))


@dataclass
class StageDecision:
    epoch_index: int
    stage: str = "UNDETERMINED"
    rule_id: str = "NONE"
    evidence_event_ids: list[str] = field(default_factory=list)
    decision_source: str = "ABSTENTION"
    previous_context: dict = field(default_factory=dict)
    quality_status: str = "AVAILABLE"
    abstention_reason: str | None = "no TRUE rule"
    anchor: int | None = None
    chain_length: int = 0
    trace: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
