"""Audio input utilities for EchoTrace.

Planned responsibilities: validated loading, conversion to mono, resampling to
48 kHz, and timestamped windows. The implementation will be expanded next.
"""
from dataclasses import dataclass

TARGET_SAMPLE_RATE = 48_000

@dataclass(frozen=True)
class AudioWindow:
    """A half-open interval [start, end) in seconds."""
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
