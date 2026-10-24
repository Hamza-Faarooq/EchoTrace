"""Audio decoding, mono conversion, resampling, and timestamped windowing."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

TARGET_SAMPLE_RATE = 48_000


@dataclass(frozen=True)
class LoadedAudio:
    """Decoded mono audio plus source metadata needed for reproducibility."""

    waveform: np.ndarray
    sample_rate: int
    original_sample_rate: int
    duration_seconds: float
    original_channels: int


@dataclass(frozen=True)
class AudioWindow:
    """A half-open interval [start, end) in seconds on the source recording."""

    start: float
    end: float
    index: int = 0
    start_sample: int | None = None
    end_sample: int | None = None
    sample_rate: int = TARGET_SAMPLE_RATE

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        if self.index < 0:
            raise ValueError("index must be non-negative")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.start_sample is not None and self.start_sample < 0:
            raise ValueError("start_sample must be non-negative")
        if self.end_sample is not None and self.end_sample <= (self.start_sample or 0):
            raise ValueError("end_sample must be greater than start_sample")


def load_audio(path: str | Path, target_sample_rate: int = TARGET_SAMPLE_RATE) -> LoadedAudio:
    """Decode an audio file, downmix to mono, and explicitly resample it.

    Returns the original duration and sample rate as metadata. Unsupported or
    corrupt files raise a ValueError with the path and decoder error.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {source}")
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be positive")
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("Install the pinned soundfile dependency to decode audio.") from exc
    try:
        samples, original_sample_rate = sf.read(source, dtype="float32", always_2d=True)
    except (RuntimeError, OSError) as exc:
        raise ValueError(f"Could not decode audio file {source}: {exc}") from exc
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError(f"Audio file has no samples or channels: {source}")
    if not np.isfinite(samples).all():
        raise ValueError(f"Audio file contains NaN or infinite samples: {source}")

    original_channels = int(samples.shape[1])
    duration_seconds = samples.shape[0] / int(original_sample_rate)
    mono = samples.mean(axis=1, dtype=np.float32)
    if int(original_sample_rate) != target_sample_rate:
        divisor = gcd(int(original_sample_rate), target_sample_rate)
        mono = resample_poly(
            mono,
            target_sample_rate // divisor,
            int(original_sample_rate) // divisor,
        ).astype(np.float32, copy=False)
    return LoadedAudio(
        waveform=np.asarray(mono, dtype=np.float32),
        sample_rate=target_sample_rate,
        original_sample_rate=int(original_sample_rate),
        duration_seconds=float(duration_seconds),
        original_channels=original_channels,
    )


def make_windows(
    waveform: np.ndarray,
    sample_rate: int,
    window_seconds: float = 10.0,
    hop_seconds: float = 5.0,
) -> list[tuple[AudioWindow, np.ndarray]]:
    """Create deterministic overlapping windows without dropping the tail."""
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono 1-D audio, got shape {audio.shape}.")
    if audio.size == 0:
        raise ValueError("Audio is empty.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer.")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or infinite samples.")
    if window_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive.")
    if hop_seconds > window_seconds:
        raise ValueError("hop_seconds cannot exceed window_seconds; that would skip audio.")

    window_size = max(1, round(window_seconds * sample_rate))
    hop_size = max(1, round(hop_seconds * sample_rate))
    windows: list[tuple[AudioWindow, np.ndarray]] = []
    start_sample = 0
    index = 0
    while start_sample < audio.size:
        end_sample = min(start_sample + window_size, audio.size)
        start = start_sample / sample_rate
        end = end_sample / sample_rate
        metadata = AudioWindow(
            start=start,
            end=end,
            index=index,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=sample_rate,
        )
        windows.append((metadata, audio[start_sample:end_sample]))
        if end_sample == audio.size:
            break
        start_sample += hop_size
        index += 1
    return windows
