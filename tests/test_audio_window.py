from src.audio.load import AudioWindow, TARGET_SAMPLE_RATE
import pytest

def test_window_interval_is_valid() -> None:
    window = AudioWindow(start=1.0, end=3.0)
    assert (window.start, window.end) == (1.0, 3.0)

def test_sample_rate_is_48khz() -> None:
    assert TARGET_SAMPLE_RATE == 48_000

def test_invalid_window_is_rejected() -> None:
    with pytest.raises(ValueError):
        AudioWindow(start=3.0, end=1.0)
