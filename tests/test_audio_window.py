from __future__ import annotations

import unittest

import numpy as np

from src.audio.load import AudioWindow, make_windows


class AudioWindowTests(unittest.TestCase):
    def test_window_interval_is_valid(self) -> None:
        window = AudioWindow(start=1.0, end=3.0)
        self.assertEqual((window.start, window.end), (1.0, 3.0))

    def test_sample_rate_is_48khz(self) -> None:
        from src.audio.load import TARGET_SAMPLE_RATE

        self.assertEqual(TARGET_SAMPLE_RATE, 48_000)

    def test_invalid_window_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AudioWindow(start=3.0, end=1.0)

    def test_overlapping_windows_preserve_tail_and_timestamps(self) -> None:
        waveform = np.arange(16, dtype=np.float32)
        windows = make_windows(waveform, sample_rate=1, window_seconds=10, hop_seconds=5)
        self.assertEqual(
            [(meta.start_sample, meta.end_sample) for meta, _ in windows],
            [(0, 10), (5, 15), (10, 16)],
        )
        self.assertEqual(windows[-1][1].tolist(), [10, 11, 12, 13, 14, 15])
        self.assertEqual(windows[-1][0].end, 16.0)

    def test_windowing_rejects_bad_audio_and_gaps(self) -> None:
        with self.assertRaisesRegex(ValueError, "mono 1-D"):
            make_windows(np.ones((2, 10)), sample_rate=1)
        with self.assertRaisesRegex(ValueError, "finite"):
            make_windows(np.array([0.0, np.nan]), sample_rate=1)
        with self.assertRaisesRegex(ValueError, "skip audio"):
            make_windows(np.ones(20), sample_rate=1, window_seconds=5, hop_seconds=6)

