from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.audio.load import TARGET_SAMPLE_RATE, load_audio


class AudioLoadTests(unittest.TestCase):
    def test_load_audio_downmixes_and_resamples(self) -> None:
        source_rate = 16_000
        samples = np.column_stack(
            [np.full(source_rate // 10, 0.25, dtype=np.float32),
             np.full(source_rate // 10, 0.75, dtype=np.float32)]
        )
        fake_soundfile = SimpleNamespace(read=Mock(return_value=(samples, source_rate)))
        with patch.dict(sys.modules, {"soundfile": fake_soundfile}):
            with patch.object(Path, "is_file", return_value=True):
                audio = load_audio("synthetic-stereo.wav")

        self.assertEqual(audio.sample_rate, TARGET_SAMPLE_RATE)
        self.assertEqual(audio.original_sample_rate, source_rate)
        self.assertEqual(audio.original_channels, 2)
        self.assertAlmostEqual(audio.duration_seconds, 0.1)
        self.assertEqual(audio.waveform.shape, (TARGET_SAMPLE_RATE // 10,))
        self.assertTrue(np.isclose(float(audio.waveform.mean()), 0.5, atol=0.01))
