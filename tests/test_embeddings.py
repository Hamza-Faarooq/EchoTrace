from __future__ import annotations

import unittest

import numpy as np
import torch

from src.embeddings.clap import ClapEncoder


class ClapEncoderInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder = ClapEncoder(model=None, processor=None, device=torch.device("cpu"))

    def test_rejects_wrong_sample_rate_and_long_windows(self) -> None:
        with self.assertRaisesRegex(ValueError, "expects 48000 Hz"):
            self.encoder.encode_audio(np.ones(100, dtype=np.float32), sample_rate=16_000)
        with self.assertRaisesRegex(ValueError, "Split the recording first"):
            self.encoder.encode_audio(np.ones(480_001, dtype=np.float32), sample_rate=48_000)

    def test_rejects_empty_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.encoder.encode_text("  ")
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.encoder.encode_text_batch([])

    def test_rejects_empty_audio_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.encoder.encode_audio_batch([], sample_rate=48_000)
