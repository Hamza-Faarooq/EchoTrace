"""Hugging Face CLAP audio/text encoder with explicit preprocessing checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

MODEL_ID = "laion/clap-htsat-unfused"
MODEL_REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"


@dataclass
class ClapEncoder:
    """One reusable CLAP model/processor pair; never load it at import time."""

    model: ClapModel
    processor: ClapProcessor
    device: torch.device
    sample_rate: int = 48_000
    max_duration_seconds: float = 10.0
    revision: str = MODEL_REVISION

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = MODEL_ID,
        device: str = "auto",
        revision: str = MODEL_REVISION,
    ) -> "ClapEncoder":
        """Load one pinned checkpoint; callers should cache this instance."""
        if device == "auto":
            selected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            selected = torch.device(device)
            if selected.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but no CUDA device is available.")
        revision_kwargs = {} if Path(model_name).is_dir() else {"revision": revision}
        model = ClapModel.from_pretrained(model_name, **revision_kwargs).to(selected).eval()
        processor = ClapProcessor.from_pretrained(model_name, **revision_kwargs)
        feature_extractor = processor.feature_extractor
        return cls(
            model=model,
            processor=processor,
            device=selected,
            sample_rate=int(feature_extractor.sampling_rate),
            max_duration_seconds=float(feature_extractor.max_length_s),
            revision=revision,
        )

    def encode_audio(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Encode one mono waveform; resample and window it before this call."""
        audio = np.asarray(waveform, dtype=np.float32)
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"CLAP expects {self.sample_rate} Hz audio; got {sample_rate} Hz. "
                "Resample explicitly before encoding."
            )
        if audio.ndim != 1:
            raise ValueError(f"Expected mono 1-D audio, got shape {audio.shape}.")
        if audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError("Audio must be non-empty and contain only finite samples.")
        max_samples = round(self.max_duration_seconds * sample_rate)
        if audio.size > max_samples:
            raise ValueError(
                f"Audio window is {audio.size / sample_rate:.2f}s; CLAP accepts up to "
                f"{self.max_duration_seconds:g}s per window. Split the recording first."
            )
        # rand_trunc keeps the pinned short-window path deterministic and avoids
        # the model's fusion behavior for long clips.
        inputs = self.processor(
            audios=audio,
            sampling_rate=sample_rate,
            truncation="rand_trunc",
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            features = self.model.get_audio_features(**inputs)
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
        return self._checked_vector(features, "audio")

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a natural-language query as a normalized CLAP vector."""
        if not text or not text.strip():
            raise ValueError("Query text must not be empty.")
        inputs: Any = self.processor(text=text.strip(), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            features = self.model.get_text_features(**inputs)
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
        return self._checked_vector(features, "text")

    @staticmethod
    def _checked_vector(features: torch.Tensor, kind: str) -> np.ndarray:
        vector = features.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            raise RuntimeError(f"CLAP returned an invalid {kind} embedding.")
        return vector
