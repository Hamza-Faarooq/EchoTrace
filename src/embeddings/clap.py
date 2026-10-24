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
        return self.encode_audio_batch([waveform], sample_rate)[0]

    def encode_audio_batch(
        self, waveforms: list[np.ndarray], sample_rate: int
    ) -> np.ndarray:
        """Encode a batch of mono windows with one processor/model call."""
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"CLAP expects {self.sample_rate} Hz audio; got {sample_rate} Hz. "
                "Resample explicitly before encoding."
            )
        max_samples = round(self.max_duration_seconds * sample_rate)
        if not waveforms:
            raise ValueError("Audio batch must contain at least one waveform.")
        batch = []
        for waveform in waveforms:
            audio = np.asarray(waveform, dtype=np.float32)
            if audio.ndim != 1:
                raise ValueError(f"Expected mono 1-D audio, got shape {audio.shape}.")
            if audio.size == 0 or not np.isfinite(audio).all():
                raise ValueError("Audio must be non-empty and contain only finite samples.")
            if audio.size > max_samples:
                raise ValueError(
                    f"Audio window is {audio.size / sample_rate:.2f}s; CLAP accepts up to "
                    f"{self.max_duration_seconds:g}s per window. Split the recording first."
                )
            batch.append(audio)
        # rand_trunc keeps the pinned short-window path deterministic and avoids
        # the model's fusion behavior for long clips.
        inputs = self.processor(
            audios=batch,
            sampling_rate=sample_rate,
            truncation="rand_trunc",
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            features = self.model.get_audio_features(**inputs)
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
        return self._checked_matrix(features, len(batch), "audio")

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a natural-language query as a normalized CLAP vector."""
        return self.encode_text_batch([text])[0]

    def encode_text_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of natural-language queries in one model call."""
        if not texts:
            raise ValueError("Text batch must contain at least one query.")
        cleaned = [text.strip() for text in texts if text and text.strip()]
        if len(cleaned) != len(texts):
            raise ValueError("Query text must not be empty.")
        inputs: Any = self.processor(
            text=cleaned, padding=True, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            features = self.model.get_text_features(**inputs)
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
        return self._checked_matrix(features, len(cleaned), "text")

    @staticmethod
    def _checked_matrix(
        features: torch.Tensor, batch_size: int, kind: str
    ) -> np.ndarray:
        matrix = features.detach().cpu().numpy().astype(np.float32, copy=False)
        if (
            matrix.ndim != 2
            or matrix.shape[0] != batch_size
            or matrix.shape[1] == 0
            or not np.isfinite(matrix).all()
        ):
            raise RuntimeError(f"CLAP returned invalid {kind} embeddings.")
        if np.any(np.linalg.norm(matrix, axis=1) == 0):
            raise RuntimeError(f"CLAP returned a zero-norm {kind} embedding.")
        return matrix
