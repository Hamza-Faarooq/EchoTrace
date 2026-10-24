"""Load CLAP once and run a shape/finite-value audio-text smoke check."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers

from src.embeddings.clap import ClapEncoder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="laion/clap-htsat-unfused", help="Hub model id or local model directory")
    parser.add_argument(
        "--revision",
        default="8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a",
        help="Pinned Hugging Face revision (ignored for a local model directory).",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device such as cuda:0")
    parser.add_argument("--query", default="A steady high-pitched electronic tone.")
    parser.add_argument("--output", type=Path, default=Path("results/clap_smoke.json"))
    args = parser.parse_args()

    sample_rate = 48_000
    duration = 5
    time_axis = np.arange(sample_rate * duration, dtype=np.float32) / sample_rate
    waveform = 0.15 * np.sin(2 * np.pi * 880 * time_axis)

    start = time.perf_counter()
    encoder = ClapEncoder.from_pretrained(model_name=args.model, device=args.device, revision=args.revision)
    load_seconds = time.perf_counter() - start
    audio_embedding = encoder.encode_audio(waveform, sample_rate)
    text_embedding = encoder.encode_text(args.query)
    similarity = float(audio_embedding @ text_embedding)

    result = {
        "purpose": "smoke_test_only_not_benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": "laion/clap-htsat-unfused" if Path(args.model).is_dir() else args.model,
        "loaded_from": args.model,
        "model_revision": encoder.revision,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device": str(encoder.device),
        "model_load_seconds": round(load_seconds, 3),
        "sample_rate": sample_rate,
        "audio_duration_seconds": duration,
        "audio_truncation": "rand_trunc",
        "query": args.query,
        "audio_embedding_shape": list(audio_embedding.shape),
        "text_embedding_shape": list(text_embedding.shape),
        "cosine_similarity": round(similarity, 6),
        "score_interpretation": "similarity score, not probability",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
