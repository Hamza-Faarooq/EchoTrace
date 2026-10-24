"""Build a local window-embedding index from one audio directory/split."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from src.audio import load_audio, make_windows
from src.embeddings.clap import MODEL_ID, MODEL_REVISION, ClapEncoder
from src.embeddings.index import save_index

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff"}
SPLITS = ("development", "validation", "evaluation", "demo")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device")
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help="Hugging Face model ID or local checkpoint directory.",
    )
    parser.add_argument(
        "--model-id",
        help="Checkpoint ID to record when --model points to a local directory.",
    )
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--hop-seconds", type=float, default=5.0)
    parser.add_argument(
        "--limit", type=int, help="Index only the first N sorted files (exploratory subset)."
    )
    args = parser.parse_args(argv)

    if not args.audio_dir.is_dir():
        parser.error(f"audio directory does not exist: {args.audio_dir}")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.window_seconds <= 0 or args.hop_seconds <= 0:
        parser.error("window and hop durations must be positive")
    if args.hop_seconds > args.window_seconds:
        parser.error("--hop-seconds cannot exceed --window-seconds")

    # Clotho split folders are flat; keeping IDs equal to file_name makes the
    # caption join explicit and prevents a hidden path-based mismatch.
    files = sorted(
        (
            item
            for item in args.audio_dir.iterdir()
            if item.is_file() and item.suffix.lower() in AUDIO_SUFFIXES
        ),
        key=lambda item: item.name.casefold(),
    )
    if not files:
        parser.error(f"no supported audio files found directly under {args.audio_dir}")
    if len({item.name for item in files}) != len(files):
        parser.error("audio filenames must be unique within a split directory")
    if args.limit is not None:
        files = files[: args.limit]

    model_start = perf_counter()
    encoder = ClapEncoder.from_pretrained(
        model_name=args.model, device=args.device, revision=args.revision
    )
    model_load_seconds = perf_counter() - model_start

    vectors: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    inference_seconds = 0.0
    for audio_path in files:
        try:
            audio = load_audio(audio_path)
            windows = make_windows(
                audio.waveform,
                audio.sample_rate,
                window_seconds=args.window_seconds,
                hop_seconds=args.hop_seconds,
            )
            start = perf_counter()
            embeddings = encoder.encode_audio_batch(
                [samples for _, samples in windows], sample_rate=audio.sample_rate
            )
            inference_seconds += perf_counter() - start
        except Exception as exc:
            raise RuntimeError(f"Could not index {audio_path.name}: {exc}") from exc

        vectors.extend(embeddings)
        for window, _ in windows:
            records.append(
                {
                    "clip_id": audio_path.name,
                    "audio_file": audio_path.name,
                    "window_index": window.index,
                    "start_seconds": window.start,
                    "end_seconds": window.end,
                    "sample_rate": window.sample_rate,
                    "original_sample_rate": audio.original_sample_rate,
                    "duration_seconds": audio.duration_seconds,
                }
            )

    matrix = np.stack(vectors).astype(np.float32, copy=False)
    model_is_local = Path(args.model).is_dir()
    metadata = {
        "model_id": args.model_id or (MODEL_ID if model_is_local else args.model),
        "model_source": "local_checkpoint_directory" if model_is_local else "huggingface_hub",
        "model_revision": args.revision,
        "split": args.split,
        "sample_rate": encoder.sample_rate,
        "window_seconds": args.window_seconds,
        "hop_seconds": args.hop_seconds,
        "num_clips": len(files),
        "is_subset": args.limit is not None,
        "file_limit": args.limit,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(encoder.device),
        "model_load_seconds": round(model_load_seconds, 3),
        "audio_embedding_seconds": round(inference_seconds, 3),
    }
    save_index(args.output_dir, matrix, records, metadata)
    print(
        json.dumps(
            {
                "index_dir": str(args.output_dir),
                "split": args.split,
                "num_clips": len(files),
                "num_windows": len(records),
                "embedding_dim": int(matrix.shape[1]),
                "model_load_seconds": round(model_load_seconds, 3),
                "audio_embedding_seconds": round(inference_seconds, 3),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
