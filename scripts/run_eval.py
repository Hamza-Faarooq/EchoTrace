"""Evaluate caption-to-audio retrieval on one official Clotho split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from src.embeddings.clap import MODEL_ID, MODEL_REVISION, ClapEncoder
from src.embeddings.index import load_index
from src.evaluation.metrics import mean_average_precision, recall_at_k
from src.retrieval.search import search_clips_batch

SPLITS = ("development", "validation", "evaluation")


def read_captions(path: Path) -> list[dict[str, str]]:
    """Read caption rows without assuming more than the audited CSV schema."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if "file_name" not in fields:
            raise ValueError(f"{path} has no file_name column")
        caption_fields = sorted(
            (field for field in fields if field.startswith("caption_")),
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )
        if not caption_fields:
            raise ValueError(f"{path} has no caption_N columns")
        return [
            {
                "file_name": row["file_name"].strip(),
                **{field: (row.get(field) or "").strip() for field in caption_fields},
            }
            for row in reader
            if row.get("file_name")
        ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--captions-csv", required=True, type=Path)
    parser.add_argument("--index-dir", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/predictions/evaluation.json"),
        help="Local output path; prediction files are ignored by Git.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device")
    parser.add_argument("--model", help="Hub ID or local checkpoint directory for the indexed model.")
    parser.add_argument("--revision", help="Must match the model revision recorded in the index.")
    parser.add_argument("--aggregation", choices=("max", "mean"), default="max")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-clips", type=int, help="Evaluate a sorted exploratory subset.")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10])
    args = parser.parse_args(argv)

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_clips is not None and args.max_clips < 1:
        parser.error("--max-clips must be at least 1")
    if any(k < 1 for k in args.k_values):
        parser.error("every --k-values value must be at least 1")

    audio_vectors, manifest = load_index(args.index_dir)
    index_metadata: dict[str, Any] = manifest.get("metadata", {})
    if index_metadata.get("split") != args.split:
        parser.error(
            f"split mismatch: index is {index_metadata.get('split')!r}, "
            f"but --split is {args.split!r}"
        )
    expected_csv_name = f"clotho_captions_{args.split}.csv"
    if args.captions_csv.name.startswith("clotho_captions_") and (
        args.captions_csv.name != expected_csv_name
    ):
        parser.error(
            f"caption CSV name indicates a different split than --split {args.split!r}"
        )
    model_id = index_metadata.get("model_id", MODEL_ID)
    model_revision = index_metadata.get("model_revision", MODEL_REVISION)
    default_local = Path(".cache/models/clap-htsat-unfused")
    if args.model:
        model_name = args.model
    elif (
        index_metadata.get("model_source") == "local_checkpoint_directory"
        and model_id == MODEL_ID
        and default_local.is_dir()
    ):
        model_name = str(default_local)
    else:
        model_name = model_id
    model_is_local = Path(model_name).is_dir()
    if not model_is_local and model_name != model_id:
        parser.error("--model must identify the same checkpoint used to build this index")
    if args.revision and args.revision != model_revision:
        parser.error("--revision must match the model revision used to build this index")
    revision = model_revision
    model_source = "local_checkpoint_directory" if model_is_local else "huggingface_hub"
    if int(index_metadata.get("sample_rate", 0)) != 48_000:
        parser.error("index manifest does not declare the expected 48 kHz audio rate")

    clip_ids = list(dict.fromkeys(record["clip_id"] for record in manifest["records"]))
    clip_ids.sort(key=str.casefold)
    if args.max_clips is not None:
        clip_ids = clip_ids[: args.max_clips]
    if not clip_ids:
        parser.error("index contains no clips")
    selected = set(clip_ids)

    rows = read_captions(args.captions_csv)
    caption_rows = [row for row in rows if row["file_name"] in selected]
    missing_caption_ids = selected - {row["file_name"] for row in caption_rows}
    if missing_caption_ids:
        sample = sorted(missing_caption_ids)[:5]
        parser.error(f"indexed clips have no matching caption rows, for example: {sample}")

    caption_fields = sorted(
        (field for field in caption_rows[0] if field.startswith("caption_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    queries = [
        {"clip_id": row["file_name"], "caption": row[field], "caption_field": field}
        for row in caption_rows
        for field in caption_fields
        if row[field]
    ]
    if not queries:
        parser.error("selected caption rows contain no non-empty captions")

    keep_rows = [
        i for i, record in enumerate(manifest["records"]) if record["clip_id"] in selected
    ]
    window_vectors = audio_vectors[keep_rows]
    window_clip_ids = [manifest["records"][i]["clip_id"] for i in keep_rows]

    model_start = perf_counter()
    encoder = ClapEncoder.from_pretrained(
        model_name=model_name, device=args.device, revision=revision
    )
    model_load_seconds = perf_counter() - model_start

    rankings: list[list[str]] = []
    predictions: list[dict[str, Any]] = []
    text_encoding_seconds = 0.0
    search_seconds = 0.0
    for offset in range(0, len(queries), args.batch_size):
        batch = queries[offset : offset + args.batch_size]
        start = perf_counter()
        text_vectors = encoder.encode_text_batch([item["caption"] for item in batch])
        text_encoding_seconds += perf_counter() - start
        start = perf_counter()
        batch_results = search_clips_batch(
            text_vectors,
            window_vectors,
            window_clip_ids,
            aggregation=args.aggregation,
        )
        search_seconds += perf_counter() - start
        for query, results in zip(batch, batch_results, strict=True):
            ranking = [result.identifier for result in results]
            rankings.append(ranking)
            predictions.append(
                {
                    "query_clip_id": query["clip_id"],
                    "caption_field": query["caption_field"],
                    "caption": query["caption"],
                    "relevant_clip_ids": [query["clip_id"]],
                    "top_results": [
                        {
                            "clip_id": result.identifier,
                            "similarity": result.similarity,
                        }
                        for result in results[:10]
                    ],
                }
            )

    relevant = [{query["clip_id"]} for query in queries]
    metrics = {
        f"recall_at_{k}": recall_at_k(rankings, relevant, k)
        for k in args.k_values
    }
    metrics["map"] = mean_average_precision(rankings, relevant)
    result = {
        "purpose": "exploratory_clotho_caption_to_audio_retrieval",
        "split": args.split,
        "is_subset": args.max_clips is not None or bool(index_metadata.get("is_subset")),
        "aggregation": args.aggregation,
        "query_count": len(queries),
        "clip_count": len(clip_ids),
        "window_count": len(keep_rows),
        "model_id": model_id,
        "model_source": model_source,
        "model_revision": revision,
        "metrics": metrics,
        "timing_seconds": {
            "model_load": round(model_load_seconds, 3),
            "text_embedding": round(text_encoding_seconds, 3),
            "search": round(search_seconds, 3),
            "total_query": round(text_encoding_seconds + search_seconds, 3),
        },
        "relevance_note": (
            "Each caption's source clip is a known positive. Other semantically relevant "
            "clips may be unannotated positives."
        ),
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "predictions"}, indent=2))
    print(f"saved predictions: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
