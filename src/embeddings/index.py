"""Local NumPy embedding-index persistence with a JSON metadata manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


def save_index(
    index_dir: str | Path,
    embeddings: np.ndarray,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> Path:
    """Write an embedding matrix and its row-aligned metadata manifest."""
    output = Path(index_dir)
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty 2-D matrix")
    if len(records) != vectors.shape[0]:
        raise ValueError("record count must match embedding row count")
    if not np.isfinite(vectors).all():
        raise ValueError("embeddings must contain only finite values")
    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("embeddings must not contain zero-norm rows")
    if any(not record.get("clip_id") for record in records):
        raise ValueError("every record must have a clip_id")

    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "num_windows": int(vectors.shape[0]),
        "embedding_dim": int(vectors.shape[1]),
        "metadata": metadata,
        "records": records,
    }
    # Validate JSON serialization before replacing existing index files.
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    np.save(output / "embeddings.npy", vectors, allow_pickle=False)
    (output / "manifest.json").write_text(manifest_text, encoding="utf-8")
    return output


def load_index(index_dir: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load and validate an EchoTrace embedding matrix and manifest."""
    root = Path(index_dir)
    manifest_path = root / "manifest.json"
    vectors_path = root / "embeddings.npy"
    if not manifest_path.is_file() or not vectors_path.is_file():
        raise FileNotFoundError(
            f"Index needs both {vectors_path.name} and {manifest_path.name}: {root}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported index schema version: {manifest.get('schema_version')}")
    vectors = np.load(vectors_path, allow_pickle=False)
    if vectors.ndim != 2:
        raise ValueError("index embeddings must be a 2-D matrix")
    if vectors.shape[0] != manifest.get("num_windows"):
        raise ValueError("index manifest row count does not match embeddings")
    if vectors.shape[1] != manifest.get("embedding_dim"):
        raise ValueError("index manifest dimension does not match embeddings")
    if len(manifest.get("records", [])) != vectors.shape[0]:
        raise ValueError("index manifest record count does not match embeddings")
    if not np.isfinite(vectors).all():
        raise ValueError("index embeddings contain NaN or infinite values")
    return np.asarray(vectors, dtype=np.float32), manifest
