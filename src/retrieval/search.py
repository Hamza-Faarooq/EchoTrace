"""Small-index cosine scoring and stable top-k retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    identifier: str
    similarity: float


def cosine_scores(query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Return cosine similarity between one query and each embedding row."""
    q = np.asarray(query, dtype=np.float32)
    matrix = np.asarray(embeddings, dtype=np.float32)
    if q.ndim != 1:
        raise ValueError(f"query must be 1-D, got shape {q.shape}.")
    if matrix.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {matrix.shape}.")
    if matrix.shape[1] != q.shape[0]:
        raise ValueError(
            f"Embedding dimension mismatch: query has {q.shape[0]}, matrix has {matrix.shape[1]}."
        )
    if not np.isfinite(q).all() or not np.isfinite(matrix).all():
        raise ValueError("Query and embeddings must contain only finite values.")
    q_norm = float(np.linalg.norm(q))
    row_norms = np.linalg.norm(matrix, axis=1)
    if q_norm == 0:
        raise ValueError("query embedding has zero norm.")
    if np.any(row_norms == 0):
        raise ValueError("one or more indexed embeddings have zero norm.")
    return (matrix / row_norms[:, None]) @ (q / q_norm)


def search(
    query: np.ndarray,
    embeddings: np.ndarray,
    identifiers: list[str],
    top_k: int = 5,
) -> list[SearchResult]:
    """Rank indexed clips by raw CLAP cosine similarity.

    Scores are similarities, not calibrated probabilities. Input order breaks
    exact ties so repeated runs return stable rankings.
    """
    scores = cosine_scores(query, embeddings)
    if scores.size != len(identifiers):
        raise ValueError("embeddings and identifiers must have equal lengths.")
    if top_k < 0:
        raise ValueError("top_k cannot be negative.")
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [SearchResult(str(identifiers[i]), float(scores[i])) for i in order]


def aggregate_scores_by_clip(
    window_scores: np.ndarray,
    clip_ids: list[str],
    aggregation: str = "max",
    top_k: int | None = None,
) -> list[SearchResult]:
    """Aggregate window scores per clip, then rank clips with stable tie breaks."""
    scores = np.asarray(window_scores, dtype=np.float32)
    if scores.ndim != 1 or scores.size != len(clip_ids):
        raise ValueError("window_scores and clip_ids must be equal-length 1-D inputs")
    if not np.isfinite(scores).all():
        raise ValueError("window scores must contain only finite values")
    if aggregation not in {"max", "mean"}:
        raise ValueError("aggregation must be 'max' or 'mean'")
    if top_k is not None and top_k < 0:
        raise ValueError("top_k cannot be negative")
    if any(not clip_id for clip_id in clip_ids):
        raise ValueError("clip IDs must be non-empty")

    grouped: dict[str, list[float]] = {}
    for clip_id, score in zip(clip_ids, scores, strict=True):
        grouped.setdefault(clip_id, []).append(float(score))
    aggregate = [
        (clip_id, max(values) if aggregation == "max" else sum(values) / len(values))
        for clip_id, values in grouped.items()
    ]
    aggregate.sort(key=lambda item: -item[1])
    if top_k is not None:
        aggregate = aggregate[:top_k]
    return [SearchResult(clip_id, float(score)) for clip_id, score in aggregate]


def search_clips(
    query: np.ndarray,
    window_embeddings: np.ndarray,
    clip_ids: list[str],
    top_k: int = 5,
    aggregation: str = "max",
) -> list[SearchResult]:
    """Search window embeddings and return one aggregated score per clip."""
    if top_k < 0:
        raise ValueError("top_k cannot be negative")
    return search_clips_batch(
        np.asarray(query, dtype=np.float32)[None, :],
        window_embeddings,
        clip_ids,
        aggregation,
    )[0][:top_k]


def search_clips_batch(
    queries: np.ndarray,
    window_embeddings: np.ndarray,
    clip_ids: list[str],
    aggregation: str = "max",
) -> list[list[SearchResult]]:
    """Vectorized query-batch search with one aggregated score per clip."""
    query_matrix = np.asarray(queries, dtype=np.float32)
    vectors = np.asarray(window_embeddings, dtype=np.float32)
    if query_matrix.ndim != 2 or vectors.ndim != 2:
        raise ValueError("queries and window_embeddings must be 2-D matrices")
    if query_matrix.shape[0] == 0 or query_matrix.shape[1] == 0:
        raise ValueError("query batch must not be empty")
    if vectors.shape[0] != len(clip_ids) or vectors.shape[0] == 0:
        raise ValueError("window embedding rows must match non-empty clip_ids")
    if vectors.shape[1] != query_matrix.shape[1]:
        raise ValueError("query and audio embedding dimensions must match")
    if aggregation not in {"max", "mean"}:
        raise ValueError("aggregation must be 'max' or 'mean'")
    if not np.isfinite(query_matrix).all() or not np.isfinite(vectors).all():
        raise ValueError("queries and window embeddings must be finite")
    query_norms = np.linalg.norm(query_matrix, axis=1)
    vector_norms = np.linalg.norm(vectors, axis=1)
    if np.any(query_norms == 0) or np.any(vector_norms == 0):
        raise ValueError("query and window embeddings must have non-zero norms")

    window_scores = (query_matrix / query_norms[:, None]) @ (
        vectors / vector_norms[:, None]
    ).T
    unique_ids = list(dict.fromkeys(clip_ids))
    grouped_rows: dict[str, list[int]] = {clip_id: [] for clip_id in unique_ids}
    for row_index, clip_id in enumerate(clip_ids):
        grouped_rows[clip_id].append(row_index)
    rows_by_clip = {
        clip_id: np.asarray(indices, dtype=np.int64)
        for clip_id, indices in grouped_rows.items()
    }
    clip_scores = np.column_stack(
        [
            (
                window_scores[:, indices].max(axis=1)
                if aggregation == "max"
                else window_scores[:, indices].mean(axis=1)
            )
            for indices in rows_by_clip.values()
        ]
    )
    ranked: list[list[SearchResult]] = []
    for scores in clip_scores:
        order = np.argsort(-scores, kind="stable")
        ranked.append(
            [SearchResult(unique_ids[i], float(scores[i])) for i in order]
        )
    return ranked
