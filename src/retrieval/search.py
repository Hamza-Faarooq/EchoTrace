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
