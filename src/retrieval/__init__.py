"""Vector retrieval and temporal result aggregation."""

from .search import (
    SearchResult,
    aggregate_scores_by_clip,
    cosine_scores,
    search,
    search_clips,
    search_clips_batch,
)

__all__ = [
    "SearchResult",
    "aggregate_scores_by_clip",
    "cosine_scores",
    "search",
    "search_clips",
    "search_clips_batch",
]
