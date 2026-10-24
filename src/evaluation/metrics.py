"""Small, dependency-free clip-retrieval metrics."""

from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(
    rankings: Sequence[Sequence[str]], relevant: Sequence[set[str]], k: int
) -> float:
    """Macro-average fraction of known relevant clips found in each top-k list."""
    _validate_inputs(rankings, relevant)
    if k < 1:
        raise ValueError("k must be at least 1")
    recalls = []
    for ranking, positives in zip(rankings, relevant, strict=True):
        retrieved = set(ranking[:k])
        recalls.append(len(retrieved & positives) / len(positives))
    return sum(recalls) / len(recalls)


def mean_average_precision(
    rankings: Sequence[Sequence[str]], relevant: Sequence[set[str]]
) -> float:
    """Compute mean AP over ranked clip IDs, counting each ID at most once."""
    _validate_inputs(rankings, relevant)
    average_precisions = []
    for ranking, positives in zip(rankings, relevant, strict=True):
        seen: set[str] = set()
        hits = 0
        precision_sum = 0.0
        for rank, identifier in enumerate(ranking, start=1):
            if identifier in seen:
                continue
            seen.add(identifier)
            if identifier in positives:
                hits += 1
                precision_sum += hits / rank
        average_precisions.append(precision_sum / len(positives))
    return sum(average_precisions) / len(average_precisions)


def _validate_inputs(
    rankings: Sequence[Sequence[str]], relevant: Sequence[set[str]]
) -> None:
    if len(rankings) == 0 or len(rankings) != len(relevant):
        raise ValueError("rankings and relevant must have the same non-zero query count")
    if any(not positives for positives in relevant):
        raise ValueError("each query must have at least one known relevant clip")
