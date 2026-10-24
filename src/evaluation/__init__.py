"""Clip-level retrieval and temporal-localization evaluation."""

from .metrics import mean_average_precision, recall_at_k
from .splits import KnownSourceSubset, known_source_disjoint_subset

__all__ = [
    "KnownSourceSubset",
    "mean_average_precision",
    "known_source_disjoint_subset",
    "recall_at_k",
]
