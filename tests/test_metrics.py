from __future__ import annotations

import unittest

from src.evaluation.metrics import mean_average_precision, recall_at_k


class RetrievalMetricTests(unittest.TestCase):
    def test_recall_at_k_and_mean_average_precision(self) -> None:
        rankings = [["a", "b", "c"], ["x", "y"]]
        relevant = [{"a", "c"}, {"y"}]
        self.assertAlmostEqual(recall_at_k(rankings, relevant, k=1), 0.25)
        self.assertAlmostEqual(recall_at_k(rankings, relevant, k=2), 0.75)
        self.assertAlmostEqual(mean_average_precision(rankings, relevant), 2 / 3)

    def test_metrics_require_nonempty_relevance(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            recall_at_k([["a"]], [set()], k=1)
