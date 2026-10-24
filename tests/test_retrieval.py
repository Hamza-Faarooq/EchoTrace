from __future__ import annotations

import unittest

import numpy as np

from src.retrieval.search import cosine_scores, search


class RetrievalTests(unittest.TestCase):
    def test_cosine_search_is_stable_for_ties(self) -> None:
        query = np.array([1.0, 0.0])
        vectors = np.array([[0.0, 1.0], [3.0, 0.0], [1.0, 0.0]])
        self.assertTrue(np.allclose(cosine_scores(query, vectors), [0.0, 1.0, 1.0]))
        results = search(query, vectors, ["first", "second", "third"], top_k=2)
        self.assertEqual(
            [(item.identifier, item.similarity) for item in results],
            [("second", 1.0), ("third", 1.0)],
        )

    def test_cosine_search_rejects_bad_dimensions_and_zero_norm(self) -> None:
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            cosine_scores(np.ones(2), np.ones((1, 3)))
        with self.assertRaisesRegex(ValueError, "zero norm"):
            cosine_scores(np.zeros(2), np.ones((1, 2)))
