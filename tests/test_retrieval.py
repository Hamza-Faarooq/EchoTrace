from __future__ import annotations

import unittest

import numpy as np

from src.retrieval.search import cosine_scores, search, search_clips_batch


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

    def test_clip_aggregation_supports_max_and_mean(self) -> None:
        queries = np.array([[1.0, 0.0]], dtype=np.float32)
        windows = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]])
        clip_ids = ["a.wav", "a.wav", "b.wav"]
        max_ranked = search_clips_batch(queries, windows, clip_ids, aggregation="max")[0]
        mean_ranked = search_clips_batch(queries, windows, clip_ids, aggregation="mean")[0]
        self.assertEqual([item.identifier for item in max_ranked], ["a.wav", "b.wav"])
        self.assertEqual([item.identifier for item in mean_ranked], ["b.wav", "a.wav"])
