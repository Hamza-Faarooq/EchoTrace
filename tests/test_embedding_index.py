from __future__ import annotations

import unittest
from pathlib import Path
from uuid import uuid4

import numpy as np

from src.embeddings.index import load_index, save_index


class EmbeddingIndexTests(unittest.TestCase):
    def make_index_path(self) -> Path:
        root = (Path(__file__).resolve().parents[1] / "work" / "index-tests").resolve()
        root.mkdir(parents=True, exist_ok=True)
        target = (root / f"case-{uuid4().hex}").resolve()
        if target.parent != root:
            raise AssertionError("test index path escaped its workspace folder")
        self.addCleanup(self.cleanup_index, target)
        return target

    @staticmethod
    def cleanup_index(path: Path) -> None:
        if not path.exists():
            return
        for name in ("embeddings.npy", "manifest.json"):
            (path / name).unlink(missing_ok=True)
        path.rmdir()

    def test_round_trip_keeps_rows_aligned_with_manifest(self) -> None:
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        records = [
            {"clip_id": "a.wav", "window_index": 0, "start_seconds": 0.0},
            {"clip_id": "b.wav", "window_index": 0, "start_seconds": 0.0},
        ]
        index_dir = self.make_index_path()
        save_index(index_dir, vectors, records, {"split": "development"})
        restored, manifest = load_index(index_dir)

        self.assertTrue(np.array_equal(restored, vectors))
        self.assertEqual(manifest["records"], records)
        self.assertEqual(manifest["metadata"]["split"], "development")

    def test_rejects_row_mismatch_and_zero_vectors(self) -> None:
        with self.assertRaisesRegex(ValueError, "record count"):
            save_index(self.make_index_path(), np.ones((1, 2)), [], {})
        with self.assertRaisesRegex(ValueError, "zero-norm"):
            save_index(
                self.make_index_path(),
                np.zeros((1, 2)),
                [{"clip_id": "a.wav"}],
                {},
            )
