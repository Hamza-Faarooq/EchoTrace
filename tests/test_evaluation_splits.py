from __future__ import annotations

import unittest

from src.evaluation.splits import known_source_disjoint_subset


class SourceOverlapFilterTests(unittest.TestCase):
    def test_excludes_known_overlaps_and_keeps_unknown_ids_auditable(self) -> None:
        current = [
            {"file_name": "overlap.wav", "sound_id": "123"},
            {"file_name": "clean.wav", "sound_id": "456"},
            {"file_name": "unknown.wav", "sound_id": "Not Found"},
            {"file_name": "empty.wav", "sound_id": ""},
        ]
        earlier = [
            {"file_name": "dev.wav", "sound_id": "123"},
            {"file_name": "val.wav", "sound_id": "789"},
            {"file_name": "no-source.wav", "sound_id": "not found"},
        ]

        result = known_source_disjoint_subset(current, earlier)

        self.assertEqual(
            result.eligible_clip_ids,
            frozenset({"clean.wav", "unknown.wav", "empty.wav"}),
        )
        self.assertEqual(result.excluded_clip_ids, frozenset({"overlap.wav"}))
        self.assertEqual(
            result.unknown_source_clip_ids,
            frozenset({"unknown.wav", "empty.wav"}),
        )

    def test_requires_metadata_schema_and_unique_clip_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "file_name and sound_id"):
            known_source_disjoint_subset([{"file_name": "a.wav"}], [])
        with self.assertRaisesRegex(ValueError, "duplicate file_name"):
            known_source_disjoint_subset(
                [
                    {"file_name": "a.wav", "sound_id": "1"},
                    {"file_name": "a.wav", "sound_id": "2"},
                ],
                [],
            )

    def test_requires_nonempty_earlier_split_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "metadata contains no rows"):
            known_source_disjoint_subset(
                [{"file_name": "eval.wav", "sound_id": "1"}],
                [],
            )
