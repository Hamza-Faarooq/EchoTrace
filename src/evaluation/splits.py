"""Split-level source-overlap filters for Clotho sensitivity analyses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

UNKNOWN_SOURCE_VALUES = {"", "not found", "unknown", "none", "nan", "n/a"}


@dataclass(frozen=True)
class KnownSourceSubset:
    """Clip IDs retained/excluded by the known-source sensitivity filter."""

    eligible_clip_ids: frozenset[str]
    excluded_clip_ids: frozenset[str]
    unknown_source_clip_ids: frozenset[str]


def known_source_disjoint_subset(
    current_metadata: Sequence[Mapping[str, str]],
    excluded_split_metadata: Sequence[Mapping[str, str]],
) -> KnownSourceSubset:
    """Exclude current clips whose known sound_id appears in earlier splits.

    Rows with unavailable source IDs are retained and returned separately so
    callers can report that they remain unaudited.
    """
    current_ids = _known_source_ids(current_metadata, require_rows=True)
    earlier_ids = _known_source_ids(excluded_split_metadata, require_rows=True)
    earlier_sources = {
        source_id
        for source_id in earlier_ids.values()
        if source_id is not None
    }

    eligible: set[str] = set()
    excluded: set[str] = set()
    unknown: set[str] = set()
    for clip_id, source_id in current_ids.items():
        if source_id is None:
            eligible.add(clip_id)
            unknown.add(clip_id)
        elif source_id in earlier_sources:
            excluded.add(clip_id)
        else:
            eligible.add(clip_id)
    return KnownSourceSubset(
        eligible_clip_ids=frozenset(eligible),
        excluded_clip_ids=frozenset(excluded),
        unknown_source_clip_ids=frozenset(unknown),
    )


def _known_source_ids(
    rows: Sequence[Mapping[str, str]], require_rows: bool
) -> dict[str, str | None]:
    if not rows:
        if require_rows:
            raise ValueError("current split metadata contains no rows")
        return {}
    if any("file_name" not in row or "sound_id" not in row for row in rows):
        raise ValueError("metadata rows must contain file_name and sound_id")

    result: dict[str, str | None] = {}
    for row in rows:
        clip_id = (row.get("file_name") or "").strip()
        if not clip_id:
            continue
        if clip_id in result:
            raise ValueError(f"metadata contains duplicate file_name: {clip_id}")
        raw_source_id = (row.get("sound_id") or "").strip()
        normalized = raw_source_id.casefold()
        source_id = None if normalized in UNKNOWN_SOURCE_VALUES else raw_source_id
        result[clip_id] = source_id
    if require_rows and not result:
        raise ValueError("current split metadata contains no valid file_name values")
    return result
