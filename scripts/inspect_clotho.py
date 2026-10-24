"""Inspect a locally downloaded Clotho layout without assuming CSV column names."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

SPLITS = ("development", "validation", "evaluation")


def inspect_csv(
    path: Path,
) -> tuple[int, list[str], str, set[str] | None, set[str] | None, int, int, int]:
    """Return row count, exact header, decoder, and unique filename IDs if present."""
    for encoding in ("utf-8-sig", "iso-8859-1"):
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                reader = csv.DictReader(file)
                columns = list(reader.fieldnames or [])
                rows = list(reader)
            identifiers = None
            duplicate_filenames = 0
            if "file_name" in columns:
                filenames = [row["file_name"] for row in rows if row.get("file_name")]
                identifiers = set(filenames)
                duplicate_filenames = len(filenames) - len(identifiers)
            source_ids = None
            unknown_sources = 0
            duplicate_sources = 0
            if "sound_id" in columns:
                known_sources = [
                    row["sound_id"].strip()
                    for row in rows
                    if row.get("sound_id") and row["sound_id"].strip().lower() != "not found"
                ]
                source_ids = set(known_sources)
                unknown_sources = len(rows) - len(known_sources)
                duplicate_sources = len(known_sources) - len(source_ids)
            return (
                len(rows), columns, encoding, identifiers, source_ids, unknown_sources,
                duplicate_filenames, duplicate_sources,
            )
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Could not decode {path} as UTF-8 or ISO-8859-1.")


def audio_file_count(path: Path) -> int:
    """Count files recursively; do not assume a particular audio extension."""
    if not path.is_dir():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def inspect_split(
    root: Path, split: str, csv_only: bool = False
) -> tuple[bool, list[str], set[str] | None, set[str] | None]:
    """Print observed CSV schema/counts and the number of extracted files."""
    ok = True
    missing: list[str] = []
    tables: dict[
        str,
        tuple[int, list[str], str, set[str] | None, set[str] | None, int, int, int],
    ] = {}
    print(f"[{split}]")
    for kind in ("captions", "metadata"):
        path = root / f"clotho_{kind}_{split}.csv"
        if not path.is_file():
            print(f"  {kind}: MISSING ({path})")
            missing.append(str(path))
            ok = False
            continue
        (
            row_count, columns, encoding, identifiers, source_ids, unknown_sources,
            duplicate_filenames, duplicate_sources,
        ) = inspect_csv(path)
        tables[kind] = (
            row_count, columns, encoding, identifiers, source_ids, unknown_sources,
            duplicate_filenames, duplicate_sources,
        )
        print(f"  {kind}: rows={row_count}; encoding={encoding}; columns={columns}")
        if source_ids is not None:
            print(
                f"  source IDs: unique_known={len(source_ids)}; "
                f"missing_or_unavailable={unknown_sources}; repeated_records={duplicate_sources}"
            )
        if duplicate_filenames:
            print(f"  repeated filenames within split: {duplicate_filenames}")

    empty_table = (0, [], "", None, None, 0, 0, 0)
    caption_ids = tables.get("captions", empty_table)[3]
    metadata_ids = tables.get("metadata", empty_table)[3]
    metadata_source_ids = tables.get("metadata", empty_table)[4]
    if caption_ids is not None and metadata_ids is not None:
        print(
            "  filename join: "
            f"caption_ids={len(caption_ids)}; metadata_ids={len(metadata_ids)}; "
            f"same_ids={caption_ids == metadata_ids}"
        )
        if caption_ids != metadata_ids:
            ok = False

    audio_dir = root / split
    if csv_only:
        print("  extracted files: skipped (--csv-only)")
    elif not audio_dir.is_dir():
        print(f"  extracted files: MISSING ({audio_dir})")
        missing.append(str(audio_dir))
        ok = False
    else:
        print(f"  extracted files: {audio_file_count(audio_dir)} under {audio_dir}")
    return ok, missing, caption_ids, metadata_source_ids


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/clotho"),
        help="Directory containing split folders and Clotho CSVs (default: data/raw/clotho).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Official split names to inspect (default: all three).",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Inspect CSV schemas and split IDs without requiring extracted audio directories.",
    )
    parser.add_argument(
        "--strict-split-disjoint",
        action="store_true",
        help="Return failure if filename or known Freesound source IDs repeat across selected splits.",
    )
    args = parser.parse_args(argv)

    all_ok = True
    seen_ids: set[str] = set()
    seen_source_ids: set[str] = set()
    for split in args.splits:
        split_ok, _, identifiers, source_ids = inspect_split(args.data_dir, split, csv_only=args.csv_only)
        overlap: set[str] = set()
        source_overlap: set[str] = set()
        if identifiers is not None:
            overlap = seen_ids.intersection(identifiers)
            if overlap:
                print(f"  repeated filename warning: {len(overlap)} IDs also appeared in an earlier selected split")
            seen_ids.update(identifiers)
        if source_ids is not None:
            source_overlap = seen_source_ids.intersection(source_ids)
            if source_overlap:
                print(
                    f"  source overlap warning: {len(source_overlap)} Freesound IDs also appeared "
                    "in an earlier selected split"
                )
            seen_source_ids.update(source_ids)
        if args.strict_split_disjoint and (source_overlap or overlap):
            split_ok = False
        all_ok = all_ok and split_ok
    if not all_ok:
        print("\nInspection failed: check missing files, caption/metadata joins, or strict split-disjoint warnings.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
