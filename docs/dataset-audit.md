# Clotho v2.1 data audit

Audited on 2026-10-24 from the official [Zenodo v2.1 record](https://zenodo.org/records/4783391). Only the six small CSV files and the `LICENSE` text were downloaded. The audio archives (about 7 GB total) have not been downloaded or extracted.

## Observed files

| Split | Caption rows | Metadata rows | Caption fields | Metadata fields | CSV IDs join |
|---|---:|---:|---|---|---|
| development | 3,839 | 3,839 | `file_name`, `caption_1`–`caption_5` | `file_name`, `keywords`, `sound_id`, `sound_link`, `start_end_samples`, `manufacturer`, `license` | yes |
| validation | 1,045 | 1,045 | `file_name`, `caption_1`–`caption_5` | same | yes |
| evaluation | 1,045 | 1,045 | `file_name`, `caption_1`–`caption_5` | same | yes |

The observed total is 5,929 CSV records, each with five captions. The Zenodo record narrative says 6,974 samples, so the narrative total does not match the files distributed in the same record. Use the inspected CSV row counts for this benchmark and check the extracted audio counts before indexing.

Captions decode as UTF-8 with BOM. Development and validation metadata decode as ISO-8859-1; evaluation metadata decodes as UTF-8 with BOM. The inspector supports these observed encodings and prints the selected decoder.

All seven downloaded files match the MD5 digests published on the Zenodo record. The row counts and field names are from those downloaded files, not inferred from other Clotho copies.

## Cross-split source overlap

The caption and metadata `file_name` sets match within each split. The official split CSVs are not fully disjoint by filename or Freesound source ID:

| Pair | Repeated filenames | Shared known `sound_id` values |
|---|---:|---:|
| development / validation | 1 | 2 |
| development / evaluation | 1 | 1 |
| validation / evaluation | 2 | 2 |

Three evaluation records have known Freesound `sound_id` values that also occur in development or validation. Several repeated source IDs have overlapping `start_end_samples` excerpts. One known source ID (`130597`) is also represented twice inside development with overlapping excerpt ranges. Eight development, six validation, and five evaluation metadata rows have an unavailable `sound_id` value, so this audit cannot establish source-level disjointness for those records.

Keep the official evaluation split intact for the headline result, and report a separate sensitivity result that excludes the three evaluation records with known source IDs in development or validation. Call that a *known-source-disjoint subset*; unknown source IDs remain unaudited. Avoid using an evaluation split embedding as part of the index when evaluating validation or development queries.

The helper can display these warnings and can fail strict checks with:

```powershell
python scripts/inspect_clotho.py --data-dir data/raw/clotho --csv-only --strict-split-disjoint
```

Strict mode is expected to fail for the published CSVs because of the observed overlaps. It does not alter or silently drop official examples.

## Licensing

The Zenodo record states that audio and metadata use per-item Freesound licenses, mostly Creative Commons with attribution, and that metadata provides each item's license link. The downloaded captions' `LICENSE` file grants experimental, non-commercial use and expressly prohibits providing services using the work. Therefore, use Clotho captions and audio only for offline research/evaluation under the applicable terms. Do not publish Clotho captions or a Clotho-derived index in the hosted public demo without separate permission. The demo should use separately cleared recordings.

The original [dataset paper](https://arxiv.org/abs/1910.09387) should be cited in research outputs. The [Clotho data-handling repository](https://github.com/audio-captioning/clotho-dataset) describes the split directories produced by its audio archives.
