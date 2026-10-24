# Download Clotho v2.1

Official record: https://zenodo.org/records/4783391

1. Read the record and the included captions LICENSE before using the data.
2. Download the official audio archives and the caption/metadata CSVs from Zenodo. Complete any required acceptance steps on the site yourself; this project does not bypass terms or scrape gated content.
3. Extract the audio into `data/raw/clotho/development/`, `data/raw/clotho/validation/`, and `data/raw/clotho/evaluation/`. Keep the matching CSVs directly in `data/raw/clotho/`.
4. Inspect what was actually downloaded with `python scripts/inspect_clotho.py --data-dir data/raw/clotho`. Use `--csv-only` before the large audio archives are available.
5. Keep the official split labels intact. Do not commit raw audio, captions, metadata, embeddings, or a derived Clotho index.

The observed CSV counts, column names, source overlap, checksums, and license caveats are recorded in `docs/dataset-audit.md`. Clotho captions are for offline experimental, non-commercial use under their accompanying license; do not serve Clotho-derived content in the public demo without separate permission. Check the per-item license metadata before using audio clips.
