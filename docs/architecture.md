# EchoTrace architecture

- `src/audio/`: load and validate audio, resample to 48 kHz mono, create timestamped windows.
- `src/embeddings/`: CLAP model lifecycle, audio/text vectors, L2 normalization, disk cache (`.npy` + `manifest.json`). No model loading at import time.
- `src/retrieval/`: NumPy cosine top-k, temporal aggregation, adjacent-hit merging.
- `src/evaluation/`: Recall@k, mAP, temporal IoU/F1 where interval labels exist, and latency/memory logs.
- `app/`: Streamlit presentation layer only; calls APIs in `src/`.
- `scripts/`: dataset/index and evaluation command-line entry points.

## Evaluation integrity

Use official Clotho splits for clip-level retrieval. Clotho captions do not provide precise event boundaries, so they cannot alone validate timestamp localization. Temporal metrics require a separately documented query-to-interval annotation set or a dataset with interval labels.
