# EchoTrace

**Natural-language semantic search for environmental audio, with timestamped results.**

EchoTrace explores retrieval of relevant environmental audio from text queries using a pretrained contrastive language-audio model (CLAP), and studies the effects of temporal windowing and aggregation.

## Repository layout

```text
echotrace/
├── src/
│   ├── audio/          # Load, validate, resample to 48 kHz mono, timestamp windows
│   ├── embeddings/     # CLAP wrapper, normalized embeddings, .npy + manifest.json cache
│   ├── retrieval/      # NumPy cosine top-k, aggregation, merge rules
│   └── evaluation/     # Recall@k, mAP, temporal IoU/F1, latency/memory logs
├── app/                # Streamlit UI only; model logic remains in src/
├── scripts/            # Dataset/index and evaluation entry points
├── configs/
├── tests/
└── docs/
```

## Setup

Python 3.10+:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[test]"
pytest
streamlit run app/streamlit_app.py
```

This first repository scaffold defines module boundaries and starter interfaces. CLAP inference, indexing, temporal retrieval, and full evaluation are not implemented yet; the CLI scripts exit with an explicit message until those steps are built.

## Dataset

Use [Clotho v2.1 on Zenodo](https://zenodo.org/records/4783391) for clip-level audio/text retrieval. Read the dataset terms and preserve its official splits. Do not commit raw audio or generated embeddings. Clotho captions do not provide precise event timestamps; temporal IoU/F1 requires a separate annotated set or a dataset with interval labels.

## Model

The initial embedding backbone is [CLAP](https://huggingface.co/docs/transformers/model_doc/clap), which maps audio and natural-language descriptions into a shared embedding space. Record the exact model identifier, library versions, experiment settings, and real results. Do not invent benchmark numbers.

## Evaluation plan

- Clip retrieval: Recall@1/5/10 and mean average precision (mAP).
- Temporal localization: event-level precision/recall/F1 and temporal IoU only with suitable interval annotations.
- Engineering: index-build time, query latency, and peak memory.
- Compare a CLAP cosine baseline with temporal-windowing/aggregation ablations. Tune on validation data; keep test data for final reporting.

## Reproducibility and data hygiene

Track configs, seeds, dependency versions, and machine details with each experiment. Keep raw audio, large embedding arrays, and model weights out of Git. Dataset and model licenses should be tracked separately. Choose and add a code license before public release.
