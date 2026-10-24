# EchoTrace

**Natural-language semantic search for environmental audio, with timestamped results.**

EchoTrace explores retrieval of relevant environmental audio from text queries using a pretrained contrastive language-audio model (CLAP), and studies temporal windowing and aggregation.

## Current status

The Day 1 research and baseline code is in place. It includes audio decoding, mono downmixing, explicit 48 kHz resampling, deterministic overlapping windows with source timestamps, a pinned CLAP wrapper, NumPy cosine search, and clip-level Recall@K/mAP functions. Window, resampling-logic, ranking, metric, and encoder-guard unit tests pass. The loader test uses a mocked decoder because soundfile was not installed in the local environment; real-file decoding still needs an integration check after dependencies and sample audio are available.

The CPU smoke check completed with 512-dimensional audio and text vectors. Its cosine similarity of 0.500575 compares one synthetic tone with one text query; it is a smoke result, not a retrieval benchmark. The model score is a similarity value, not a probability. No benchmark, deployment, index builder, Streamlit search flow, or validated event-localization result is claimed yet.

The first-party repository scaffold already defines module boundaries; see the [architecture notes](docs/architecture.md). More detail on model behavior, data risks, and the first experiment is in [implementation notes](docs/implementation-notes.md).

## Repository layout

- src/audio: decoding, mono conversion, 48 kHz resampling, deterministic timestamped windows.
- src/embeddings: reusable CLAP audio/text model wrapper.
- src/retrieval: NumPy cosine scoring and stable top-k ranking.
- src/evaluation: clip-level Recall@K and mean average precision.
- app: Streamlit interface; it will call the library rather than own ML logic.
- scripts: dataset inspection, model download, smoke check, indexing and evaluation entry points.
- configs: preprocessing, retrieval, and evaluation settings.
- docs: architecture, dataset audit, and implementation decisions.
- tests: audio, encoder guard, retrieval, and metric checks.
- results: small checked-in smoke metadata only; no dataset or model weights.

## Setup

Use Python 3.10–3.12. Install a PyTorch build that matches your CPU or CUDA machine from the official [PyTorch installation selector](https://pytorch.org/get-started/locally/), then install EchoTrace with its test extra:

    python -m venv .venv
    python -m pip install --upgrade pip
    python -m pip install -e ".[test]"

Run tests with pytest. The code does not load the model at import time. To download the pinned model to the ignored local cache and run the CPU smoke check:

    python scripts/download_model.py
    $env:PYTHONPATH = "."
    python scripts/clap_smoke.py --device cpu --model ".cache/models/clap-htsat-unfused"

The checkpoint is about 618 MB, including roughly 615 MB of weights. To use the normal Hugging Face cache, omit the model path. See [the CLAP implementation notes](docs/implementation-notes.md) for the exact checkpoint revision and preprocessing behavior.

## Dataset and licensing

The primary research dataset is [Clotho v2.1 on Zenodo](https://zenodo.org/records/4783391). Review its terms, download the official files yourself, and extract audio under data/raw/clotho with development, validation, and evaluation directories. The project does not bypass acceptance steps. Inspect actual CSV schemas and audio files with scripts/inspect_clotho.py; use its csv-only option before downloading the large audio archives.

Raw audio, caption/metadata CSVs, embeddings, and model weights are ignored by Git. Clotho captions are licensed for experimental, non-commercial use and their accompanying terms prohibit providing services with them. Audio and metadata also have per-item Freesound licensing. Use Clotho for offline research only unless you obtain separate permission. The hosted demo needs self-recorded or individually cleared examples and must not serve Clotho captions or a Clotho-derived index. See the [observed data audit](docs/dataset-audit.md).

## Evaluation protocol

- Use development for implementation and exploratory analysis.
- Choose settings on validation, then freeze them before official evaluation.
- Keep the official evaluation result intact. The observed metadata contains repeated Freesound IDs across splits, so also report the documented known-source-disjoint sensitivity subset; unknown IDs remain unaudited.
- For caption-to-audio retrieval, count the source clip as a known positive and report Recall@1/5/10 and mAP. Other semantically relevant clips may be unannotated positives, so interpret these as incomplete relevance labels.
- Keep clip retrieval separate from event localization. Clotho has clip captions, not verified event boundaries. Timestamp claims need a separately documented and manually checked interval set.
- Record configuration, predictions, software versions, and measured runtime for each experiment. Do not report unrun values.

## Day 1 checklist

- [x] Inspect the existing repository and local CPU environment.
- [x] Verify the CLAP checkpoint, Transformers API, 48 kHz sample rate, and 10-second input window limit.
- [x] Inspect Clotho CSV fields, split counts, checksums, source overlap, and license conditions.
- [x] Run an audio-text CPU smoke check with a pinned model revision.
- [x] Implement the first preprocessing, embedding, cosine-search, and metric modules with unit coverage.
- [ ] Build a reproducible local index and evaluate retrieval on a development subset.

The largest near-term risks are Clotho's service restrictions, model cold-start memory on free hosting, source overlap and incomplete relevance labels, and CPU indexing time. The next smallest milestone is to build a local development index and save predictions and metrics before adding the UI.

## References

- [Transformers CLAP documentation](https://huggingface.co/docs/transformers/model_doc/clap)
- [Clotho v2.1 dataset record](https://zenodo.org/records/4783391)
- [Clotho dataset paper](https://arxiv.org/abs/1910.09387)

Choose and add a separate code license before public release; dataset and model terms remain independent.
