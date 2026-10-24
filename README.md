# EchoTrace

**Natural-language semantic search for environmental audio, with timestamped results.**

EchoTrace explores retrieval of relevant environmental audio from text queries using a pretrained contrastive language-audio model (CLAP), and studies temporal windowing and aggregation.

## Current status

The Day 1 foundation and next indexing/evaluation milestone are implemented. The code includes audio decoding, mono downmixing, explicit 48 kHz resampling, deterministic timestamped windows, batched CLAP embeddings, a local NumPy index, max/mean window aggregation, and a split-aware caption retrieval evaluator. Unit tests cover the core logic. The loader test uses a mocked decoder because soundfile was not installed locally; real-file decoding still needs an integration check.

The pinned CLAP checkpoint produced finite 512-dimensional audio and text batches on CPU. A small synthetic-index run exercised the evaluation command end to end, but its scores are not a dataset result. The separate smoke report's 0.500575 cosine compares one synthetic tone with one text query; it is not a retrieval benchmark. Scores are similarities, not probabilities. No Clotho benchmark, deployment, Streamlit search flow, or validated event-localization result is claimed yet.

The first-party repository scaffold already defines module boundaries; see the [architecture notes](docs/architecture.md). More detail on model behavior, data risks, and the first experiment is in [implementation notes](docs/implementation-notes.md).

## Repository layout

- src/audio: decoding, mono conversion, 48 kHz resampling, deterministic timestamped windows.
- src/embeddings: reusable CLAP audio/text model wrapper.
- src/embeddings/index.py: local vectors plus a row-aligned JSON manifest.
- src/retrieval: NumPy cosine scoring, stable top-k ranking, and max/mean window aggregation.
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

After downloading and extracting a split, build an exploratory 100-clip index and run its caption queries:

    python scripts/build_index.py --audio-dir data/raw/clotho/development --output-dir data/index/development --split development --limit 100 --device cpu --model .cache/models/clap-htsat-unfused
    python scripts/run_eval.py --split development --captions-csv data/raw/clotho/clotho_captions_development.csv --index-dir data/index/development --max-clips 100 --device cpu --model .cache/models/clap-htsat-unfused

Both commands sort file names for a repeatable subset. The evaluator checks that the requested split matches the index, supports max or mean window aggregation, writes metrics and per-caption predictions locally, and does not treat scores as probabilities. Indexes and prediction outputs are ignored by Git.

## Evaluation protocol

- Use development for implementation and exploratory analysis.
- Choose settings on validation, then freeze them before official evaluation.
- Keep the official evaluation result intact. The observed metadata contains repeated Freesound IDs across splits, so also report the documented known-source-disjoint sensitivity subset; unknown IDs remain unaudited.
- For caption-to-audio retrieval, count the source clip as a known positive and report Recall@1/5/10 and mAP. Other semantically relevant clips may be unannotated positives, so interpret these as incomplete relevance labels.
- Keep clip retrieval separate from event localization. Clotho has clip captions, not verified event boundaries. Timestamp claims need a separately documented and manually checked interval set.
- Record configuration, predictions, software versions, and measured runtime for each experiment. Do not report unrun values.

## Milestone checklist

- [x] Inspect the existing repository and local CPU environment.
- [x] Verify the CLAP checkpoint, Transformers API, 48 kHz sample rate, and 10-second input window limit.
- [x] Inspect Clotho CSV fields, split counts, checksums, source overlap, and license conditions.
- [x] Run an audio-text CPU smoke check with a pinned model revision.
- [x] Implement batched preprocessing, embeddings, index persistence, cosine retrieval, aggregation, and clip metrics with unit coverage.
- [x] Exercise index loading and the evaluation command using a small synthetic index; this is only a wiring check.
- [ ] Verify decoding with a real audio file, then benchmark a Clotho development subset.

The largest near-term risks are Clotho's service restrictions, model cold-start memory on free hosting, source overlap and incomplete relevance labels, and CPU indexing time. The next step is a real development-split run after the audio files and pinned decoder dependency are available, then a validation-only aggregation ablation.

## References

- [Transformers CLAP documentation](https://huggingface.co/docs/transformers/model_doc/clap)
- [Clotho v2.1 dataset record](https://zenodo.org/records/4783391)
- [Clotho dataset paper](https://arxiv.org/abs/1910.09387)

Choose and add a separate code license before public release; dataset and model terms remain independent.
