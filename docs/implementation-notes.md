# Day 1 implementation notes

## CLAP API and preprocessing

- The official Transformers CLAP guide demonstrates `ClapModel.from_pretrained("laion/clap-htsat-unfused")`, tokenization for text, `ClapFeatureExtractor`/`ClapProcessor` for audio, and `get_text_features` / `get_audio_features` under inference mode.
- The checkpoint's processor uses 48,000 Hz audio and a default maximum input duration of 10 seconds. The feature extractor warns about a sample-rate mismatch but does not resample the waveform for us.
- For inputs longer than 10 seconds, the default `fusion` truncation uses three random crops and a downsampled full clip. That behavior can make a repeatable temporal benchmark awkward. EchoTrace will explicitly resample, slice deterministic 10-second windows with configurable overlap, and retain each window's original start/end time. A separate ablation can test the model's fused long-clip behavior.
- The local installed Transformers version is 4.42.4. Its `ClapProcessor` call uses the keyword `audios=...` and takes `sampling_rate=...`; a wrong rate raises an error in this version, but the processor never resamples for us. `ClapModel.get_audio_features` and `get_text_features` both return projected, L2-normalized embeddings. We'll keep an explicit shape/finite-value check and cosine scoring in our code.
- A local preprocessing probe exposed a subtlety: under `truncation="fusion"`, 4.42.4 marks one batch item as `is_longer=True` when a batch contains no long clips (for a batch of one, that item is always the sample). For our deterministic <=10-second slices, the wrapper explicitly requests `truncation="rand_trunc"`; the probe produced one mel input and `is_longer=False`. The wrapper also rejects clips longer than the processor limit so an accidental random crop cannot masquerade as full-clip inference.
- The checkpoint repository is about 618 MB, including about 615 MB of weights. EchoTrace pins the resolved Hub revision `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`. Current runtime has Python 3.12.4, PyTorch 2.13.0+cpu, 16 GiB RAM, no CUDA device, and 71.6 GB free on C:. GPU performance must be measured later on Colab if needed.

## Minimum architecture

1. `audio`: decode, validate, mono-convert, resample, create deterministic windows with timestamp metadata.
2. `embeddings`: one cached CLAP model wrapper for audio/text features; normalize and record model/config identity.
3. `retrieval`: raw CLAP cosine baseline; compare with max/mean/top-window temporal aggregation over the same deterministic windows.
4. `evaluation`: split-aware caption-to-clip Recall@1/5/10 and mAP, validation-only parameter selection, held-out evaluation output, and separate temporal metrics for manually verified intervals.
5. `app`: Streamlit UI calling the same library functions; model loaded once per session/process; audio demo catalog separate from Clotho.

No vector database is needed for a few thousand clips. A NumPy matrix multiplication is simpler and auditable at this scale.

## Data facts and guardrails

- Zenodo v2.1 publishes development, validation, and evaluation archives, captions CSVs, metadata CSVs, and a separate `LICENSE` file. Its headline description says 6,974 audio samples; the version-notes paragraph separately lists 3,840 development and 1,046 validation files but does not give the evaluation count there. The inspector therefore reports actual local row/file counts; confirm every split before benchmarking and preserve the official split labels.
- Metadata holds source file name, keywords, Freesound URL, excerpt sample boundaries, uploader, and per-item license information. Confirm actual CSV headers and source/audio pairing from downloaded files before coding joins.
- The caption license grants experimental/non-commercial use and forbids commercial use, including providing services using the work. Don't publish Clotho captions or a Clotho-derived demo index in the hosted service. Evaluate on Clotho locally/Colab; use separately cleared demo audio.
- The dataset license identifies no precise event boundaries for every described sound. Similarity peaks can be shown as candidate intervals, but must not be presented as validated event annotations.
- The audio archives are intentionally not downloaded in this first milestone. The six official CSVs and license text are local; the actual schema, checksums, and row counts are documented in [the Clotho audit](dataset-audit.md). All caption/metadata filename sets match within each split, but the CSVs contain repeated filenames and Freesound source IDs across splits. Several repeated sources also have overlapping excerpt sample ranges, so evaluation needs a full official result plus a known-source-disjoint sensitivity result. Audio archive contents and file-to-audio existence remain unchecked.

## Largest schedule risks

1. **Licensing the public demo:** Clotho captions cannot be served as a public service under their accompanying terms. Build a small separately licensed demo set.
2. **Model cold-start and hosting memory:** The model is roughly 615 MB on disk before runtime allocations. Test CPU load time/RAM early and keep an alternate precomputed, license-cleared demo path.
3. **Audio preprocessing correctness:** 48 kHz is required; silent downsampling or the processor's random long-audio fusion can corrupt comparisons. Keep explicit sample-rate and timestamp handling.
4. **Weak retrieval labels and no time labels:** Clotho is suitable for caption retrieval, but missing alternative positives limit interpretation, and it cannot alone validate event localization.
5. **Source overlap across official splits:** metadata contains repeated Freesound IDs and overlapping excerpt ranges across development, validation, and evaluation. Preserve the official split, disclose this, and report a known-source-disjoint sensitivity subset; some rows cannot be checked because their source ID is unavailable.
6. **Evaluation time:** Embedding ~4k clips over overlapping windows on CPU may take substantial time. Start with a small development subset, then use free Colab GPU if the measured CPU estimate is too high.

## Validation boundary

The synthetic CLAP smoke run used the pinned checkpoint on CPU and succeeded. Unit tests exercise timestamp boundaries, cosine ranking, clip-level metrics, encoder input guards, and the downmix/resampling calculation. The audio-loader test stubs the decoder: soundfile was not installed in the local environment, and the Clotho audio archives were not downloaded. Verify decoding on a real audio file after installing the pinned dependencies. No Clotho index, retrieval benchmark, temporal annotation evaluation, or deployment has been run.

## Next milestone: local index and clip retrieval

- src/embeddings/clap.py now supports batches of mono windows and text queries. The installed Transformers 4.42.4 processor accepted two audio arrays and two text queries; both outputs were finite 2 x 512 matrices on CPU. The official [CLAP documentation](https://huggingface.co/docs/transformers/model_doc/clap) describes the batched input_features shape and processor interface.
- src/embeddings/index.py writes vectors to embeddings.npy and row-aligned, relative-path metadata to manifest.json. The index loader checks schema, row count, dimensions, and finite values.
- scripts/build_index.py processes sorted files directly inside one split directory and records every deterministic window timestamp. Its limit flag creates a repeatable exploratory subset.
- scripts/run_eval.py joins caption rows by file_name, rejects index/split mismatches, embeds captions in batches, aggregates window scores by max or mean, and reports Recall@K and mAP. Per-caption predictions are written to the ignored local results/predictions directory.
- The evaluator's known-source-disjoint option filters evaluation clips whose sound_id occurs in supplied development/validation metadata. It retains unknown IDs and reports their count, while the full official evaluation remains the unfiltered run.

Validation: 19 unit tests pass; compileall passes; the index/evaluation CLI also ran against synthetic indices to verify the full and known-source-filtered paths. Those synthetic scores are wiring checks, not benchmarks. The model batch API was separately exercised with the pinned checkpoint on CPU. Actual audio decoding and a Clotho evaluation still have not been run because soundfile and the Clotho audio archives are unavailable in the local environment. Obtain the files through the official record after reviewing its terms, install the pinned dependency set, then run a real development subset.
