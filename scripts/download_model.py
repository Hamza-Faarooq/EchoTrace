"""Download the pinned CLAP checkpoint into the ignored project-local cache."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "hfhome"))

from huggingface_hub import snapshot_download

MODEL_ID = "laion/clap-htsat-unfused"
MODEL_REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
LOCAL_DIR = PROJECT_ROOT / ".cache" / "models" / "clap-htsat-unfused"
MODEL_FILES = [
    "config.json",
    "pytorch_model.bin",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
]


def main() -> int:
    path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=LOCAL_DIR,
        allow_patterns=MODEL_FILES,
    )
    print(f"model_id: {MODEL_ID}")
    print(f"revision: {MODEL_REVISION}")
    print(f"local_path: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
