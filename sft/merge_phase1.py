"""Merge the Phase 1 LoRA adapter into a 16-bit Qwen2.5-1.5B base.

The merge runs on CPU so it fits a 4 GB GPU machine. The result is the base
that Phase 2 QLoRA trains on.

    python -m sft.merge_phase1
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["USE_TORCH"] = "1"
# The fast downloader left a full-size file of zeros, and the CDN stalled
# past the 10s default. Sequential download with a longer read timeout.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft.config import BASE_MODEL_NAME, MERGED_BASE_DIR, PHASE1_ADAPTER_DIR


def _apply_download_settings() -> None:
    import huggingface_hub.constants as hf_constants

    hf_constants.HF_HUB_ENABLE_HF_TRANSFER = False
    hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = 120
    hf_constants.HF_HUB_ETAG_TIMEOUT = 60


def _safetensors_header_ok(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) < 8:
                return False
            header_len = int.from_bytes(raw, "little")
            if header_len < 2 or header_len > 50_000_000:
                return False
            return handle.read(1) == b"{"
    except OSError:
        return False


def _drop_bad_base_weights() -> None:
    """Remove a cached weight file whose size matches the model but whose bytes do not."""
    import huggingface_hub.constants as hf_constants

    root = Path(hf_constants.HF_HUB_CACHE) / f"models--{BASE_MODEL_NAME.replace('/', '--')}"
    if not root.exists():
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.endswith(".incomplete") or (
            path.name == "model.safetensors" and not _safetensors_header_ok(path)
        ):
            print(f"Removing unusable weight file {path}")
            path.unlink(missing_ok=True)


def merge() -> None:
    if not (PHASE1_ADAPTER_DIR / "adapter_model.safetensors").exists():
        raise FileNotFoundError(
            f"Missing Phase 1 adapter at {PHASE1_ADAPTER_DIR}. "
            "Expected adapter_model.safetensors from mid-training."
        )
    if (MERGED_BASE_DIR / "config.json").exists():
        print(f"Merged base already exists at {MERGED_BASE_DIR}. Skipping.")
        return

    _apply_download_settings()
    _drop_bad_base_weights()
    print(f"Loading {BASE_MODEL_NAME} in bfloat16 on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print(f"Applying adapter {PHASE1_ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, str(PHASE1_ADAPTER_DIR))
    model = model.merge_and_unload()
    MERGED_BASE_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MERGED_BASE_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_BASE_DIR)
    print(f"Saved merged 16-bit base to {MERGED_BASE_DIR}")


if __name__ == "__main__":
    merge()
