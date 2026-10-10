"""Paths and generation limits for the local analyst app."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "grpo" / "merged_16bit"
VIEWS_DIR = Path(__file__).resolve().parent / "views"

HOST = "127.0.0.1"
PORT = 8000

# Prompt cap used while training. Completions sit on top of this.
MAX_SEQ_LENGTH = 2048
MAX_PROMPT_TOKENS = 1024
SFT_MAX_NEW_TOKENS = 1024
GRPO_MAX_NEW_TOKENS = 1024
# Wider than the training mix. The model keeps the headlines that matter.
NEWS_BUCKET_LIMIT = 6

HORIZONS = (1, 3, 5, 10)
DEFAULT_HORIZON = 5
