"""Dynamically sample local EDGAR JSON until the token budget is filled."""

from __future__ import annotations

import json
import random
import re

from phase1.config import (
    CORPUS_DIR,
    CORPUS_SECTIONS,
    MIN_CORPUS_SECTION_WORDS,
    SEED,
)

BOILERPLATE_RE = re.compile(
    r"^(omitted|none|not applicable|n/?a)\.?$",
    re.IGNORECASE,
)
ITEM_HEADER_RE = re.compile(
    r"^Item\s+\d+[A-Z]?\.\s*[^\n]*\n?",
    re.IGNORECASE,
)


def _clean_section(raw: str) -> str:
    text = raw.replace("\xa0", " ").replace("\r", "\n").strip()
    text = ITEM_HEADER_RE.sub("", text, count=1).strip()
    compact = re.sub(r"\s+", " ", text)
    if not compact or BOILERPLATE_RE.match(compact):
        return ""
    if len(compact.split()) < MIN_CORPUS_SECTION_WORDS:
        return ""
    return text


def filing_text(payload: dict) -> str:
    parts = []
    for key in CORPUS_SECTIONS:
        raw = payload.get(key)
        if not isinstance(raw, str):
            continue
        cleaned = _clean_section(raw)
        if cleaned:
            parts.append(cleaned)
    return "\n\n".join(parts).strip()


def load_corpus(token_budget: int, count_tokens) -> tuple[list[str], int]:
    """Read shuffled local filings until `token_budget` is consumed.

    `count_tokens` is a callable from text -> int using the shared tokenizer.
    Returns (documents, tokens_added). If the budget is already exhausted by
    books and articles, nothing is added.
    """
    if token_budget <= 0:
        print("\n=== EDGAR corpus ===")
        print("  books + articles already cover the token target; corpus skipped.")
        return [], 0

    paths = sorted(CORPUS_DIR.glob("*.json"))
    rng = random.Random(SEED)
    rng.shuffle(paths)
    selected: list[str] = []
    used = 0
    scanned = 0
    skipped_empty = 0
    print("\n=== EDGAR corpus ===")
    print(f"  remaining token budget: {token_budget:,}")
    for path in paths:
        if used >= token_budget:
            break
        scanned += 1
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        text = filing_text(payload)
        if not text:
            skipped_empty += 1
            continue
        n_tokens = count_tokens(text)
        if n_tokens <= 0:
            skipped_empty += 1
            continue
        if used + n_tokens > token_budget and selected:
            break
        selected.append(text)
        used += n_tokens
    print(
        f"  scanned={scanned:,} kept={len(selected):,} "
        f"empty_skipped={skipped_empty:,} tokens={used:,}"
    )
    return selected, used
