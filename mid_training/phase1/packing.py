"""Token counting, packing, and the pre-training token report."""

from __future__ import annotations

import json

from datasets import Dataset
from transformers import AutoTokenizer

from phase1.config import MAX_SEQ_LENGTH, MODEL_NAME, TARGET_TOKENS, TOKEN_REPORT_PATH


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def source_token_count(tokenizer, texts: list[str]) -> int:
    total = 0
    for text in texts:
        total += count_tokens(tokenizer, text)
    return total


def print_token_report(books: int, articles: int, corpus: int, packed_sequences: int) -> dict:
    grand = books + articles + corpus
    delta = grand - TARGET_TOKENS
    report = {
        "books_tokens": books,
        "articles_tokens": articles,
        "corpus_tokens": corpus,
        "grand_total_tokens": grand,
        "target_tokens": TARGET_TOKENS,
        "delta_vs_target": delta,
        "packed_sequences": packed_sequences,
        "max_seq_length": MAX_SEQ_LENGTH,
    }
    print("\n=== Token Report ===")
    print(f"  Books:              {books:>12,}")
    print(f"  Kaggle Articles:    {articles:>12,}")
    print(f"  Local EDGAR Corpus: {corpus:>12,}")
    print(f"  Grand Total:        {grand:>12,}")
    print(f"  Target:             {TARGET_TOKENS:>12,}")
    print(f"  Delta vs target:    {delta:>+12,}")
    print(f"  Packed sequences:   {packed_sequences:>12,}  (len={MAX_SEQ_LENGTH})")
    TOKEN_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def pack_documents(tokenizer, documents: list[str], max_seq_length: int = MAX_SEQ_LENGTH) -> Dataset:
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no EOS token; packing would merge documents.")
    flat: list[int] = []
    for doc in documents:
        ids = tokenizer.encode(doc, add_special_tokens=False)
        if not ids:
            continue
        ids.append(eos_id)
        flat.extend(ids)
    usable = (len(flat) // max_seq_length) * max_seq_length
    rows = []
    for start in range(0, usable, max_seq_length):
        chunk = flat[start : start + max_seq_length]
        rows.append(
            {
                "input_ids": chunk,
                "attention_mask": [1] * max_seq_length,
                "labels": chunk,
            }
        )
    return Dataset.from_list(rows)
