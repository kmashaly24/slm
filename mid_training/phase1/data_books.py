"""Extract and clean finance-book PDFs using pages.json bounds."""

from __future__ import annotations

import json
import re
from collections import Counter

import pymupdf as fitz

from phase1.config import (
    BOOKS_DIR,
    MIN_BOOK_BLOCK_WORDS,
    PAGES_JSON,
    TABLE_DIGIT_SYMBOL_RATIO,
)

CITATION_RE = re.compile(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]")
PAGE_NUM_RE = re.compile(
    r"^(?:page\s+)?[-–—]?\s*\d{1,4}\s*[-–—]?$",
    re.IGNORECASE,
)
MULTI_SPACE_RE = re.compile(r"[ \t]+")
MULTI_NL_RE = re.compile(r"\n{3,}")
HYPHEN_BREAK_RE = re.compile(r"([A-Za-z])-\n([A-Za-z])")


def load_page_map(path=PAGES_JSON) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        entries = json.load(handle)
    if not isinstance(entries, list):
        raise ValueError(f"{path} must be a list of book page ranges.")
    return entries


def _is_table_block(text: str) -> bool:
    if not text:
        return True
    digit_symbol = sum(1 for ch in text if not ch.isalpha() and not ch.isspace())
    return (digit_symbol / len(text)) > TABLE_DIGIT_SYMBOL_RATIO


def _normalize_block(text: str) -> str:
    text = text.replace("\r", "\n").replace("\xa0", " ")
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)
    lines = []
    for raw in text.split("\n"):
        line = MULTI_SPACE_RE.sub(" ", raw).strip()
        if not line or PAGE_NUM_RE.match(line):
            continue
        lines.append(line)
    if not lines:
        return ""
    merged: list[str] = []
    buffer = lines[0]
    for line in lines[1:]:
        if buffer.endswith(("-", "—")) or not buffer[-1] in ".!?:;\"'”":
            buffer = f"{buffer} {line}"
        else:
            merged.append(buffer)
            buffer = line
    merged.append(buffer)
    cleaned = " ".join(merged)
    cleaned = CITATION_RE.sub("", cleaned)
    cleaned = MULTI_SPACE_RE.sub(" ", cleaned).strip()
    if len(cleaned.split()) < MIN_BOOK_BLOCK_WORDS:
        return ""
    return cleaned


def _header_footer_lines(pages: list[fitz.Page], threshold: float = 0.4) -> set[str]:
    counts: Counter[str] = Counter()
    for page in pages:
        lines = [ln.strip() for ln in page.get_text("text").splitlines() if ln.strip()]
        edges = lines[:2] + lines[-2:]
        for line in set(edges):
            counts[line] += 1
    min_hits = max(3, int(len(pages) * threshold))
    return {line for line, n in counts.items() if n >= min_hits}


def extract_book(entry: dict) -> dict:
    name = entry["book_name"]
    start = int(entry["start"])
    end = int(entry["end"])
    pdf_path = BOOKS_DIR / f"{name}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    doc = fitz.open(pdf_path)
    if start < 1 or end > doc.page_count or start > end:
        raise ValueError(
            f"{name}: page range {start}-{end} is outside 1..{doc.page_count}"
        )
    pages = [doc[i] for i in range(start - 1, end)]
    noise = _header_footer_lines(pages)
    paragraphs: list[str] = []
    for page in pages:
        blocks = page.get_text("blocks")
        for block in blocks:
            raw = block[4] if len(block) > 4 else ""
            if not isinstance(raw, str):
                continue
            raw = "\n".join(
                line for line in raw.splitlines() if line.strip() not in noise
            )
            if _is_table_block(raw):
                continue
            cleaned = _normalize_block(raw)
            if cleaned:
                paragraphs.append(cleaned)
    text = MULTI_NL_RE.sub("\n\n", "\n\n".join(paragraphs)).strip()
    preview_first = pages[0].get_text("text")[:180].replace("\n", " ")
    preview_last = pages[-1].get_text("text")[:180].replace("\n", " ")
    doc.close()
    return {
        "book_name": name,
        "start": start,
        "end": end,
        "text": text,
        "preview_first": preview_first,
        "preview_last": preview_last,
    }


def load_books() -> list[dict]:
    books = [extract_book(entry) for entry in load_page_map()]
    print("\n=== Book page-range spot check ===")
    for book in books:
        print(f"[{book['book_name']}] pages {book['start']}-{book['end']}")
        print(f"  first: {book['preview_first']}")
        print(f"  last:  {book['preview_last']}")
        print(f"  chars: {len(book['text']):,}")
    return books
