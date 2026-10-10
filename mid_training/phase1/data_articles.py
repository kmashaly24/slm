"""Parse Kaggle US financial news and take a 10% sample per month."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from phase1.config import (
    ARTICLE_SAMPLE_FRACTION,
    ARTICLES_DIR,
    MIN_ARTICLE_WORDS,
    SEED,
)

SPACE_RE = re.compile(r"\s+")


def _word_count(text: str) -> int:
    return len(text.split())


def _fingerprint(title: str, body: str) -> str:
    normalized = SPACE_RE.sub(" ", f"{title} {body[:400]}".lower()).strip()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _article_text(payload: dict) -> str | None:
    language = str(payload.get("language") or "").strip().lower()
    if language and language != "english":
        return None
    title = str(payload.get("title") or "").strip()
    body = str(payload.get("text") or "").strip()
    published = str(payload.get("published") or "")
    date = published[:10]
    if _word_count(body) < MIN_ARTICLE_WORDS or not title:
        return None
    return f"Date: {date}\nTitle: {title}\n\n{body}"


def _month_dirs() -> list[Path]:
    dirs = sorted(p for p in ARTICLES_DIR.iterdir() if p.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No month folders under {ARTICLES_DIR}")
    return dirs


def load_articles() -> list[str]:
    rng = random.Random(SEED)
    seen: set[str] = set()
    selected: list[str] = []
    print("\n=== Article sampling ===")
    for month_dir in _month_dirs():
        eligible: list[str] = []
        scanned = 0
        for path in month_dir.glob("*.json"):
            scanned += 1
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            text = _article_text(payload)
            if text is None:
                continue
            title = str(payload.get("title") or "")
            body = str(payload.get("text") or "")
            key = _fingerprint(title, body)
            if key in seen:
                continue
            seen.add(key)
            eligible.append(text)
        keep = max(1, int(round(len(eligible) * ARTICLE_SAMPLE_FRACTION))) if eligible else 0
        if keep and keep < len(eligible):
            chosen = rng.sample(eligible, keep)
        else:
            chosen = eligible
        selected.extend(chosen)
        print(
            f"  {month_dir.name}: scanned={scanned:,} eligible={len(eligible):,} "
            f"sampled={len(chosen):,}"
        )
    print(f"  total sampled articles: {len(selected):,}")
    return selected
