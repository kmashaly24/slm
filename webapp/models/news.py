"""News block for T-7 through T, wider than the training mix."""

from __future__ import annotations

from datetime import date, timedelta

from sft.config import NEWS_LOOKBACK_DAYS, TICKER_META
from sft.news_fetch import (
    EGYPT_RE,
    _fingerprint,
    _headline,
    _in_window,
    _macro_rank,
    _rss_items,
    _span,
    corporate_query,
    format_news,
    is_corporate_headline,
    is_macro_headline,
    is_sector_headline,
    macro_query,
    sector_query,
)
from webapp.config import NEWS_BUCKET_LIMIT


def get_block(symbol: str, analysis_date: date, company: str, sector: str) -> str:
    """Up to six headlines in each bucket for the same week used in training."""
    start = analysis_date - timedelta(days=NEWS_LOOKBACK_DAYS)
    if symbol in TICKER_META:
        return format_news(_trained_items(symbol, start, analysis_date))
    return format_news(_generic_items(company, sector, start, analysis_date))


def _trained_items(symbol: str, start: date, end: date) -> list[dict]:
    sector = TICKER_META[symbol][2]
    seen: set[str] = set()
    corporate = _take(
        _rss_items(corporate_query(symbol, start, end)),
        start,
        end,
        "corporate",
        seen,
        lambda headline: is_corporate_headline(symbol, headline),
    )
    sector_items = _take(
        _rss_items(sector_query(sector, start, end)),
        start,
        end,
        "sector",
        seen,
        lambda headline: is_sector_headline(sector, headline) and not is_corporate_headline(symbol, headline),
    )
    return corporate + sector_items + _macro_items(start, end, seen)


def _generic_items(company: str, sector: str, start: date, end: date) -> list[dict]:
    seen: set[str] = set()
    corporate = _take(
        _rss_items(f'"{_quote(company)}" Egypt {_span(start, end)}') if _quote(company) else [],
        start,
        end,
        "corporate",
        seen,
    )
    sector_name = _quote(sector)
    sector_items = _take(
        _rss_items(f'Egypt "{sector_name}" {_span(start, end)}') if sector_name else [],
        start,
        end,
        "sector",
        seen,
        lambda headline: bool(EGYPT_RE.search(headline)),
    )
    return corporate + sector_items + _macro_items(start, end, seen)


def _macro_items(start: date, end: date, seen: set[str]) -> list[dict]:
    raw = [item for item in _rss_items(macro_query(start, end)) if _in_window(item, start, end)]
    ranked = sorted(enumerate(raw), key=lambda pair: (_macro_rank(_headline(pair[1])), pair[0]))
    chosen = []
    for _index, item in ranked:
        headline = _headline(item)
        if not headline or not is_macro_headline(headline):
            continue
        key = _fingerprint(headline)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(_row("macro", item, headline))
        if len(chosen) >= NEWS_BUCKET_LIMIT:
            break
    return chosen


def _take(
    raw: list[dict],
    start: date,
    end: date,
    bucket: str,
    seen: set[str],
    accept=None,
) -> list[dict]:
    chosen = []
    for item in raw:
        if not _in_window(item, start, end):
            continue
        headline = _headline(item)
        if not headline:
            continue
        if accept is not None and not accept(headline):
            continue
        key = _fingerprint(headline)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(_row(bucket, item, headline))
        if len(chosen) >= NEWS_BUCKET_LIMIT:
            break
    return chosen


def _row(bucket: str, item: dict, headline: str) -> dict:
    return {
        "title": headline,
        "source": item.get("source", ""),
        "published": item.get("published", ""),
        "bucket": bucket,
    }


def _quote(text: str) -> str:
    return " ".join((text or "").replace('"', " ").split())
