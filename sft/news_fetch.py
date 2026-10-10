"""Google News RSS fetcher.

Corporate headlines must name that company. Short ambiguous names also need an
Egypt marker in the headline. Sector headlines are about the Egyptian industry
and do not require the company name. Macro headlines are about Egypt, or about
a foreign shock that reaches Egypt (Fed, dollar, oil prices, IMF, Gulf, wheat,
Suez, Red Sea). An oil headline counts only when it is a price or supply shock. The macro block is shared by every ticker on the same dates.

Each block has at most two headlines. An empty block is written as
"none in this window." A date is usable when corporate or macro has an item.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote

import requests

from sft.config import MACRO_QUERY, NEWS_MIX, SECTOR_KEYWORDS, TICKER_META

RSS = "https://news.google.com/rss/search?q={query}&hl=en&gl=EG&ceid=EG:en"
SPACE_RE = re.compile(r"\s+")
TAG_RE = re.compile(r"<[^>]+>")
EGYPT_RE = re.compile(r"\b(egypt|egyptian|egx|cairo|egp|cbe)\b|egyptian pound", re.IGNORECASE)
MACRO_TOPIC_RE = re.compile(
    r"\binflation\b|interest rates?|\bcentral bank\b|\bcbe\b|\bpound\b|\begp\b|"
    r"\bimf\b|remittances?",
    re.IGNORECASE,
)
SHOCK_RE = re.compile(
    r"federal reserve|\bfed\b|\bdollars?\b|\bwheat\b|\bgulf\b",
    re.IGNORECASE,
)
OIL_MENTION_RE = re.compile(r"\boil\b|\bcrude\b", re.IGNORECASE)
OIL_PRICE_RE = re.compile(r"\bprices?\b|\bbrent\b|\bbarrels?\b|\bopec\b|\bsupply\b", re.IGNORECASE)
SUEZ_RE = re.compile(r"\bsuez\b|red sea", re.IGNORECASE)
SUEZ_SHOCK_RE = re.compile(
    r"canal|shipping|revenue|attacks?|tensions?|disrupt|trade|\bvessels?\b|"
    r"houthi|blockade|\bwar\b",
    re.IGNORECASE,
)
BANK_SECTOR_RE = re.compile(
    r"egyptian banks?|egypt'?s banks?|\begypt banks?\b|"
    r"banks operating in egypt|banks in egypt|banking sector|bank of egypt|\bstate bank\b",
    re.IGNORECASE,
)
BUCKETS = ("corporate", "sector", "macro")
EMPTY_BLOCK = "none in this window."

# query without the date span, regex that is enough on its own, regex that
# also needs an Egypt marker. Bare short names such as CIB and ADIB are only
# in the third field.
_COMPANY_RULES: dict[str, tuple[str, str, str]] = {
    "COMI.CA": (
        '"Commercial International Bank" OR "CIB Egypt" OR COMI OR CIB',
        r"commercial international bank|\bcib egypt\b|\bcomi\b",
        r"\bcib\b",
    ),
    "HRHO.CA": (
        '"EFG Hermes" OR "Hermes Holding" OR HRHO',
        r"efg hermes|hermes holding|\bhrho\b",
        "",
    ),
    "FWRY.CA": ('Fawry OR FWRY', r"\bfawry\b|\bfwry\b", ""),
    "BTFH.CA": ("Beltone OR BTFH", r"\bbeltone\b|\bbtfh\b", ""),
    "TMGH.CA": (
        '"Talaat Moustafa" OR TMGH',
        r"talaat moustafa|\btmgh\b",
        r"\btmg\b",
    ),
    "PHDC.CA": ('"Palm Hills" OR PHDC', r"palm hills|\bphdc\b", ""),
    "MASR.CA": ('"Madinet Masr" OR MASR', r"madinet masr|\bmasr\b", ""),
    "ORAS.CA": (
        '"Orascom Construction" OR ORAS',
        r"orascom construction|\boras\b",
        "",
    ),
    "SWDY.CA": ("Elsewedy OR SWDY", r"\belsewedy\b|\bswdy\b", ""),
    "ESRS.CA": (
        '"Ezz Steel" OR "Ezz Dekheila" OR ESRS',
        r"ezz steel|ezz dekheila|\besrs\b",
        "",
    ),
    "ORWE.CA": ('"Oriental Weavers" OR ORWE', r"oriental weavers|\borwe\b", ""),
    "ABUK.CA": ('"Abu Qir" OR ABUK', r"abu qir|\babuk\b", ""),
    "AMOC.CA": ('AMOC OR "Alexandria Mineral Oils"', r"\bamoc\b|alexandria mineral oils", ""),
    "SKPC.CA": (
        '"Sidi Kerir" OR SIDPEC OR SKPC',
        r"sidi kerir|\bsidpec\b|\bskpc\b",
        "",
    ),
    "EAST.CA": (
        '"Eastern Company" OR "Eastern Tobacco"',
        r"eastern company|eastern tobacco",
        "",
    ),
    "JUFO.CA": ("Juhayna OR JUFO", r"\bjuhayna\b|\bjufo\b", ""),
    "ISPH.CA": (
        '"Ibnsina Pharma" OR "Ibn Sina Pharma" OR ISPH',
        r"ibnsina pharma|ibn sina pharma|\bisph\b",
        "",
    ),
    "ETEL.CA": (
        '"Telecom Egypt" OR ETEL',
        r"telecom egypt|\betel\b",
        "",
    ),
    "ALCN.CA": (
        '"Alexandria Container" OR ALCN',
        r"alexandria container|\balcn\b",
        "",
    ),
    "CCAP.CA": ('"Qalaa Holdings" OR Qalaa OR CCAP', r"\bqalaa\b|\bccap\b", ""),
    "EKHO.CA": (
        '"EK Holding" OR "Egyptian Kuwaiti Holding" OR EKHO',
        r"\bek holding\b|egyptian kuwaiti|\bekho\b",
        "",
    ),
    "CIEB.CA": (
        '"Credit Agricole Egypt" OR CIEB',
        r"credit agricole egypt|\bcieb\b",
        r"credit agricole",
    ),
    "MFPC.CA": (
        'MOPCO OR "Misr Fertilizers" OR MFPC',
        r"\bmopco\b|misr fertilizers|\bmfpc\b",
        "",
    ),
    "SUGR.CA": ('"Delta Sugar" OR SUGR', r"delta sugar|\bsugr\b", ""),
    "OLFI.CA": ('"Obour Land" OR OLFI', r"obour land|\bolfi\b", ""),
    "EFID.CA": ('"Edita Food" OR EFID', r"edita food|\befid\b", r"\bedita\b"),
    "GBCO.CA": ('"GB Auto" OR GBCO', r"gb auto|\bgbco\b", ""),
    "MTIE.CA": ('"MM Group" OR MTIE', r"\bmm group\b|\bmtie\b", ""),
    "EFIH.CA": (
        '"e-finance" OR efinance OR EFIH',
        r"e-finance|e finance|\befinance\b|\befih\b",
        "",
    ),
    "ADIB.CA": (
        '"ADIB Egypt" OR "Abu Dhabi Islamic" Egypt',
        r"adib egypt",
        r"\badib\b|abu dhabi islamic bank",
    ),
    "HELI.CA": ('"Heliopolis Housing" OR HELI', r"heliopolis housing|\bheli\b", ""),
    "MNHD.CA": (
        '"Madinet Nasr" OR "Nasr City Housing" OR MNHD',
        r"madinet nasr|nasr city housing|\bmnhd\b",
        "",
    ),
    "SVCE.CA": ('"South Valley Cement" OR SVCE', r"south valley cement|\bsvce\b", ""),
    "ORHD.CA": (
        '"Orascom Development" OR ORHD',
        r"orascom development|\borhd\b",
        "",
    ),
}

_COMPILED: dict[str, tuple[re.Pattern[str], re.Pattern[str] | None]] = {}
for _ticker, (_query, _accept, _short) in _COMPANY_RULES.items():
    _COMPILED[_ticker] = (
        re.compile(_accept, re.IGNORECASE),
        re.compile(_short, re.IGNORECASE) if _short else None,
    )

_MACRO_CACHE: dict[tuple[date, date], list[dict]] = {}


def clear_macro_cache() -> None:
    _MACRO_CACHE.clear()


def _clean(text: str) -> str:
    text = TAG_RE.sub(" ", text or "")
    return SPACE_RE.sub(" ", text).strip()


def _headline(item: dict) -> str:
    """Headline text without the trailing Google News source suffix."""
    title = _clean(item.get("title") or "")
    source = _clean(item.get("source") or "")
    if source and " - " in title:
        head, _, tail = title.rpartition(" - ")
        if source.lower() in tail.lower():
            return head.strip()
    return title


def _fingerprint(title: str) -> str:
    return hashlib.sha1(_clean(title).lower().encode("utf-8")).hexdigest()


def _span(start: date, end: date) -> str:
    # Google News treats `before` as exclusive, so add one day.
    before = end + timedelta(days=1)
    return f"after:{start.isoformat()} before:{before.isoformat()}"


def _has_term(headline: str, terms: tuple[str, ...]) -> bool:
    low = headline.lower()
    for term in terms:
        if " " in term:
            if term in low:
                return True
        elif re.search(rf"\b{re.escape(term)}\b", low):
            return True
    return False


def is_corporate_headline(ticker: str, headline: str) -> bool:
    accept, short = _COMPILED[ticker]
    if accept.search(headline):
        return True
    return bool(short and short.search(headline) and EGYPT_RE.search(headline))


def _published_on(item: dict) -> date | None:
    raw = item.get("published") or ""
    try:
        return parsedate_to_datetime(raw).date()
    except (TypeError, ValueError, OverflowError):
        return None


def _in_window(item: dict, start: date | None, end: date | None) -> bool:
    if start is None or end is None:
        return True
    published = _published_on(item)
    if published is None:
        return True
    return start <= published <= end


def is_sector_headline(sector: str, headline: str) -> bool:
    if sector == "banking":
        return bool(BANK_SECTOR_RE.search(headline))
    if not EGYPT_RE.search(headline):
        return False
    return _has_term(headline, SECTOR_KEYWORDS[sector])


def _oil_price_shock(headline: str) -> bool:
    """Oil counts only as a price or supply shock, not as pollution or a project."""
    if re.search(r"\bbrent\b|\bopec\b", headline, re.IGNORECASE):
        return True
    return bool(OIL_MENTION_RE.search(headline) and OIL_PRICE_RE.search(headline))


def _suez_shock(headline: str) -> bool:
    return bool(SUEZ_RE.search(headline) and SUEZ_SHOCK_RE.search(headline))


def is_macro_headline(headline: str) -> bool:
    if EGYPT_RE.search(headline) and MACRO_TOPIC_RE.search(headline):
        return True
    if SHOCK_RE.search(headline) or _oil_price_shock(headline):
        return True
    return _suez_shock(headline)


def _macro_rank(headline: str) -> int:
    egypt = bool(EGYPT_RE.search(headline))
    topic = bool(MACRO_TOPIC_RE.search(headline))
    shock = bool(SHOCK_RE.search(headline) or _oil_price_shock(headline) or _suez_shock(headline))
    if egypt and (topic or shock):
        return 0
    if topic or shock:
        return 1
    return 2


def corporate_query(ticker: str, start: date, end: date) -> str:
    return f"({_COMPANY_RULES[ticker][0]}) Egypt {_span(start, end)}"


def sector_query(sector: str, start: date, end: date) -> str:
    parts = [f'"{term}"' if " " in term else term for term in SECTOR_KEYWORDS[sector]]
    return f"Egypt ({' OR '.join(parts)}) {_span(start, end)}"


def macro_query(start: date, end: date) -> str:
    return f"{MACRO_QUERY} {_span(start, end)}"


def _rss_items(query: str, timeout: float = 20.0) -> list[dict]:
    url = RSS.format(query=quote(query))
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "egx-sft-news/1.0"},
        )
        response.raise_for_status()
    except requests.RequestException:
        return []
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError:
        return []
    items = []
    for node in root.findall("./channel/item"):
        title = _clean(node.findtext("title") or "")
        if not title:
            continue
        source = _clean((node.find("source").text if node.find("source") is not None else "") or "")
        published = _clean(node.findtext("pubDate") or "")
        items.append({"title": title, "source": source, "published": published})
    return items


def select_macro(raw: list[dict], start: date | None = None, end: date | None = None) -> list[dict]:
    raw = [item for item in raw if _in_window(item, start, end)]
    ranked = sorted(enumerate(raw), key=lambda pair: (_macro_rank(_headline(pair[1])), pair[0]))
    chosen: list[dict] = []
    seen: set[str] = set()
    for _index, item in ranked:
        headline = _headline(item)
        if not is_macro_headline(headline):
            continue
        key = _fingerprint(headline)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(
            {
                "title": headline,
                "source": item.get("source", ""),
                "published": item.get("published", ""),
                "bucket": "macro",
            }
        )
        if len(chosen) >= NEWS_MIX["macro"]:
            break
    return chosen


def _macro_items(start: date, end: date) -> list[dict]:
    key = (start, end)
    if key not in _MACRO_CACHE:
        _MACRO_CACHE[key] = select_macro(_rss_items(macro_query(start, end)), start, end)
    return _MACRO_CACHE[key]


def assemble_news(
    ticker: str,
    corporate_raw: list[dict],
    sector_raw: list[dict],
    macro_selected: list[dict],
    start: date | None = None,
    end: date | None = None,
) -> list[dict]:
    """Apply the headline rules and caps. Macro items are appended unchanged."""
    corporate_raw = [item for item in corporate_raw if _in_window(item, start, end)]
    sector_raw = [item for item in sector_raw if _in_window(item, start, end)]
    sector = TICKER_META[ticker][2]
    macro_keys = {_fingerprint(item["title"]) for item in macro_selected}
    selected: list[dict] = []
    seen: set[str] = set()

    def add(bucket: str, item: dict, headline: str) -> None:
        selected.append(
            {
                "title": headline,
                "source": item.get("source", ""),
                "published": item.get("published", ""),
                "bucket": bucket,
            }
        )
        seen.add(_fingerprint(headline))

    kept = 0
    for item in corporate_raw:
        if kept >= NEWS_MIX["corporate"]:
            break
        headline = _headline(item)
        if not is_corporate_headline(ticker, headline):
            continue
        key = _fingerprint(headline)
        if key in seen:
            continue
        add("corporate", item, headline)
        kept += 1

    kept = 0
    for item in sector_raw:
        if kept >= NEWS_MIX["sector"]:
            break
        headline = _headline(item)
        key = _fingerprint(headline)
        if key in seen or key in macro_keys:
            continue
        if is_corporate_headline(ticker, headline) or not is_sector_headline(sector, headline):
            continue
        add("sector", item, headline)
        kept += 1

    selected.extend(macro_selected)
    return selected


def fetch_window(ticker: str, start: date, end: date) -> list[dict]:
    """Return corporate, sector, and shared macro headlines for [start, end]."""
    sector = TICKER_META[ticker][2]
    return assemble_news(
        ticker,
        _rss_items(corporate_query(ticker, start, end)),
        _rss_items(sector_query(sector, start, end)),
        _macro_items(start, end),
        start,
        end,
    )


def window_is_usable(items: list[dict]) -> bool:
    buckets = {item["bucket"] for item in items}
    return "corporate" in buckets or "macro" in buckets


def has_old_news_layout(path: Path) -> bool:
    """True when an existing jsonl row predates the Corporate/Sector/Macro blocks."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            user = row["messages"][1]["content"]
            return "\nCorporate:\n" not in f"\n{user}"
    return False


ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_RE = re.compile(r"[A-Za-z]")


def needs_english_translation(text: str) -> bool:
    """True when Arabic letters outnumber Latin letters, so the block is not yet English."""
    return len(ARABIC_RE.findall(text)) > len(LATIN_RE.findall(text))


def translate_news_to_english(client, news_block: str) -> str:
    """Turn Arabic headlines into English. Already-English blocks are not sent to Gemini."""
    if not news_block.strip() or not needs_english_translation(news_block):
        return news_block
    prompt = (
        "Translate these Egyptian stock-market news lines into English. "
        "Keep the Corporate, Sector, and Macro headings. "
        "Keep a line that says 'none in this window.' unchanged. "
        "Keep the source in parentheses and the date in brackets. "
        "Translate the headline and any Arabic words. If a line is already English, copy it unchanged. "
        "Return only the block, with no introduction.\n\n"
        f"{news_block}"
    )
    translated = client.generate(prompt).strip()
    return translated or news_block


def format_news(items: list[dict]) -> str:
    grouped = {bucket: [] for bucket in BUCKETS}
    for item in items:
        source = f" ({item['source']})" if item.get("source") else ""
        published = f" [{item['published']}]" if item.get("published") else ""
        grouped[item["bucket"]].append(f"- {item['title']}{source}{published}")
    blocks = []
    for bucket in BUCKETS:
        body = "\n".join(grouped[bucket]) if grouped[bucket] else EMPTY_BLOCK
        blocks.append(f"{bucket.capitalize()}:\n{body}")
    return "\n".join(blocks)


def probe_start(ticker: str, earliest: date, latest: date, pause: float = 1.0) -> date | None:
    """Oldest date in [earliest, latest] that still has corporate or macro news.

    Probes 14-day slices stepping backward by about a month. Returns None
    when even the most recent slice is empty.
    """
    if latest <= earliest:
        return None
    found: date | None = None
    cursor_end = latest
    while cursor_end > earliest:
        cursor_start = max(earliest, cursor_end - timedelta(days=14))
        items = fetch_window(ticker, cursor_start, cursor_end)
        if window_is_usable(items):
            found = cursor_start
            cursor_end = cursor_start - timedelta(days=16)
            time.sleep(pause)
            continue
        break
    return found
