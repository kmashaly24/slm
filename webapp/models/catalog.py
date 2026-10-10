"""EGX name search and company metadata.

Yahoo's search endpoint currently returns long ISIN symbols such as
EGS30901C010.CA, and those series have almost no daily history. The short
tickers used in training still do, so a typed name is matched against the
trained universe first and a Yahoo hit is rewritten to that ticker when the
company name lines up.
"""

from __future__ import annotations

import re
import threading

import requests
from difflib import SequenceMatcher

from sft.config import TICKER_META
from sft.news_fetch import _COMPANY_RULES

SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
SUMMARY_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
CRUMB_URL = "https://query1.finance.yahoo.com/v1/test/getcrumb"
LISTED_URL = "https://english.mubasher.info/api/1/listed-companies"
HEADERS = {"User-Agent": "Mozilla/5.0"}
_RIGHTS_CODE = re.compile(r"_r\d+$", re.IGNORECASE)
_DIR_LOCK = threading.Lock()
_DIRECTORY: list[dict] | None = None
_SESSION: requests.Session | None = None
_CRUMB = ""


def _aliases(symbol: str) -> list[str]:
    arabic, english, _sector = TICKER_META[symbol]
    names = [english, arabic, symbol, symbol.replace(".CA", "")]
    query = _COMPANY_RULES.get(symbol, ("",))[0]
    for part in query.split(" OR "):
        part = part.strip().strip('"')
        if part:
            names.append(part)
    unique = []
    seen = set()
    for name in names:
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(name)
    return unique


_UNIVERSE = [
    {
        "symbol": symbol,
        "english": english,
        "arabic": arabic,
        "sector": sector,
        "aliases": _aliases(symbol),
    }
    for symbol, (arabic, english, sector) in TICKER_META.items()
]


def search(query: str) -> list[dict]:
    """EGX matches for a typed name or ticker. Empty query returns nothing."""
    query = (query or "").strip()
    if len(query) < 2:
        return []
    folded = query.casefold()
    hits = []
    seen = set()
    for row in _UNIVERSE:
        if not any(_hit(folded, alias) for alias in row["aliases"]):
            continue
        hits.append({"symbol": row["symbol"], "name": row["english"]})
        seen.add(row["symbol"])
    hits.sort(key=lambda item: _rank(folded, item))
    try:
        yahoo_hits = _yahoo_search(query)
    except requests.RequestException:
        yahoo_hits = []
    for item in yahoo_hits:
        symbol = _known_symbol(item["name"]) or _listed_symbol(item["name"]) or item["symbol"]
        if symbol in seen:
            continue
        seen.add(symbol)
        name = item["name"]
        if symbol in TICKER_META:
            name = TICKER_META[symbol][1]
        hits.append({"symbol": symbol, "name": name})
    return hits[:12]


def profile(symbol: str) -> dict:
    """Company string and sector. Trained names come from the SFT universe."""
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise ValueError("Missing ticker")
    known = _known_symbol(symbol) or symbol
    if known not in TICKER_META:
        listed = _listed_match(_chart_name(known) or "")
        if listed:
            known = listed["symbol"]
    if known in TICKER_META:
        arabic, english, sector_key = TICKER_META[known]
        return {
            "symbol": known,
            "company": f"{english} / {arabic}",
            "sector": sector_key.replace("_", " "),
            "trained": True,
        }
    company, sector = _yahoo_profile(known)
    return {
        "symbol": known,
        "company": company or known,
        "sector": sector,
        "trained": False,
    }


def _yahoo_search(query: str) -> list[dict]:
    response = requests.get(
        SEARCH_URL,
        params={"q": query, "quotesCount": 25, "newsCount": 0},
        headers=HEADERS,
        timeout=20,
    )
    response.raise_for_status()
    matches = []
    for item in response.json().get("quotes") or []:
        symbol = str(item.get("symbol") or "")
        exchange = str(item.get("exchange") or "")
        if exchange != "CAI" and not symbol.endswith(".CA"):
            continue
        name = str(item.get("shortname") or item.get("longname") or symbol)
        matches.append({"symbol": symbol, "name": name})
    return matches


def _yahoo_profile(symbol: str) -> tuple[str, str]:
    company = _chart_name(symbol)
    sector = _quote_sector(symbol)
    listed = _listed_by_symbol(symbol) or _listed_match(company)
    if listed:
        company = company or listed["name"]
        sector = sector or listed["sector"]
    return company, sector


def _quote_sector(symbol: str) -> str:
    session, crumb = _yahoo_session()
    if not crumb:
        return ""
    try:
        response = session.get(
            SUMMARY_URL.format(symbol=symbol),
            params={"modules": "assetProfile,price", "crumb": crumb},
            timeout=20,
        )
        if response.status_code != 200:
            return ""
        result = (response.json().get("quoteSummary") or {}).get("result") or []
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
        return ""
    if not result:
        return ""
    asset = result[0].get("assetProfile") or {}
    return _text(asset.get("sector"))


def _yahoo_session() -> tuple[requests.Session, str]:
    global _SESSION, _CRUMB
    if _SESSION is not None:
        return _SESSION, _CRUMB
    session = requests.Session()
    session.headers.update(HEADERS)
    crumb = ""
    try:
        session.get("https://fc.yahoo.com", timeout=20, allow_redirects=True)
        crumb = session.get(CRUMB_URL, timeout=20).text.strip()
    except requests.RequestException:
        crumb = ""
    if "<" in crumb or "too many" in crumb.casefold():
        crumb = ""
    _SESSION = session
    _CRUMB = crumb
    return session, crumb


def _directory() -> list[dict]:
    """EGX names and sectors from the public listed-companies feed. Cached."""
    global _DIRECTORY
    if _DIRECTORY is not None:
        return _DIRECTORY
    with _DIR_LOCK:
        if _DIRECTORY is not None:
            return _DIRECTORY
        rows = []
        seen = set()
        for start in range(0, 400, 20):
            try:
                response = requests.get(
                    LISTED_URL,
                    params={"market": "EGX", "start": start},
                    headers=HEADERS,
                    timeout=20,
                )
                response.raise_for_status()
                page = response.json().get("rows") or []
            except (requests.RequestException, ValueError, TypeError):
                break
            if not page:
                break
            for row in page:
                name = str(row.get("name") or "").strip()
                code = _RIGHTS_CODE.sub("", str(row.get("symbol") or "").strip())
                sector = _clean_sector(str(row.get("sector") or ""))
                if not name or not code:
                    continue
                symbol = f"{code.upper()}.CA"
                if symbol in seen:
                    continue
                seen.add(symbol)
                rows.append({"symbol": symbol, "name": name, "sector": sector})
        _DIRECTORY = rows
        return rows


def _listed_symbol(company: str) -> str:
    match = _listed_match(company)
    return match["symbol"] if match else ""


def _listed_match(company: str) -> dict | None:
    folded = (company or "").casefold().strip()
    if len(folded) < 3:
        return None
    best = None
    best_len = 0
    for row in _directory():
        name = row["name"].casefold()
        if len(name) < 4:
            continue
        if name in folded or folded in name:
            if len(name) > best_len:
                best = row
                best_len = len(name)
    if best:
        return best
    best_row = None
    best_score = 0.0
    for row in _directory():
        if len(row["name"]) < 8:
            continue
        score = SequenceMatcher(None, row["name"].casefold(), folded).ratio()
        if score > best_score:
            best_score = score
            best_row = row
    if best_score >= 0.84:
        return best_row
    return None


def _clean_sector(sector: str) -> str:
    text = " ".join((sector or "").split())
    for suffix in (" Sector Index", " Index"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    return text


def _chart_name(symbol: str) -> str:
    try:
        response = requests.get(
            CHART_URL.format(symbol=symbol),
            params={"interval": "1d", "range": "5d"},
            headers=HEADERS,
            timeout=20,
        )
        response.raise_for_status()
        meta = response.json()["chart"]["result"][0]["meta"]
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
        return ""
    long_name = str(meta.get("longName") or "").strip()
    short_name = str(meta.get("shortName") or "").strip()
    for name in (long_name, short_name):
        if _usable_name(name):
            return name
    return ""


def _usable_name(name: str) -> bool:
    if len(name) < 3 or "," in name or name.upper().endswith(".CA"):
        return False
    return True


def _listed_by_symbol(symbol: str) -> dict | None:
    target = (symbol or "").upper()
    for row in _directory():
        if row["symbol"] == target:
            return row
    return None


def _known_symbol(name_or_symbol: str) -> str:
    folded = (name_or_symbol or "").casefold().strip()
    if not folded:
        return ""
    if folded.endswith(".ca"):
        direct = folded.upper()
        if direct in TICKER_META:
            return direct
    for row in _UNIVERSE:
        for alias in row["aliases"]:
            if len(alias) < 4:
                continue
            if alias.casefold() in folded:
                return row["symbol"]
    best_symbol = ""
    best_score = 0.0
    for row in _UNIVERSE:
        for alias in row["aliases"]:
            if len(alias) < 8:
                continue
            score = SequenceMatcher(None, alias.casefold(), folded).ratio()
            if score > best_score:
                best_score = score
                best_symbol = row["symbol"]
    if best_score >= 0.84:
        return best_symbol
    return ""


def _hit(query: str, alias: str) -> bool:
    alias = alias.casefold()
    if alias.startswith(query):
        return True
    return len(query) >= 3 and query in alias


def _rank(query: str, item: dict) -> tuple:
    symbol = item["symbol"].casefold()
    name = item["name"].casefold()
    code = symbol.replace(".ca", "")
    exact = 0 if query in (code, name) else 1
    prefix = 0 if code.startswith(query) or name.startswith(query) else 1
    return (exact, prefix, name)


def _text(value) -> str:
    if isinstance(value, dict):
        return str(value.get("fmt") or value.get("raw") or "").strip()
    return str(value or "").strip()
