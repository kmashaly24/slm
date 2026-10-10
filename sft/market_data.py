"""yfinance helpers for EGX daily candles and target-date calendars."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import requests
import pandas as pd
import yfinance as yf

from sft.config import (
    CUTOFF_LAG_DAYS,
    DATE_SPACING_TRADING_DAYS,
    EARLIEST_START,
    FUTURE_DAYS,
    LOOKBACK_CANDLES,
    MAX_DATES_PER_TICKER,
)


def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def _normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.rename(columns=str.title)
    needed = [col for col in ("Open", "High", "Low", "Close", "Volume") if col in frame.columns]
    frame = frame[needed].dropna(subset=["Close"])
    # Yahoo daily stamps are session times, not midnight. Collapse to the UTC date.
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(None).normalize()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame.sort_index()


def _download_chart(ticker: str, start: str) -> pd.DataFrame:
    """Yahoo chart endpoint. yfinance 0.2.43 fails because its crumb request comes back empty."""
    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
    end_ts = int(pd.Timestamp.now(tz="UTC").timestamp()) + 86400
    response = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
        params={"period1": start_ts, "period2": end_ts, "interval": "1d"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()["chart"]["result"]
    if not result:
        return pd.DataFrame()
    payload = result[0]
    stamps = payload.get("timestamp") or []
    quote = (payload.get("indicators") or {}).get("quote") or [{}]
    bars = quote[0]
    if not stamps or not bars.get("close"):
        return pd.DataFrame()
    frame = pd.DataFrame(
        {
            "Open": bars.get("open"),
            "High": bars.get("high"),
            "Low": bars.get("low"),
            "Close": bars.get("close"),
            "Volume": bars.get("volume"),
        },
        index=pd.to_datetime(stamps, unit="s"),
    )
    return _normalize_ohlcv(frame)


def download_history(ticker: str, start: str | None = None) -> pd.DataFrame:
    """Daily OHLCV indexed by date. Empty frame if Yahoo has no bars."""
    start = start or EARLIEST_START
    try:
        frame = _download_chart(ticker, start)
        if not frame.empty:
            return frame
    except (requests.RequestException, ValueError, KeyError, IndexError):
        frame = pd.DataFrame()
    fallback = yf.download(
        ticker,
        start=start,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    return _normalize_ohlcv(fallback)


def trading_days(frame: pd.DataFrame) -> list[date]:
    return [_as_date(ts) for ts in frame.index]


def cutoff_date(today: date | None = None) -> date:
    today = today or date.today()
    return today - timedelta(days=CUTOFF_LAG_DAYS)


def select_target_dates(
    frame: pd.DataFrame,
    start: date,
    end: date,
    spacing: int = DATE_SPACING_TRADING_DAYS,
    limit: int = MAX_DATES_PER_TICKER,
) -> list[date]:
    """Up to `limit` dates in [start, end], spaced and ending at the latest eligible day.

    The latest day is the last session on or before `end` that still has 60 past
    bars and 5 following closes. Earlier dates step back by `spacing` sessions.
    """
    days = trading_days(frame)
    if len(days) < LOOKBACK_CANDLES + FUTURE_DAYS + 1:
        return []
    eligible: list[tuple[int, date]] = []
    for index, day in enumerate(days):
        if day < start or day > end:
            continue
        if index < LOOKBACK_CANDLES or index + FUTURE_DAYS >= len(days):
            continue
        eligible.append((index, day))
    chosen: list[date] = []
    last_index: int | None = None
    for index, day in reversed(eligible):
        if last_index is not None and last_index - index < spacing:
            continue
        chosen.append(day)
        last_index = index
        if len(chosen) >= limit:
            break
    chosen.reverse()
    return chosen


def past_candles(frame: pd.DataFrame, target: date, lookback: int = LOOKBACK_CANDLES) -> pd.DataFrame:
    days = trading_days(frame)
    if target not in days:
        return pd.DataFrame()
    index = days.index(target)
    if index < lookback:
        return pd.DataFrame()
    return frame.iloc[index - lookback : index]


def future_closes(frame: pd.DataFrame, target: date, horizon: int = FUTURE_DAYS) -> list[tuple[date, float]]:
    days = trading_days(frame)
    if target not in days:
        return []
    index = days.index(target)
    if index + horizon >= len(days):
        return []
    out = []
    for offset in range(1, horizon + 1):
        row = frame.iloc[index + offset]
        close = row["Close"]
        if pd.isna(close):
            return []
        out.append((days[index + offset], float(close)))
    return out


def summarize_candles(frame: pd.DataFrame) -> str:
    """Compact past-window summary plus the last 10 daily closes."""
    if frame.empty:
        return ""
    closes = frame["Close"].astype(float)
    first = float(closes.iloc[0])
    last = float(closes.iloc[-1])
    change = (last / first - 1.0) * 100.0 if first else 0.0
    high = float(frame["High"].astype(float).max()) if "High" in frame else float(closes.max())
    low = float(frame["Low"].astype(float).min()) if "Low" in frame else float(closes.min())
    tail = closes.tail(10)
    tail_txt = ", ".join(f"{pd.Timestamp(idx).date().isoformat()}: {val:.2f}" for idx, val in tail.items())
    return (
        f"60-session window ending the session before T. "
        f"Start close {first:.2f}, last close {last:.2f}, change {change:+.1f}%. "
        f"High {high:.2f}, low {low:.2f}. "
        f"Last 10 closes: {tail_txt}."
    )


def format_future(path: list[tuple[date, float]]) -> str:
    return "; ".join(f"T+{i} {day.isoformat()} close {price:.2f}" for i, (day, price) in enumerate(path, start=1))
