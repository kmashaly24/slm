"""Latest 60 EGX sessions for the chart and the model prompt."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from sft.config import LOOKBACK_CANDLES
from sft.market_data import download_history, summarize_candles


def get_window(symbol: str) -> dict:
    """Most recent sessions through the latest bar. T is that last session."""
    history_start = (date.today() - timedelta(days=LOOKBACK_CANDLES * 3)).isoformat()
    frame = download_history(symbol, history_start)
    if frame.empty:
        raise LookupError(f"No price history for {symbol}")
    window = frame.tail(LOOKBACK_CANDLES)
    analysis_date = pd.Timestamp(window.index[-1]).date()
    dates = [pd.Timestamp(idx).date().isoformat() for idx in window.index]
    closes = [round(float(value), 4) for value in window["Close"].astype(float)]
    return {
        "dates": dates,
        "closes": closes,
        "summary": summarize_candles(window),
        "analysis_date": analysis_date.isoformat(),
    }
