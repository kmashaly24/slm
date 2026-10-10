"""Deterministic GRPO prompts for the 20 primary EGX tickers.

No language model is called. Each row is a price-and-news prompt plus the
return class realized from day T to day T+H.

    python -m grpo.grpo_dataset_gen
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from grpo.config import (
    GRPO_DATASET_PATH,
    GRPO_HORIZONS,
    GRPO_LABELS,
    GRPO_MAX_HORIZON,
    GRPO_SAMPLES_PER_TICKER,
    GRPO_SEED,
    GRPO_START,
)


def _load_generation_modules():
    """Import price and news helpers only while building prompts.

    Training imports this module for the date split and reads the finished
    jsonl, so those files are not required on Colab.
    """
    global pd
    global GRPO_SYSTEM_PROMPT, build_grpo_user_message
    global LOOKBACK_CANDLES, NEWS_LOOKBACK_DAYS, PRIMARY_TICKERS, TICKER_META
    global cutoff_date, download_history, past_candles, summarize_candles, trading_days
    global fetch_window, format_news
    import pandas as pd
    from grpo.prompts import GRPO_SYSTEM_PROMPT, build_grpo_user_message
    from sft.config import LOOKBACK_CANDLES, NEWS_LOOKBACK_DAYS, PRIMARY_TICKERS, TICKER_META
    from sft.market_data import cutoff_date, download_history, past_candles, summarize_candles, trading_days
    from sft.news_fetch import fetch_window, format_news


def label_for_return(return_pct: float) -> str:
    """Map a percent return onto one of the five classes.

    Strong Bullish:  > +5
    Slight Bullish:  +1 < return <= +5
    Neutral:         -1 <= return <= +1
    Slight Bearish:  -5 <= return < -1
    Strong Bearish:  < -5
    """
    if return_pct > 5.0:
        return GRPO_LABELS[0]
    if return_pct > 1.0:
        return GRPO_LABELS[1]
    if return_pct >= -1.0:
        return GRPO_LABELS[2]
    if return_pct >= -5.0:
        return GRPO_LABELS[3]
    return GRPO_LABELS[4]


def _load_done(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add((row["ticker"], row["date"]))
    return done


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _eligible_indexes(history: pd.DataFrame, start: date, end: date) -> list[int]:
    days = trading_days(history)
    eligible = []
    for index, day in enumerate(days):
        if day < start or day > end:
            continue
        if index < LOOKBACK_CANDLES or index + GRPO_MAX_HORIZON >= len(days):
            continue
        eligible.append(index)
    return eligible


def _close_at(history: pd.DataFrame, index: int) -> float | None:
    close = history.iloc[index]["Close"]
    if pd.isna(close):
        return None
    value = float(close)
    if value == 0.0:
        return None
    return value


def build_ticker(
    ticker: str,
    per_ticker: int,
    done: set[tuple[str, str]],
    path: Path,
    pause: float,
) -> int:
    _load_generation_modules()
    print(f"\n=== {ticker} ===", flush=True)
    start = date.fromisoformat(GRPO_START)
    history_start = start - timedelta(days=LOOKBACK_CANDLES * 3)
    history = download_history(ticker, history_start.isoformat())
    if history.empty:
        print("  no price history; skipping ticker", flush=True)
        return 0
    end = cutoff_date()
    eligible = _eligible_indexes(history, start, end)
    print(f"  eligible dates: {len(eligible)}  cutoff: {end.isoformat()}", flush=True)
    if not eligible:
        print("  none have 60 past bars and 10 future closes; skipping", flush=True)
        return 0

    rng = random.Random(f"{GRPO_SEED}:{ticker}")
    order = list(eligible)
    rng.shuffle(order)
    arabic, english, _sector = TICKER_META[ticker]
    company = f"{english} / {arabic}"
    days = trading_days(history)
    already = sum(1 for saved_ticker, _day in done if saved_ticker == ticker)
    written = 0
    for index in order:
        if already + written >= per_ticker:
            break
        target = days[index]
        key = (ticker, target.isoformat())
        if key in done:
            continue
        price_t = _close_at(history, index)
        # Horizon depends on the date, not on how many earlier dates were skipped,
        # so a resumed run assigns the same H.
        horizon = random.Random(f"{GRPO_SEED}:{ticker}:{target.isoformat()}").choice(GRPO_HORIZONS)
        price_th = _close_at(history, index + horizon)
        if price_t is None or price_th is None:
            print(f"  drop {target.isoformat()} (missing close)", flush=True)
            continue
        candles = past_candles(history, target)
        if candles.empty:
            print(f"  drop {target.isoformat()} (missing lookback)", flush=True)
            continue
        return_pct = (price_th - price_t) / price_t * 100.0
        label = label_for_return(return_pct)
        news_start = target - timedelta(days=NEWS_LOOKBACK_DAYS)
        news = format_news(fetch_window(ticker, news_start, target))
        user = build_grpo_user_message(
            ticker,
            company,
            target.isoformat(),
            summarize_candles(candles),
            news,
            horizon,
        )
        row = {
            "ticker": ticker,
            "date": target.isoformat(),
            "horizon": horizon,
            "return_pct": round(return_pct, 4),
            "target_answer": label,
            "prompt": [
                {"role": "system", "content": GRPO_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        }
        _append(path, row)
        done.add(key)
        written += 1
        print(
            f"  saved {already + written}/{per_ticker} {key} H={horizon} "
            f"{return_pct:+.2f}% -> {label}",
            flush=True,
        )
        if pause:
            time.sleep(pause)
    if already + written < per_ticker:
        print(
            f"  only {already + written} dates available (asked for {per_ticker})",
            flush=True,
        )
    return written


def split_by_recent_dates(rows: list[dict], fraction: float) -> tuple[list[dict], list[dict]]:
    """Hold out the most recent dates for each ticker.

    The cutoff is per ticker. Dates are sorted, and the last round(fraction)
    of them become validation. At least one date stays in training when a
    ticker has two or more dates.
    """
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["ticker"], []).append(row)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for ticker in sorted(grouped):
        group = sorted(grouped[ticker], key=lambda row: row["date"])
        dates = sorted({row["date"] for row in group})
        val_count = int(len(dates) * fraction + 0.5)
        if len(dates) >= 2:
            val_count = min(max(val_count, 1), len(dates) - 1)
        else:
            val_count = 0
        val_dates = set(dates[-val_count:]) if val_count else set()
        for row in group:
            if row["date"] in val_dates:
                val_rows.append(row)
            else:
                train_rows.append(row)
        if val_dates:
            print(
                f"  {ticker}: train {len(dates) - val_count} dates through {dates[-val_count - 1]}, "
                f"val {val_count} dates from {dates[-val_count]} to {dates[-1]}",
                flush=True,
            )
        else:
            print(f"  {ticker}: train {len(dates)} dates, no validation holdout", flush=True)
    return train_rows, val_rows


def build(
    tickers: list[str] | None = None,
    per_ticker: int = GRPO_SAMPLES_PER_TICKER,
    output: Path | None = None,
    pause: float = 0.4,
) -> int:
    _load_generation_modules()
    path = output or GRPO_DATASET_PATH
    tickers = list(tickers or PRIMARY_TICKERS)
    done = _load_done(path)
    print(f"Already saved: {len(done)} rows. Output: {path}", flush=True)
    written = 0
    for ticker in tickers:
        written += build_ticker(ticker, per_ticker, done, path, pause)
    counts: dict[str, int] = {}
    labels: dict[str, int] = {}
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                counts[row["ticker"]] = counts.get(row["ticker"], 0) + 1
                labels[row["target_answer"]] = labels.get(row["target_answer"], 0) + 1
    total = sum(counts.values())
    print(f"\nFinished. New rows this run: {written}. Rows in file: {total}.", flush=True)
    for ticker in tickers:
        print(f"  {ticker}: {counts.get(ticker, 0)}", flush=True)
    for label in GRPO_LABELS:
        print(f"  {label}: {labels.get(label, 0)}", flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the GRPO prompt dataset.")
    parser.add_argument(
        "--tickers",
        default="",
        help="Comma-separated tickers. Default: the 20 primary EGX tickers.",
    )
    parser.add_argument("--per-ticker", type=int, default=GRPO_SAMPLES_PER_TICKER)
    parser.add_argument("--output", default=str(GRPO_DATASET_PATH))
    args = parser.parse_args()
    tickers = [item.strip() for item in args.tickers.split(",") if item.strip()] or None
    sys.exit(build(tickers=tickers, per_ticker=args.per_ticker, output=Path(args.output)))


if __name__ == "__main__":
    main()
