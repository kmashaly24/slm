"""Build a resumable EGX SFT dataset with Gemini hindsight relabeling.

Run from the repository root:

    python -m sft.creating_sft_data

If a Gemini key runs out of quota, the process prints an error and exits.
Put a new key in .env and rerun; finished examples are skipped.
"""

from __future__ import annotations

import json
import random
import sys
from datetime import date, timedelta

from sft.config import (
    BACKUP_TICKERS,
    DATASET_PATH,
    EARLIEST_START,
    LOOKBACK_CANDLES,
    NEWS_LOOKBACK_DAYS,
    PRIMARY_TICKERS,
    SEED,
    TARGET_EXAMPLES,
    TASKS,
    TICKER_META,
)
from sft.gemini_client import GeminiClient, QuotaExhausted
from sft.market_data import (
    cutoff_date,
    download_history,
    format_future,
    future_closes,
    past_candles,
    select_target_dates,
    summarize_candles,
)
from sft.news_fetch import (
    fetch_window,
    format_news,
    has_old_news_layout,
    probe_start,
    translate_news_to_english,
    window_is_usable,
)
from sft.prompts import SYSTEM_PROMPT, build_hidden_prompt, build_user_message


def _task_for(ticker: str, day: date) -> str:
    rng = random.Random(f"{SEED}:{ticker}:{day.isoformat()}")
    return rng.choice(TASKS)


def _load_done(path) -> set[tuple[str, str, str]]:
    done: set[tuple[str, str, str]] = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add((row["ticker"], row["date"], row["task"]))
    return done


def _append(path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _ticker_jobs(ticker: str, done: set[tuple[str, str, str]]) -> list[dict]:
    print(f"\n=== {ticker} ===")
    earliest = date.fromisoformat(EARLIEST_START)
    # The first target still needs 60 sessions before it.
    history_start = earliest - timedelta(days=LOOKBACK_CANDLES * 3)
    history = download_history(ticker, history_start.isoformat())
    if history.empty:
        print("  no yfinance history; skipping ticker")
        return []
    end = cutoff_date()
    start = probe_start(ticker, earliest, end)
    if start is None:
        print("  no reliable recent news; skipping ticker")
        return []
    print(f"  adaptive news start: {start.isoformat()}  cutoff: {end.isoformat()}")
    dates = select_target_dates(history, earliest, end)
    print(f"  candidate dates: {len(dates)}")
    jobs = []
    arabic, english, _sector = TICKER_META[ticker]
    company = f"{english} / {arabic}"
    for target in dates:
        task = _task_for(ticker, target)
        key = (ticker, target.isoformat(), task)
        if key in done:
            jobs.append({"skip": True, "key": key})
            continue
        candles = past_candles(history, target)
        future = future_closes(history, target)
        news_start = target - timedelta(days=NEWS_LOOKBACK_DAYS)
        news = fetch_window(ticker, news_start, target)
        if candles.empty or len(future) < 5 or not window_is_usable(news):
            print(f"  drop {target.isoformat()} (incomplete market data or news)")
            continue
        jobs.append(
            {
                "skip": False,
                "key": key,
                "ticker": ticker,
                "company": company,
                "target": target,
                "task": task,
                "candles": summarize_candles(candles),
                "news": format_news(news),
                "future": format_future(future),
            }
        )
    return jobs


def build() -> int:
    if has_old_news_layout(DATASET_PATH):
        print(
            f"{DATASET_PATH} uses the old news layout. Delete that file before "
            "rerunning so new examples are not mixed with the old headlines."
        )
        return 2
    done = _load_done(DATASET_PATH)
    written = len(done)
    print(f"Already saved: {written} examples. Target: {TARGET_EXAMPLES}.")
    if written >= TARGET_EXAMPLES:
        print("Target already reached.")
        return 0

    client = GeminiClient()
    order = list(PRIMARY_TICKERS)
    backup_index = 0
    try:
        while written < TARGET_EXAMPLES:
            if not order:
                if backup_index >= len(BACKUP_TICKERS):
                    print(
                        f"Stopped at {written} examples. Primary and backup tickers "
                        "are exhausted before the 4,000 target."
                    )
                    return 0
                ticker = BACKUP_TICKERS[backup_index]
                backup_index += 1
                print(f"Expanding universe with backup ticker {ticker}.")
            else:
                ticker = order.pop(0)
            for job in _ticker_jobs(ticker, done):
                if written >= TARGET_EXAMPLES:
                    break
                if job["skip"]:
                    continue
                job["news"] = translate_news_to_english(client, job["news"])
                prompt = build_hidden_prompt(
                    job["ticker"],
                    job["company"],
                    job["target"].isoformat(),
                    job["candles"],
                    job["news"],
                    job["future"],
                    job["task"],
                )
                answer = client.generate(prompt)
                user = build_user_message(
                    job["ticker"],
                    job["company"],
                    job["target"].isoformat(),
                    job["candles"],
                    job["news"],
                    job["task"],
                )
                row = {
                    "ticker": job["ticker"],
                    "date": job["target"].isoformat(),
                    "task": job["task"],
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": answer},
                    ],
                }
                _append(DATASET_PATH, row)
                done.add(job["key"])
                written += 1
                print(f"  saved {written}/{TARGET_EXAMPLES} {job['key']}")
    except QuotaExhausted as exc:
        print("\n=== Gemini quota stopped generation ===")
        print(exc)
        print(f"Examples on disk: {written}. File: {DATASET_PATH}")
        print("Add a fresh key to GEMINI_API_KEYS in .env and rerun this command.")
        return 2
    print(f"\nFinished with {written} examples at {DATASET_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(build())
