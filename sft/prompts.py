"""Hidden hindsight prompts and the user-visible ChatML prompt.

The user message must never contain the T+1..T+5 prices. Those closes are
passed only to Gemini so it can relabel the assistant reasoning.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are an Egyptian Stock Exchange analyst. Reason from the supplied "
    "price history and news only. Write in English. Do not invent numbers "
    "that are not supported by the context. Do not say you were shown future prices."
)

TASK_QUESTIONS = {
    "A": (
        "Predict the likely direction of this stock over the next five trading days "
        "and explain the reasoning step by step."
    ),
    "B": (
        "Identify the main risk factors for this stock and classify the near-term "
        "sentiment as Bullish or Bearish."
    ),
    "C": (
        "Explain how the current macroeconomic and sector news is likely to affect "
        "this specific stock."
    ),
}

TASK_HIDDEN = {
    "A": (
        "Write a chain-of-thought forecast of the next five trading days. "
        "The trajectory you were given is the realized path. Your reasoning must "
        "be consistent with that path (direction and whether the move extends or fades), "
        "but phrase it as a forecast made on date T. End with a one-line direction: "
        "Up, Down, or Flat over five sessions."
    ),
    "B": (
        "Write a risk analysis consistent with the realized five-day path. "
        "Classify sentiment as Bullish if the path finishes higher, Bearish if it "
        "finishes lower, otherwise Mixed. Phrase it as analysis available on date T. "
        "End with a line: Sentiment: Bullish or Bearish or Mixed."
    ),
    "C": (
        "Explain the macro and sector transmission into this stock so that it "
        "accounts for the realized five-day path, phrased as analysis on date T. "
        "Do not quote the future prices."
    ),
}


def build_user_message(ticker: str, company: str, target: str, candle_summary: str, news_block: str, task: str) -> str:
    return (
        f"Ticker: {ticker} ({company})\n"
        f"Analysis date T: {target}\n\n"
        f"Recent price history (sessions strictly before T):\n{candle_summary}\n\n"
        f"News from T-7 through T:\n{news_block}\n\n"
        f"Task: {TASK_QUESTIONS[task]}"
    )


def build_hidden_prompt(
    ticker: str,
    company: str,
    target: str,
    candle_summary: str,
    news_block: str,
    future_block: str,
    task: str,
) -> str:
    user = build_user_message(ticker, company, target, candle_summary, news_block, task)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Hidden realized outcome, for your reasoning only. Never mention these "
        "prices, never say they are future data, and never quote them:\n"
        f"{future_block}\n\n"
        f"{TASK_HIDDEN[task]}\n\n"
        "The user message you are answering is:\n"
        f"{user}"
    )
