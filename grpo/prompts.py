"""GRPO prompt: SFT context layout, one direction task, strict XML answer."""

from __future__ import annotations

from grpo.config import GRPO_LABELS

GRPO_SYSTEM_PROMPT = (
    "You are an Egyptian Stock Exchange analyst. Reason from the supplied "
    "price history and news only. Write in English. Do not invent numbers "
    "that are not supported by the context. "
    "Put the reasoning inside <think></think> and the return class inside "
    "<answer></answer>. The answer must be exactly one of: "
    "Strong Bullish, Slight Bullish, Neutral, Slight Bearish, Strong Bearish."
)


def build_grpo_user_message(
    ticker: str,
    company: str,
    target: str,
    candle_summary: str,
    news_block: str,
    horizon: int,
) -> str:
    labels = ", ".join(GRPO_LABELS)
    task = (
        f"Predict the return class of this stock over the next {horizon} trading days. "
        f"The allowed labels are: {labels}. "
        "Reply in exactly this form and nothing else: "
        "<think>your reasoning</think><answer>Label</answer>. "
        "The answer tag must contain one allowed label and no other text."
    )
    return (
        f"Ticker: {ticker} ({company})\n"
        f"Analysis date T: {target}\n\n"
        f"Recent price history (sessions strictly before T):\n{candle_summary}\n\n"
        f"News from T-7 through T:\n{news_block}\n\n"
        f"Task: {task}"
    )
