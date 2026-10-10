"""Gemini 2.5 Flash client with free-tier pacing and key rotation."""

from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv

from sft.config import GEMINI_MAX_RETRIES, GEMINI_MODEL, GEMINI_SLEEP_SECONDS, ROOT


class QuotaExhausted(RuntimeError):
    """Every configured Gemini key is out of quota."""


def load_api_keys(env_path: Path | None = None) -> list[str]:
    load_dotenv(env_path or (ROOT / ".env"))
    raw = os.environ.get("GEMINI_API_KEYS", "")
    keys = [part.strip() for part in raw.split(",") if part.strip() and not part.strip().startswith("your_")]
    if not keys:
        raise QuotaExhausted(
            "No Gemini API keys in .env. Set GEMINI_API_KEYS=key1,key2 and rerun. "
            "Completed examples already in sft/sft_dataset.jsonl will be skipped."
        )
    return keys


class GeminiClient:
    def __init__(self, keys: list[str] | None = None):
        self.keys = keys or load_api_keys()
        self.index = 0
        self._client = None
        self._bind()

    def _bind(self) -> None:
        from google import genai

        self._client = genai.Client(api_key=self.keys[self.index])

    def _rotate(self) -> None:
        self.index += 1
        if self.index >= len(self.keys):
            raise QuotaExhausted(
                "All Gemini API keys are exhausted. Add a new key to GEMINI_API_KEYS in .env "
                "and rerun the same command. Generation resumes from the last saved example."
            )
        print(f"Gemini quota hit. Switching to API key {self.index + 1} of {len(self.keys)}.")
        self._bind()

    @staticmethod
    def _is_quota(exc: Exception) -> bool:
        text = str(exc).lower()
        return "429" in text or "resource_exhausted" in text or "quota" in text

    def generate(self, prompt: str) -> str:
        last_error: Exception | None = None
        while True:
            for attempt in range(GEMINI_MAX_RETRIES):
                try:
                    response = self._client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=prompt,
                    )
                    text = (getattr(response, "text", None) or "").strip()
                    if not text:
                        raise RuntimeError("Gemini returned an empty response.")
                    time.sleep(GEMINI_SLEEP_SECONDS)
                    return text
                except QuotaExhausted:
                    raise
                except Exception as exc:
                    last_error = exc
                    if self._is_quota(exc):
                        self._rotate()
                        break
                    wait = GEMINI_SLEEP_SECONDS * (attempt + 1)
                    print(f"Gemini error ({exc.__class__.__name__}): {exc}. Retrying in {wait:.0f}s.")
                    time.sleep(wait)
            else:
                raise RuntimeError(f"Gemini failed after retries: {last_error}")
