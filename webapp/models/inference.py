"""Load the merged GRPO model once with Unsloth and answer the trained prompts."""

from __future__ import annotations

import os

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["USE_TORCH"] = "1"

import gc
import re
import threading

import torch
from unsloth import FastLanguageModel

from grpo.config import GRPO_LABELS
from grpo.prompts import GRPO_SYSTEM_PROMPT, build_grpo_user_message
from sft.prompts import SYSTEM_PROMPT, TASK_QUESTIONS, build_user_message
from webapp.config import (
    GRPO_MAX_NEW_TOKENS,
    MAX_PROMPT_TOKENS,
    MAX_SEQ_LENGTH,
    MODEL_DIR,
    SFT_MAX_NEW_TOKENS,
)

_LOCK = threading.Lock()
_MODEL = None
_TOKENIZER = None
_LOAD_IN_4BIT = False

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

SELECT_SYSTEM = (
    "You are an Egyptian Stock Exchange analyst. Reason from the supplied "
    "price history and news only. Write in English. Do not invent headlines. "
    "Put the reasoning inside <think></think> and the kept headline numbers inside "
    "<answer></answer>. The answer must be a comma-separated list of numbers, or none."
)
SELECT_MAX_NEW_TOKENS = 640


def select_news(
    symbol: str,
    company: str,
    sector: str,
    analysis_date: str,
    candle_summary: str,
    news_block: str,
) -> str:
    """Keep the candidate headlines that can affect this stock. Drop the think block."""
    rows = _headline_rows(news_block)
    numbered = "\n".join(
        f"{index}. [{bucket}] {text}" for index, (bucket, text) in enumerate(rows, start=1)
    )
    user = (
        f"Ticker: {symbol} ({company})\n"
        f"Sector: {sector or 'unknown'}\n"
        f"Analysis date: {analysis_date}\n"
        f"Price summary: {candle_summary}\n\n"
        f"Headlines:\n{numbered}\n\n"
        "Example:\n"
        "<think>food and inflation can move this company; unrelated telecom does not</think>"
        "<answer>1, 6</answer>\n\n"
        "Choose the headline numbers that can affect this stock. "
        "Reply in exactly this form and nothing else: "
        "<think>your reasoning</think><answer>1, 4</answer>. "
        "The answer tag must contain only the kept numbers, separated by commas, or none."
    )
    raw = _generate(
        [
            {"role": "system", "content": SELECT_SYSTEM},
            {"role": "user", "content": user},
        ],
        SELECT_MAX_NEW_TOKENS,
        stop_text="</answer>",
    )
    numbers = _select_indexes(raw, rows)
    if numbers is None:
        return news_block
    return _render_kept(rows, numbers)


def run_all(
    symbol: str,
    company: str,
    analysis_date: str,
    candle_summary: str,
    news_block: str,
    horizon: int,
) -> list[dict]:
    """Direction and macro answers, closed on a sentence, plus the GRPO label."""
    results = []
    for task in ("A", "C"):
        user = build_user_message(symbol, company, analysis_date, candle_summary, news_block, task)
        text = _generate_sentence(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            SFT_MAX_NEW_TOKENS,
        )
        results.append({"key": task, "question": TASK_QUESTIONS[task], "response": text})

    grpo_user = build_grpo_user_message(
        symbol, company, analysis_date, candle_summary, news_block, horizon
    )
    raw = _generate(
        [
            {"role": "system", "content": GRPO_SYSTEM_PROMPT},
            {"role": "user", "content": grpo_user},
        ],
        GRPO_MAX_NEW_TOKENS,
    )
    question = f"Predict the return class of this stock over the next {horizon} trading days."
    results.append({"key": "GRPO", "question": question, "response": _label_only(raw)})
    return results


def _generate(messages: list[dict], max_new_tokens: int | None, stop_text: str | None = None) -> str:
    text, _count = _generate_counted(messages, max_new_tokens, stop_text)
    return text


def _generate_sentence(messages: list[dict], max_new_tokens: int) -> str:
    """Stop at the token cap, then keep text through the last sentence period."""
    text, count = _generate_counted(messages, max_new_tokens)
    text = text.strip()
    if count >= max_new_tokens:
        return _close_at_sentence(text)
    return text


def _close_at_sentence(text: str) -> str:
    last = -1
    for index, char in enumerate(text):
        if char != ".":
            continue
        before = text[index - 1].isdigit() if index else False
        after = text[index + 1].isdigit() if index + 1 < len(text) else False
        if before and after:
            continue
        last = index
    if last < 0:
        return text
    return text[: last + 1].strip()


def _generate_counted(
    messages: list[dict], max_new_tokens: int | None, stop_text: str | None = None
) -> tuple[str, int]:
    tokenizer, model = _get_model()
    try:
        return _generate_with(tokenizer, model, messages, max_new_tokens, stop_text)
    except Exception as exc:
        if _LOAD_IN_4BIT or not torch.cuda.is_available() or not _is_oom(exc):
            raise
        _reload_4bit()
        tokenizer, model = _get_model()
        return _generate_with(tokenizer, model, messages, max_new_tokens, stop_text)


def _generate_with(
    tokenizer, model, messages: list[dict], max_new_tokens: int | None, stop_text: str | None = None
) -> tuple[str, int]:
    prompt = _fit_messages(tokenizer, messages)
    input_ids = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if input_ids.shape[-1] > MAX_PROMPT_TOKENS:
        input_ids = input_ids[:, -MAX_PROMPT_TOKENS:]
    input_ids = input_ids.to(model.device)
    if max_new_tokens is None:
        max_new_tokens = max(1, MAX_SEQ_LENGTH - int(input_ids.shape[-1]))
    attention_mask = torch.ones_like(input_ids)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    end_of_text = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    stopping = _stop_on(tokenizer, input_ids.shape[-1], stop_text)
    with _LOCK:
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=[token for token in (im_end, end_of_text) if isinstance(token, int)],
                pad_token_id=end_of_text if isinstance(end_of_text, int) else None,
                stopping_criteria=stopping,
            )
    new_tokens = output[0, input_ids.shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text, int(new_tokens.shape[-1])


def _get_model():
    global _MODEL, _TOKENIZER, _LOAD_IN_4BIT
    if _MODEL is not None and _TOKENIZER is not None:
        return _TOKENIZER, _MODEL
    with _LOCK:
        if _MODEL is not None and _TOKENIZER is not None:
            return _TOKENIZER, _MODEL
        if not MODEL_DIR.joinpath("config.json").exists():
            raise FileNotFoundError(f"Merged model not found at {MODEL_DIR}")
        if not torch.cuda.is_available():
            _MODEL, _TOKENIZER = _load_unsloth(False)
            _LOAD_IN_4BIT = False
            return _TOKENIZER, _MODEL
        try:
            _MODEL, _TOKENIZER = _load_unsloth(False)
            _LOAD_IN_4BIT = False
        except Exception as exc:
            if not _is_oom(exc):
                raise
            _release_partial()
            _MODEL, _TOKENIZER = _load_unsloth(True)
            _LOAD_IN_4BIT = True
        return _TOKENIZER, _MODEL


def _reload_4bit() -> None:
    global _MODEL, _TOKENIZER, _LOAD_IN_4BIT
    with _LOCK:
        _release_partial()
        _LOAD_IN_4BIT = True
        _MODEL, _TOKENIZER = _load_unsloth(True)


def _load_unsloth(load_in_4bit: bool):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(MODEL_DIR),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_new_tokens = None
        model.generation_config.max_length = MAX_PROMPT_TOKENS + GRPO_MAX_NEW_TOKENS
    return model, tokenizer


def _release_partial() -> None:
    global _MODEL, _TOKENIZER
    _MODEL = None
    _TOKENIZER = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _fit_messages(tokenizer, messages: list[dict]) -> list[dict]:
    """Shorten the news block until the chat prompt fits the training cap."""
    fitted = [dict(message) for message in messages]
    user = fitted[-1]["content"]
    marker = "\n\nNews from T-7 through T:\n"
    while _token_count(tokenizer, fitted) > MAX_PROMPT_TOKENS and marker in user:
        head, _, tail = user.partition(marker)
        news, task_marker, task = tail.partition("\n\nTask: ")
        if len(news) < 80:
            return fitted
        shortened = news[: int(len(news) * 0.75)].rsplit("\n", 1)[0].strip()
        if not shortened or shortened == news:
            return fitted
        user = f"{head}{marker}{shortened}{task_marker}{task}"
        fitted[-1]["content"] = user
    return fitted


def _token_count(tokenizer, messages: list[dict]) -> int:
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "shape"):
        return int(ids.shape[-1])
    return len(ids)


def _headline_rows(news_block: str) -> list[tuple[str, str]]:
    bucket = "corporate"
    rows = []
    for line in (news_block or "").splitlines():
        stripped = line.strip()
        if stripped.endswith(":") and not stripped.startswith("-"):
            bucket = stripped[:-1].strip().lower()
            continue
        if stripped.startswith("- "):
            rows.append((bucket, stripped[2:].strip()))
    return rows


_NEG_NEAR = re.compile(
    r"irrelevant|not relevant|unrelated|does not affect|do not affect|won't affect|no effect|not about",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z]{4,}")
_STOP_WORDS = {
    "egypt", "egyptian", "with", "from", "that", "this", "after", "more", "week",
    "news", "says", "said", "into", "over", "their", "about",
}


def _select_indexes(raw: str, rows: list[tuple[str, str]]) -> list[int] | None:
    match = _ANSWER_RE.search(raw or "")
    if match:
        numbers = _kept_numbers(match.group(1), len(rows))
        if numbers is not None:
            return numbers
    mentioned = _from_commentary(raw or "", rows)
    if mentioned:
        return mentioned
    return None


def _from_commentary(raw: str, rows: list[tuple[str, str]]) -> list[int]:
    kept = []
    for index, (_bucket, headline) in enumerate(rows, start=1):
        pos, phrase = _mention(raw, headline)
        if pos < 0:
            continue
        window = raw[max(0, pos - 140) : pos + len(phrase) + 140]
        if _NEG_NEAR.search(window):
            continue
        kept.append(index)
    return kept


def _mention(raw: str, headline: str) -> tuple[int, str]:
    undated = re.sub(r"\s*\[[^\]]*\]\s*$", "", headline).strip()
    title = re.sub(r"\s*\([^)]*\)\s*", " ", undated)
    title = re.sub(r"\s+", " ", title).strip()
    words = [word for word in _WORD_RE.findall(title) if word.lower() not in _STOP_WORDS]
    if len(words) >= 3:
        pattern = r".{0,30}".join(re.escape(word) for word in words[:4])
        found = re.search(pattern, raw, flags=re.IGNORECASE)
        if found:
            return found.start(), found.group(0)
    source = re.search(r"\(([^)]+)\)", undated)
    if source and len(source.group(1)) >= 4:
        pos = raw.lower().find(source.group(1).lower())
        if pos >= 0:
            return pos, source.group(1)
    if len(title) >= 8:
        pos = raw.lower().find(title.lower())
        if pos >= 0:
            return pos, title
    return -1, ""


def _kept_numbers(body: str, limit: int) -> list[int] | None:
    if not body or limit <= 0:
        return None
    if re.fullmatch(r"none\.?", body.strip(), flags=re.IGNORECASE):
        return []
    if re.search(r"[A-Za-z]{3,}", body):
        return None
    numbers = []
    for token in re.findall(r"\d+", body):
        value = int(token)
        if 1 <= value <= limit and value not in numbers:
            numbers.append(value)
    return numbers or None


def _render_kept(rows: list[tuple[str, str]], numbers: list[int]) -> str:
    grouped = {"corporate": [], "sector": [], "macro": []}
    for number in numbers:
        bucket, text = rows[number - 1]
        if bucket in grouped and text not in grouped[bucket]:
            grouped[bucket].append(text)
    blocks = []
    for bucket in ("corporate", "sector", "macro"):
        body = "\n".join(f"- {line}" for line in grouped[bucket]) or "none in this window."
        blocks.append(f"{bucket.capitalize()}:\n{body}")
    return "\n".join(blocks)


def _stop_on(tokenizer, prompt_length: int, stop_text: str | None):
    if not stop_text:
        return None
    needle = tokenizer.encode(stop_text, add_special_tokens=False)
    if not needle:
        return None
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _StopAfter(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            generated = input_ids[0, prompt_length:].tolist()
            return len(generated) >= len(needle) and generated[-len(needle) :] == needle

    return StoppingCriteriaList([_StopAfter()])


def _answer_body(text: str) -> str:
    match = _ANSWER_RE.search(text or "")
    body = match.group(1) if match else _THINK_RE.sub("", text or "")
    return body.strip()


def _label_only(text: str) -> str:
    """Return one allowed class, even when the completion surrounds it with reasoning."""
    match = _ANSWER_RE.search(text or "")
    if match:
        found = _find_label(match.group(1))
        if found:
            return found
    return _find_label(text or "") or "No label returned"


def _find_label(text: str) -> str:
    best_at = -1
    best = ""
    folded = text.lower()
    for label in sorted(GRPO_LABELS, key=len, reverse=True):
        needle = label.lower()
        start = 0
        while True:
            pos = folded.find(needle, start)
            if pos < 0:
                break
            if pos > best_at:
                best_at = pos
                best = label
            start = pos + len(needle)
    return best
