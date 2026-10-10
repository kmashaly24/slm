"""Regex rewards for the GRPO return-class format.

Completions may be plain strings or the conversational shape TRL passes
through: a list of {"role": "assistant", "content": "..."} messages.
"""

from __future__ import annotations

import re

FORMAT_RE = re.compile(r"^<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>$")
ANSWER_RE = re.compile(r"<answer>([\s\S]*?)</answer>")
THINK_BODY_RE = re.compile(r"<think>([\s\S]*?)</think>")
SPACE_RE = re.compile(r"\s+")
TAG_COUNTS = (
    ("<think>", re.compile(r"<think>")),
    ("</think>", re.compile(r"</think>")),
    ("<answer>", re.compile(r"<answer>")),
    ("</answer>", re.compile(r"</answer>")),
)


def completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content") or "")
    if isinstance(completion, list):
        parts = []
        for message in completion:
            if isinstance(message, dict):
                parts.append(str(message.get("content") or ""))
            else:
                parts.append(str(message))
        return "".join(parts)
    return str(completion)


def think_length(completion, tokenizer=None) -> int:
    """Token count inside <think> tags. Zero when the tag is missing. Words if no tokenizer."""
    bodies = [part.strip() for part in THINK_BODY_RE.findall(completion_text(completion))]
    body = "\n".join(part for part in bodies if part)
    if not body:
        return 0
    if tokenizer is None:
        return len(body.split())
    return len(tokenizer.encode(body, add_special_tokens=False))


def format_reward_func(completions, **kwargs) -> list[float]:
    """1.0 for the exact think/answer pattern, otherwise 0.2 per tag piece.

    Pieces are: starts with <think>, contains </think>, contains <answer>,
    ends with </answer>. Four pieces score 0.8. Only the exact pattern scores 1.0.
    """
    rewards = []
    for completion in completions:
        text = completion_text(completion).strip()
        if FORMAT_RE.match(text):
            rewards.append(1.0)
            continue
        score = 0.0
        if text.startswith("<think>"):
            score += 0.2
        if "</think>" in text:
            score += 0.2
        if "<answer>" in text:
            score += 0.2
        if text.endswith("</answer>"):
            score += 0.2
        rewards.append(score)
    return rewards


def xml_count_reward_func(completions, **kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        text = completion_text(completion)
        counts = [len(pattern.findall(text)) for _name, pattern in TAG_COUNTS]
        rewards.append(0.5 if counts == [1, 1, 1, 1] else 0.0)
    return rewards


def accuracy_reward_func(completions, target_answer, **kwargs) -> list[float]:
    if isinstance(target_answer, str):
        targets = [target_answer] * len(completions)
    else:
        targets = list(target_answer)
    rewards = []
    for completion, target in zip(completions, targets):
        match = ANSWER_RE.search(completion_text(completion))
        if match is None:
            rewards.append(0.0)
            continue
        got = SPACE_RE.sub(" ", match.group(1)).strip()
        want = SPACE_RE.sub(" ", str(target)).strip()
        rewards.append(2.0 if got == want else 0.0)
    return rewards
