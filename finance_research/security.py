"""Prompt-injection defences.

Three layers, each cheap and explainable:

1. User input: block questions that try to override the assistant's rules.
2. Untrusted content (10-K text from the knowledge base): neutralise instruction-like
   phrases and wrap the text in markers the system prompt tells the model to treat as data.
3. Output: stop the answer if it reproduces the system prompt.

Pattern matching cannot catch every attack, so these checks sit alongside the system
prompt's rules and the fact that every tool is read-only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

UNTRUSTED_OPEN = "<<untrusted_external_text>>"
UNTRUSTED_CLOSE = "<</untrusted_external_text>>"
REDACTION = "[removed: instruction-like text]"

INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "override previous instructions",
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}\b"
            r"(previous|prior|above|earlier|all|your|system|these)\b[^.\n]{0,30}\b"
            r"(instructions?|rules?|prompts?|guidelines?|directions?)",
            re.I,
        ),
    ),
    (
        "reveal hidden prompt",
        re.compile(
            r"\b(reveal|show|print|repeat|output|display|tell me|what (is|are))\b[^.\n]{0,40}\b"
            r"(system|hidden|initial|original|developer)\s+(prompt|instructions?|message|rules?)",
            re.I,
        ),
    ),
    (
        "role reassignment",
        re.compile(
            r"\b(you are now|from now on,? you (are|will)|act as (an? )?(unrestricted|unfiltered|jailbroken)"
            r"|pretend (to be|you are)|new persona)\b",
            re.I,
        ),
    ),
    ("jailbreak mode", re.compile(r"\b(DAN mode|developer mode|jailbreak|do anything now)\b", re.I)),
    (
        "fake chat role marker",
        re.compile(r"(^|\n)\s*(system|assistant|developer)\s*:|<\|?(im_start|system)\|?>|\[/?INST\]", re.I),
    ),
    (
        "delimiter escape",
        re.compile(re.escape(UNTRUSTED_CLOSE) + "|" + re.escape(UNTRUSTED_OPEN), re.I),
    ),
]


@dataclass
class ScanResult:
    flagged: bool
    reasons: list[str] = field(default_factory=list)


def scan_text(text: str) -> ScanResult:
    reasons = [label for label, pattern in INJECTION_PATTERNS if pattern.search(text or "")]
    return ScanResult(flagged=bool(reasons), reasons=reasons)


class PromptInjectionError(ValueError):
    """Raised when a user question looks like an attempt to override the assistant's rules."""


def check_user_question(question: str) -> None:
    result = scan_text(question)
    if result.flagged:
        raise PromptInjectionError(
            "This question was blocked because it looks like an attempt to change the assistant's "
            f"rules ({', '.join(result.reasons)}). Rephrase it as a research question about a company."
        )


def sanitize_untrusted(text: str) -> tuple[str, list[str]]:
    """Redact instruction-like phrases in retrieved content and wrap it as data.

    Returns the safe text and the reasons anything was removed.
    """
    cleaned = text or ""
    reasons: list[str] = []
    for label, pattern in INJECTION_PATTERNS:
        cleaned, count = pattern.subn(REDACTION, cleaned)
        if count:
            reasons.append(label)
    return f"{UNTRUSTED_OPEN}\n{cleaned}\n{UNTRUSTED_CLOSE}", reasons


def strip_untrusted_markers(text: str) -> str:
    """Remove the wrapper markers for display to people."""
    return (text or "").replace(UNTRUSTED_OPEN, "").replace(UNTRUSTED_CLOSE, "").strip()


def leaks_system_prompt(answer: str, system_prompt: str, window: int = 60) -> bool:
    """True when the answer contains a long verbatim slice of the system prompt."""
    normalized_answer = " ".join((answer or "").split()).lower()
    normalized_prompt = " ".join(system_prompt.split()).lower()
    if len(normalized_prompt) < window:
        return normalized_prompt in normalized_answer
    step = window // 2
    return any(
        normalized_prompt[start : start + window] in normalized_answer
        for start in range(0, len(normalized_prompt) - window + 1, step)
    )
