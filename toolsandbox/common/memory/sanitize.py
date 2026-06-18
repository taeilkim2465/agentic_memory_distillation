# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""PII masking utilities for memory text."""

from __future__ import annotations

import re
from typing import Any

# Text-level patterns: applied to free-form strings (insights, results)
_TEXT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "<EMAIL>"),
    (re.compile(r"\b\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "<PHONE>"),
    (re.compile(r"\$\s?\d+(?:[,.]\d+)*"), "<AMOUNT>"),
    (re.compile(r"\b\d{6,}\b"), "<NUM>"),
]

# Argument-key patterns: mask by key name first, then fall back to value analysis
_ARG_KEY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"email|mail", re.I), "<EMAIL>"),
    (re.compile(r"password|passwd|pwd", re.I), "<PASSWORD>"),
    (re.compile(r"\bphone|mobile|cell\b", re.I), "<PHONE>"),
    (re.compile(r"\bdir(ectory)?|folder\b", re.I), "<DIR>"),
    (re.compile(r"file|path|src|dest|source|destination|filename", re.I), "<FILE_PATH>"),
    (re.compile(r"\bid\b|_id$", re.I), "<ID>"),
    (re.compile(r"token|secret|api_?key", re.I), "<TOKEN>"),
    (re.compile(r"\bname|contact\b", re.I), "<NAME>"),
    (re.compile(r"number|amount", re.I), "<VALUE>"),
]

# Value-level fallback patterns
_VALUE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "<EMAIL>"),
    (re.compile(r"\b\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "<PHONE>"),
    (re.compile(r"\b\d{6,}\b"), "<NUM>"),
]


def sanitize_text(text: str) -> str:
    """Replace PII tokens in free-form text."""
    for pattern, placeholder in _TEXT_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def mask_argument(key: str, value: Any) -> Any:
    """Mask a single argument value based on its key name or content."""
    if not isinstance(value, str):
        return value
    for pattern, placeholder in _ARG_KEY_PATTERNS:
        if pattern.search(key):
            return placeholder
    text = str(value)
    for pattern, placeholder in _VALUE_PATTERNS:
        if pattern.search(text):
            return placeholder
    return value


def mask_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Mask PII in an arguments dict."""
    return {k: mask_argument(k, v) for k, v in arguments.items()}


def sanitize_result(result: str, max_len: int = 500) -> str:
    """Sanitise a tool result: mask PII and truncate."""
    result = sanitize_text(result)
    if len(result) > max_len:
        result = result[:max_len] + "…"
    return result
