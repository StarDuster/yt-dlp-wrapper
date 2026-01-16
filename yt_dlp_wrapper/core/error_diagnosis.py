"""
Shared error diagnosis helpers for yt-dlp/ffmpeg failures.

This module is intentionally lightweight so both list and channel downloaders can reuse it
without introducing circular imports.
"""

from __future__ import annotations

import contextlib
import re
from typing import Optional

from .models import MEMBERS_ONLY_PATTERNS, RATE_LIMIT_PATTERNS, UNAVAILABLE_PATTERNS


def _normalize_error_text(text: str) -> str:
    # Unify fancy apostrophes for matching consistency.
    return (text or "").replace("’", "'")


def extract_http_status_from_text(text: str) -> Optional[int]:
    """
    Best-effort extraction of HTTP status codes from yt-dlp/ffmpeg output.
    Keep this intentionally conservative to avoid false positives.
    """
    s = _normalize_error_text(text or "")
    if not s:
        return None

    patterns = [
        r"http\s*error\s*(\d{3})",
        r"server\s+returned\s*(\d{3})",
        r"returned\s+error:\s*(\d{3})",
        r"\b(\d{3})\s+too\s+many\s+requests\b",
        r"\b(\d{3})\s+forbidden\b",
        r"\b(\d{3})\s+unauthorized\b",
        r"\b(\d{3})\s+not\s+found\b",
        r"\b(\d{3})\s+gone\b",
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if not m:
            continue
        with contextlib.suppress(Exception):
            code = int(m.group(1))
            if 100 <= code <= 599:
                return code
    return None


def classify_list_error(error_msg: str) -> str:
    """
    List-downloader error classification.

    Returns:
      - "rate_limit"
      - "unavailable"
      - "retry"
      - "failed"
    """
    error_lower = (error_msg or "").lower()

    rate_limit_patterns = [
        "http error 429",
        "429 too many requests",
        "rate-limited",
        "rate limited",
        "too many requests",
        "try again later",
        "confirm you're not a bot",
        "confirm you’re not a bot",
        "sign in to confirm you're not a bot",
        "sign in to confirm you’re not a bot",
        "this helps protect our community",
    ]
    for pattern in rate_limit_patterns:
        if pattern in error_lower:
            return "rate_limit"

    unavailable_patterns = [
        "this video is private",
        "this video has been removed",
        "this video is no longer available",
        "sign in to confirm your age",
        "this video may be inappropriate",
        "members-only content",
        "join this channel",
        "not made this video available in your country",
        "video is not available in your country",
        "blocked in your country",
        "the uploader has not made this video available",
        "uploader has closed their youtube account",
        "account associated with this video has been terminated",
        "removed for violating",
        "video unavailable",
    ]
    for pattern in unavailable_patterns:
        if pattern in error_lower:
            return "unavailable"

    retry_patterns = [
        "http error 500",
        "http error 503",
        "http error 502",
        "internal server error",
        "did not get any data blocks",
        "connection reset",
        "connection timed out",
        "ssl",
        "temporary",
        "incompleteread",
    ]
    for pattern in retry_patterns:
        if pattern in error_lower:
            return "retry"

    return "failed"


def classify_channel_error_line(error_line: str) -> str:
    """
    Channel-downloader error line classification.

    Returns:
      - "rate_limit"
      - "members_only"
      - "unavailable"
      - "other"
    """
    normalized = _normalize_error_text(error_line or "")

    # Fast-path on explicit HTTP status if present.
    http_status = extract_http_status_from_text(normalized)
    if http_status == 429:
        return "rate_limit"

    for pattern in RATE_LIMIT_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return "rate_limit"

    for pattern in MEMBERS_ONLY_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return "members_only"

    for pattern in UNAVAILABLE_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return "unavailable"

    # Conservative fallback: treat explicit 4xx as unavailable when patterns miss.
    if http_status in {401, 403, 404, 410}:
        return "unavailable"

    return "other"


def diagnose_ffmpeg_error(error_msg: str) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Classify ffmpeg failures into a coarse diagnosis for observability only.

    Returns (diagnosis, http_status, hint) where diagnosis is one of:
      - "rate_limit": likely 429 / bot protection / too many requests
      - "access": likely cannot access media (403/401/404/410/private/age-restricted/etc.)
      - "retry": likely transient (5xx/connection resets/etc.)
      - None: unknown
    """
    if not error_msg:
        return None, None, None

    base = classify_list_error(error_msg)
    http_status = extract_http_status_from_text(error_msg)

    diagnosis: Optional[str] = None
    if base == "rate_limit":
        diagnosis = "rate_limit"
    elif base == "unavailable":
        diagnosis = "access"
    elif base == "retry":
        diagnosis = "retry"
    else:
        if http_status == 429:
            diagnosis = "rate_limit"
        elif http_status in {401, 403, 404, 410}:
            diagnosis = "access"
        elif http_status is not None and 500 <= int(http_status) <= 599:
            diagnosis = "retry"

    hint: Optional[str] = None
    if diagnosis == "rate_limit":
        hint = "Suspected rate limit"
    elif diagnosis == "access":
        hint = "Suspected access restriction"
    elif diagnosis == "retry":
        hint = "Suspected transient error"

    if hint and http_status:
        hint = f"{hint} (HTTP {http_status})"
    elif http_status:
        hint = f"HTTP {http_status}"

    return diagnosis, http_status, hint

