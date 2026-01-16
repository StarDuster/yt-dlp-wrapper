"""
Error diagnosis models, constants, and helpers for yt-dlp/ffmpeg failures.

This module consolidates error patterns and diagnosis logic into a single source of truth.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field
from typing import Optional

# =============================================================================
# 1. Error Patterns (Constants)
# =============================================================================

# Rate limiting / anti-bot patterns (should trigger retry later)
RATE_LIMIT_PATTERNS = [
    r"HTTP Error 429",
    r"Too Many Requests",
    r"Sign in to confirm you'?re not a bot",
    r"confirm you'?re not a bot",
    r"This helps protect our community",
    r"rate.?limit",
    r"Requested format is not available",
    r"Unable to extract.*?player",
    r"HTTP Error 403.*?Forbidden",
    r"IncompleteRead",
    r"Connection reset by peer",
    r"Read timed out",
    r"urlopen error",
    r"Got error: The read operation timed out",
    r"giving up after.*?retries",
]

# Members-only / Premium patterns (expected, not retryable)
MEMBERS_ONLY_PATTERNS = [
    r"Join this channel to get access",
    r"members.?only",
    r"This video is only available to members",
    r"available to members",
    r"members-only content",
    r"channel membership",
    r"This video requires payment",
    r"Premium members only",
    r"Join this channel to unlock",
    r"Memberships are not available",
    r"This video is available to this channel's members",
    r"This content isn't available",
    r"purchase.*?required",
]

# Private / unavailable patterns (expected, not retryable)
UNAVAILABLE_PATTERNS = [
    r"This video is private",
    r"Private video",
    r"Video unavailable",
    r"This video has been removed",
    r"This video is no longer available",
    r"The uploader has not made this video available",
    r"This video contains content from.*?who has blocked",
    r"blocked it in your country",
    r"age-restricted",
    r"This video may be inappropriate",
    r"Sign in to confirm your age",
    r"Premiere will begin",
    r"This live event will begin",
    r"Please sign in",
    r"not made this video available in your country",
    r"video is not available in your country",
    r"uploader has closed their youtube account",
    r"account associated with this video has been terminated",
    r"removed for violating",
]

# Transient / Retryable patterns
RETRY_PATTERNS = [
    r"HTTP Error 5\d{2}",
    r"Internal Server Error",
    r"did not get any data blocks",
    r"connection reset",
    r"connection timed out",
    r"ssl",
    r"temporary",
]


# =============================================================================
# 2. Data Models
# =============================================================================

@dataclass
class DownloadResult:
    """Track download results and error types"""

    # Counts
    success_count: int = 0
    rate_limited_count: int = 0
    members_only_count: int = 0
    unavailable_count: int = 0  # Private/removed/geo-blocked videos (expected)
    other_error_count: int = 0  # Unknown errors (critical)
    already_downloaded_count: int = 0

    # Error details
    rate_limit_errors: list = field(default_factory=list)
    members_only_errors: list = field(default_factory=list)
    unavailable_errors: list = field(default_factory=list)
    other_errors: list = field(default_factory=list)

    # Return code from yt-dlp
    return_code: int = 0
    # Optional summary for channel-level result
    final_status: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def total_errors(self) -> int:
        return self.rate_limited_count + self.members_only_count + self.unavailable_count + self.other_error_count

    @property
    def expected_errors(self) -> int:
        """Errors that are expected and don't indicate a problem (members-only, unavailable)"""
        return self.members_only_count + self.unavailable_count

    @property
    def critical_errors(self) -> int:
        """Errors that indicate a real problem (rate limit, unknown errors)"""
        return self.other_error_count

    @property
    def has_rate_limit(self) -> bool:
        return self.rate_limited_count > 0

    @property
    def has_members_only(self) -> bool:
        return self.members_only_count > 0

    def merge(self, other: "DownloadResult"):
        """Merge another result into this one"""
        self.success_count += other.success_count
        self.rate_limited_count += other.rate_limited_count
        self.members_only_count += other.members_only_count
        self.unavailable_count += other.unavailable_count
        self.other_error_count += other.other_error_count
        self.already_downloaded_count += other.already_downloaded_count
        self.rate_limit_errors.extend(other.rate_limit_errors)
        self.members_only_errors.extend(other.members_only_errors)
        self.unavailable_errors.extend(other.unavailable_errors)
        self.other_errors.extend(other.other_errors)
        # Keep worst return code
        if other.return_code != 0:
            self.return_code = other.return_code


# =============================================================================
# 3. Diagnosis Helpers
# =============================================================================

def _normalize_error_text(text: str) -> str:
    """Unify fancy apostrophes for matching consistency."""
    return (text or "").replace("'", "'")


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
    List-downloader error classification using shared patterns.

    Returns:
      - "rate_limit"
      - "unavailable"
      - "retry"
      - "failed"
    """
    text = _normalize_error_text(error_msg)

    for pattern in RATE_LIMIT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return "rate_limit"

    for pattern in UNAVAILABLE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return "unavailable"

    for pattern in MEMBERS_ONLY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return "unavailable"

    for pattern in RETRY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
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
