"""
Data models and constants for YouTube downloader.
"""

from dataclasses import dataclass, field
from typing import Optional

# Error patterns for classification
# Rate limiting / anti-bot patterns (should trigger retry later)
RATE_LIMIT_PATTERNS = [
    r"HTTP Error 429",
    r"Too Many Requests",
    r"Sign in to confirm you'?re not a bot",  # Bot detection
    r"confirm you'?re not a bot",
    r"This helps protect our community",  # Bot detection message
    r"rate.?limit",
    r"Requested format is not available",  # Sometimes caused by rate limiting
    r"Unable to extract.*?player",  # Player extraction fails under rate limit
    r"HTTP Error 403.*?Forbidden",  # Sometimes rate limit manifests as 403
    r"IncompleteRead",  # Connection issues due to throttling
    r"Connection reset by peer",  # Server dropped connection
    r"Read timed out",  # Timeout due to throttling
    r"urlopen error",  # Network issues
    r"Got error: The read operation timed out",
    r"giving up after.*?retries",  # Retry exhaustion (likely rate limit)
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
    r"This content isn't available",  # Generic unavailable sometimes means members
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
    r"This video contains content from.*?who has blocked",  # Copyright block
    r"blocked it in your country",  # Geo-restriction
    r"age-restricted",  # Age restriction (without login)
    r"This video may be inappropriate",  # Age gate
    r"Sign in to confirm your age",  # Age verification required
    r"Premiere will begin",  # Scheduled premiere
    r"This live event will begin",  # Scheduled live
    r"Please sign in",  # Generic sign-in requirement (not bot detection)
]


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
