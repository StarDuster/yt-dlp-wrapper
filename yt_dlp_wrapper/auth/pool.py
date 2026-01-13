"""
YouTube account pool utilities.

This module provides:
- Multi-account discovery from a conventional directory layout
- A thread-safe cooldown-based account pool (for rate-limit switching)
- Helpers to parse yt-dlp speed strings for overall speed aggregation
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_ACCOUNTS_DIR = Path.home() / ".config/yt-dlp-wrapper/youtube-accounts"


def _resolve_accounts_dir(accounts_dir: Optional[Path]) -> Path:
    if accounts_dir is not None:
        return Path(accounts_dir).expanduser().resolve()
    try:
        from .. import config  # local import to avoid forcing config at module import time

        cfg_dir = getattr(config, "YOUTUBE_ACCOUNTS_DIR", None)
        if cfg_dir:
            return Path(cfg_dir).expanduser().resolve()
    except Exception:
        pass
    return DEFAULT_ACCOUNTS_DIR.expanduser().resolve()


@dataclass
class YouTubeAccount:
    name: str
    cookies_file: Path
    profile_dir: Optional[Path] = None
    cooldown_until: float = 0.0
    rate_limited: int = 0


class YouTubeAccountPool:
    """
    Thread-safe pool for rotating YouTube accounts when rate-limited.

    Notes:
    - The pool only controls *selection* and *cooldown bookkeeping*.
    - Callers are responsible for actually switching yt-dlp cookies/profile.
    """

    def __init__(self, accounts: list[YouTubeAccount], cooldown_seconds: float = 900.0) -> None:
        self.accounts = accounts
        try:
            self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        except Exception:
            self.cooldown_seconds = 900.0
        self._lock = threading.Lock()

    def size(self) -> int:
        return len(self.accounts)

    def mark_rate_limited(self, idx: int) -> float:
        """Mark an account as rate-limited and return its cooldown-until timestamp."""
        with self._lock:
            if idx < 0 or idx >= len(self.accounts):
                return 0.0
            now = time.time()
            until = max(self.accounts[idx].cooldown_until, now + self.cooldown_seconds)
            self.accounts[idx].cooldown_until = until
            self.accounts[idx].rate_limited += 1
            return until

    def pick_next(
        self,
        current_idx: Optional[int],
        *,
        exclude_current: bool = False,
    ) -> tuple[int, float]:
        """
        Pick the next available account index.

        Returns:
            (idx, wait_seconds)
        - idx is always within [0, n) when n>0
        - wait_seconds>0 means all accounts are cooling down and caller may wait
        """
        with self._lock:
            n = len(self.accounts)
            if n <= 0:
                return -1, 0.0

            now = time.time()

            if current_idx is not None and 0 <= current_idx < n:
                order = [(current_idx + 1 + k) % n for k in range(n)]
            else:
                order = list(range(n))

            if exclude_current and current_idx is not None:
                order = [i for i in order if i != current_idx]
                if not order:
                    order = [current_idx]

            for i in order:
                if self.accounts[i].cooldown_until <= now:
                    return i, 0.0

            earliest = min(order, key=lambda i: self.accounts[i].cooldown_until)
            wait = max(0.0, self.accounts[earliest].cooldown_until - now)
            return earliest, wait


def get_account_paths(account_name: str, accounts_dir: Optional[Path] = None) -> tuple[Path, Path]:
    """
    Resolve (profile_dir, cookies_file) for a named account.

    Layout:
      ~/.config/yt-dlp-wrapper/youtube-accounts/<name>/
        - browser-profile/        (Playwright persistent profile dir)
        - youtube_cookies.txt     (Netscape cookies file for yt-dlp)
    """
    base = _resolve_accounts_dir(accounts_dir)
    d = (base / (account_name or "")).expanduser().resolve()
    profile_dir = d / "browser-profile"
    cookies_file = d / "youtube_cookies.txt"
    return profile_dir, cookies_file


def discover_accounts(accounts_dir: Optional[Path] = None) -> list[YouTubeAccount]:
    """
    Discover accounts from the conventional directory layout.

    Returns:
        A list of accounts that have an existing youtube_cookies.txt.
    """
    base = _resolve_accounts_dir(accounts_dir)
    if not base.exists():
        return []

    accounts: list[YouTubeAccount] = []
    try:
        children = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name)
    except Exception:
        return []

    for d in children:
        cookies_file = d / "youtube_cookies.txt"
        if not cookies_file.exists():
            continue
        profile_dir = d / "browser-profile"
        accounts.append(
            YouTubeAccount(
                name=d.name,
                cookies_file=cookies_file,
                profile_dir=profile_dir,
            )
        )

    return accounts


def load_accounts_from_config(
    *,
    cooldown_seconds_default: float = 900.0,
    accounts_dir: Optional[Path] = None,
) -> tuple[list[YouTubeAccount], Optional[YouTubeAccountPool]]:
    """
    Load accounts using the multi-account directory first; fallback to legacy config cookie.

    Returns:
        (accounts, pool)
        pool is None when <=1 account is available.
    """
    from .. import config  # local import to avoid forcing config at module import time

    accounts = discover_accounts(accounts_dir=accounts_dir)

    # Fallback: legacy single-account cookies file
    if not accounts:
        cookies_file = getattr(config, "YOUTUBE_COOKIES_FILE", None)
        profile_dir = getattr(config, "YOUTUBE_BROWSER_PROFILE", None)
        if isinstance(cookies_file, Path) and cookies_file.exists():
            accounts = [
                YouTubeAccount(
                    name="default",
                    cookies_file=cookies_file,
                    profile_dir=profile_dir if isinstance(profile_dir, Path) else None,
                )
            ]

    cooldown = getattr(config, "YOUTUBE_ACCOUNT_COOLDOWN_SECONDS", cooldown_seconds_default)
    pool = YouTubeAccountPool(accounts, cooldown_seconds=float(cooldown)) if len(accounts) > 1 else None
    return accounts, pool


_SPEED_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)?\s*$")


def parse_ytdlp_speed_to_bps(speed: Optional[str]) -> float:
    """
    Parse yt-dlp speed strings like '12.3MiB/s' into bytes/sec.
    Returns 0.0 when unknown/unparseable.
    """
    if not speed:
        return 0.0
    s = str(speed).strip()
    if not s or s == "...":
        return 0.0
    if s.endswith("/s"):
        s = s[:-2]

    m = _SPEED_RE.match(s)
    if not m:
        return 0.0

    try:
        value = float(m.group(1))
    except Exception:
        return 0.0

    unit = (m.group(2) or "B").strip()
    if not unit:
        unit = "B"

    unit_map: dict[str, float] = {
        "B": 1.0,
        "KB": 1000.0,
        "MB": 1000.0**2,
        "GB": 1000.0**3,
        "TB": 1000.0**4,
        "KiB": 1024.0,
        "MiB": 1024.0**2,
        "GiB": 1024.0**3,
        "TiB": 1024.0**4,
        # Some variants
        "kB": 1000.0,
        "mB": 1000.0**2,
        "gB": 1000.0**3,
    }

    mult = unit_map.get(unit)
    if mult is None:
        # Rare variants like K/M/G without 'B'
        if unit in {"K", "k"}:
            mult = 1000.0
        elif unit in {"M", "m"}:
            mult = 1000.0**2
        elif unit in {"G", "g"}:
            mult = 1000.0**3
        else:
            return 0.0

    bps = value * mult
    if bps < 0:
        return 0.0
    return float(bps)
