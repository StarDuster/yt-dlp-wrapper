"""
YouTube account pool utilities.

This module provides:
- Multi-account discovery from a conventional directory layout
- A thread-safe cooldown-based account pool (for rate-limit switching)
- Helpers to parse yt-dlp speed strings for overall speed aggregation
"""

from __future__ import annotations

import atexit
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_ACCOUNTS_DIR = Path.home() / ".config/yt-dlp-wrapper/youtube-accounts"
_STATE_FILE_NAME = ".account_pool_state.json"
_STATE_VERSION = 1


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


def _resolve_state_file(accounts_dir: Optional[Path]) -> Path:
    base = _resolve_accounts_dir(accounts_dir)
    return (base / _STATE_FILE_NAME).expanduser().resolve()


def _load_usage_state(state_file: Path) -> dict[str, dict[str, object]]:
    try:
        raw = state_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {}
    try:
        data = json.loads(raw or "")
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    accounts = data.get("accounts")
    if not isinstance(accounts, dict):
        return {}
    out: dict[str, dict[str, object]] = {}
    for k, v in accounts.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, dict):
            out[k] = v
    return out


def _apply_usage_state(accounts: list["YouTubeAccount"], state: dict[str, dict[str, object]]) -> None:
    for acc in accounts:
        d = state.get(acc.name)
        if not d:
            continue
        try:
            acc.usage_count = int(d.get("usage_count") or 0)
        except Exception:
            acc.usage_count = 0
        try:
            acc.usage_updated_at = float(d.get("usage_updated_at") or 0.0)
        except Exception:
            acc.usage_updated_at = 0.0


@dataclass
class YouTubeAccount:
    name: str
    cookies_file: Path
    profile_dir: Optional[Path] = None
    cooldown_until: float = 0.0
    rate_limited: int = 0
    usage_count: int = 0
    usage_updated_at: float = 0.0


class YouTubeAccountPool:
    """
    Thread-safe pool for rotating YouTube accounts when rate-limited.

    Notes:
    - The pool only controls *selection* and *cooldown bookkeeping*.
    - Callers are responsible for actually switching yt-dlp cookies/profile.
    """

    def __init__(
        self,
        accounts: list[YouTubeAccount],
        cooldown_seconds: float = 900.0,
        *,
        state_file: Optional[Path] = None,
        autosave: bool = True,
    ) -> None:
        self.accounts = accounts
        try:
            self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        except Exception:
            self.cooldown_seconds = 900.0
        self._lock = threading.Lock()
        self._state_file = Path(state_file).expanduser().resolve() if state_file is not None else None
        self._dirty = False
        self._autosave = bool(autosave) and (self._state_file is not None)
        self._atexit_registered = False

        if self._autosave:
            self._register_atexit()

    def size(self) -> int:
        return len(self.accounts)

    def _register_atexit(self) -> None:
        if self._atexit_registered:
            return
        self._atexit_registered = True
        try:
            atexit.register(self.save_state)
        except Exception:
            pass

    def save_state(self) -> bool:
        """
        Persist usage counters to a JSON file (best-effort).

        This is intended for long-running jobs so the next run can continue balancing
        from the previous distribution. It only saves when the pool is marked dirty.
        """
        state_file = self._state_file
        if state_file is None:
            return False

        import json as _json
        import time as _time

        with self._lock:
            if not self._dirty:
                return True

            now = float(_time.time())
            payload: dict[str, object] = {
                "version": _STATE_VERSION,
                "updated_at": now,
                "accounts": {
                    str(a.name): {
                        "usage_count": int(getattr(a, "usage_count", 0) or 0),
                        "usage_updated_at": float(getattr(a, "usage_updated_at", 0.0) or 0.0),
                    }
                    for a in self.accounts
                },
            }

            try:
                state_file.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            try:
                tmp = state_file.with_suffix(state_file.suffix + ".tmp")
                tmp.write_text(_json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
                tmp.replace(state_file)
                self._dirty = False
                return True
            except Exception:
                return False

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

    def pick_least_used(self, exclude_idx: Optional[int] = None) -> tuple[int, float]:
        """
        Pick the least-used available account index (by video count).

        Returns:
            (idx, wait_seconds)
        - idx is always within [0, n) when n>0
        - wait_seconds>0 means all eligible accounts are cooling down and caller may wait

        Notes:
        - When an account is selected with wait_seconds==0, its usage_count is incremented
          immediately (reservation) to make this method concurrency-safe across workers.
        """
        with self._lock:
            n = len(self.accounts)
            if n <= 0:
                return -1, 0.0

            now = time.time()
            candidates: list[int] = []
            for i, acc in enumerate(self.accounts):
                if exclude_idx is not None and i == exclude_idx:
                    continue
                if acc.cooldown_until <= now:
                    candidates.append(i)

            if candidates:
                best = min(candidates, key=lambda i: (self.accounts[i].usage_count, i))
                try:
                    self.accounts[best].usage_count += 1
                except Exception:
                    self.accounts[best].usage_count = 1
                self.accounts[best].usage_updated_at = float(now)
                self._dirty = True
                return best, 0.0

            order = list(range(n))
            if exclude_idx is not None and 0 <= exclude_idx < n and n > 1:
                order = [i for i in order if i != exclude_idx] or list(range(n))

            earliest = min(order, key=lambda i: self.accounts[i].cooldown_until)
            wait = max(0.0, self.accounts[earliest].cooldown_until - now)
            return earliest, wait

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
    state_file = _resolve_state_file(accounts_dir)
    _apply_usage_state(accounts, _load_usage_state(state_file))

    pool = (
        YouTubeAccountPool(accounts, cooldown_seconds=float(cooldown), state_file=state_file)
        if len(accounts) > 1
        else None
    )
    return accounts, pool
