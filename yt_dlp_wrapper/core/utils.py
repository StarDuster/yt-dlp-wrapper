"""
Shared lightweight utilities for yt-dlp-wrapper.

This module intentionally avoids importing yt_dlp / rich so it can be reused by both
list_downloader and channel_downloader without introducing heavy dependencies or
circular imports.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import subprocess
import threading
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

YOUTUBE_ID_RE = re.compile(r"^[0-9A-Za-z_-]{11}$")


def _has_nvidia_gpu() -> bool:
    """
    Best-effort NVIDIA GPU detection.
    We prefer checking nvidia-smi since it is the most common indicator in server environments.
    """
    try:
        p = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
        )
        return p.returncode == 0
    except Exception:
        return False


def _get_physical_cpu_cores() -> int:
    """
    Best-effort physical core count.

    Keep it simple: on Linux, parse /proc/cpuinfo and count unique (physical id, core id) pairs.
    Fall back to os.cpu_count() if unavailable.
    """
    try:
        if Path("/proc/cpuinfo").exists():
            pairs: set[tuple[int, int]] = set()
            physical_id: Optional[int] = None
            core_id: Optional[int] = None
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for raw in f:
                    line = (raw or "").strip()
                    if not line:
                        if core_id is not None:
                            pairs.add((physical_id or 0, core_id))
                        physical_id = None
                        core_id = None
                        continue
                    if line.startswith("physical id"):
                        try:
                            physical_id = int(line.split(":", 1)[1].strip())
                        except Exception:
                            pass
                    elif line.startswith("core id"):
                        try:
                            core_id = int(line.split(":", 1)[1].strip())
                        except Exception:
                            pass
            if core_id is not None:
                pairs.add((physical_id or 0, core_id))
            if pairs:
                return max(1, len(pairs))
    except Exception:
        pass

    try:
        return max(1, int(os.cpu_count() or 1))
    except Exception:
        return 1


def _parse_ffmpeg_speed(value: str) -> Optional[float]:
    s = (value or "").strip().lower()
    if not s:
        return None
    if s.endswith("x"):
        s = s[:-1].strip()
    try:
        v = float(s)
    except Exception:
        return None
    # Keep 0.00x as 0.0 so UI can display it (instead of treating it as missing).
    return v if v >= 0 else None


class _FFmpegProgressTail:
    """
    Incrementally read ffmpeg -progress output from a file.

    ffmpeg writes lines like:
      out_time_ms=1234567
      speed=1.23x
      progress=continue
    """

    def __init__(self, path: Path):
        self.path = path
        self.pos = 0
        self.out_time_s: float = 0.0
        self.speed_factor: Optional[float] = None
        # Keep the raw speed text when ffmpeg reports non-numeric values (e.g. N/A).
        self.speed_text: Optional[str] = None
        self.last_progress: Optional[str] = None

    def poll(self) -> None:
        try:
            st = self.path.stat()
        except Exception:
            return
        size = int(getattr(st, "st_size", 0) or 0)
        if size <= 0:
            return
        if size < self.pos:
            self.pos = 0
        try:
            with self.path.open("r", encoding="utf-8", errors="ignore") as f:
                f.seek(self.pos)
                chunk = f.read()
                self.pos = f.tell()
        except Exception:
            return

        for line in (chunk or "").splitlines():
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = (k or "").strip()
            v = (v or "").strip()
            if not k:
                continue
            if k in {"out_time_us", "out_time_ms"}:
                # Despite the name, ffmpeg often reports microseconds for out_time_ms too.
                try:
                    us = int(v)
                    if us > 0:
                        self.out_time_s = float(us) / 1_000_000.0
                except Exception:
                    pass
            elif k == "speed":
                vv = (v or "").strip()
                if vv.lower() in {"n/a", "na"}:
                    self.speed_factor = None
                    self.speed_text = "N/A"
                else:
                    sf = _parse_ffmpeg_speed(vv)
                    if sf is not None:
                        self.speed_factor = sf
                        self.speed_text = None
                    else:
                        # Preserve any other raw text as-is for display.
                        self.speed_factor = None
                        self.speed_text = vv or None
            elif k == "progress":
                self.last_progress = v


def _format_eta(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    try:
        seconds = int(seconds)
    except Exception:
        return ""
    if seconds < 0:
        return ""
    d, rem = divmod(seconds, 86400)
    if d:
        h, rem = divmod(rem, 3600)
        m, s = divmod(rem, 60)
        return f"{d:d}d{h:02d}:{m:02d}:{s:02d}"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _format_time(seconds: Optional[float]) -> str:
    """Format seconds with decimals for segment/transcode progress."""
    if seconds is None:
        return ""
    try:
        seconds_f = float(seconds)
    except Exception:
        return ""
    if seconds_f < 0:
        return ""
    m, s = divmod(seconds_f, 60.0)
    h, m = divmod(m, 60.0)
    if h:
        return f"{int(h):d}:{int(m):02d}:{s:05.2f}"
    return f"{int(m):02d}:{s:05.2f}"


def _backoff_sleep(attempt: int, base: float, cap: float, jitter: float) -> float:
    """Exponential backoff seconds for yt-dlp retry_sleep_functions."""
    try:
        a = int(attempt)
    except Exception:
        a = 1
    if a < 1:
        a = 1

    try:
        base_f = float(base)
        cap_f = float(cap)
        jitter_f = float(jitter)
    except Exception:
        return 0.0

    if base_f <= 0 or cap_f <= 0:
        return 0.0

    if jitter_f < 0:
        jitter_f = 0.0
    if jitter_f > 1:
        jitter_f = 1.0

    t = min(cap_f, base_f * (2 ** (a - 1)))
    if jitter_f > 0:
        t *= (1.0 - jitter_f) + (2.0 * jitter_f * random.random())
    if t < 0:
        return 0.0
    return float(t)


def parse_youtube_id(value: str) -> Optional[str]:
    s = (value or "").strip()
    if not s:
        return None
    if YOUTUBE_ID_RE.match(s):
        return s
    try:
        u = urlparse(s)
    except Exception:
        return None

    host = (u.netloc or "").lower()
    path = u.path or ""

    if host in {"youtu.be", "www.youtu.be"}:
        vid = path.lstrip("/").split("/")[0]
        return vid if YOUTUBE_ID_RE.match(vid or "") else None

    if "youtube.com" in host or host in {"m.youtube.com", "music.youtube.com"}:
        qs = parse_qs(u.query or "")
        v = (qs.get("v") or [None])[0]
        if v and YOUTUBE_ID_RE.match(v):
            return v
        if path.startswith("/shorts/"):
            parts = path.split("/")
            if len(parts) >= 3 and parts[2] and YOUTUBE_ID_RE.match(parts[2]):
                return parts[2]
        if path.startswith("/embed/"):
            parts = path.split("/")
            if len(parts) >= 3 and parts[2] and YOUTUBE_ID_RE.match(parts[2]):
                return parts[2]

    return None


def _to_msec(seconds: float) -> int:
    try:
        return int(round(float(seconds) * 1000.0))
    except Exception:
        return 0


def load_archive_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = (line or "").strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2 and parts[0] == "youtube":
                    key = (parts[1] or "").strip()
                    if key and (
                        YOUTUBE_ID_RE.match(key.split(":", 1)[0]) if ":" in key else YOUTUBE_ID_RE.match(key)
                    ):
                        ids.add(key)
                elif len(parts) == 1:
                    key = (parts[0] or "").strip()
                    if key and (
                        YOUTUBE_ID_RE.match(key.split(":", 1)[0]) if ":" in key else YOUTUBE_ID_RE.match(key)
                    ):
                        ids.add(key)
    except Exception:
        return set()
    return ids


def append_archive_id(path: Path, vid: str, lock: threading.Lock) -> None:
    if not vid:
        return
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(f"youtube {vid}\n")


def append_failed_id(path: Path, vid: str, reason: str, lock: threading.Lock) -> None:
    if not vid:
        return
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(f"{vid}\t{reason}\n")


def load_failed_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = (line or "").strip()
                if not line:
                    continue
                parts = line.split("\t")
                if not parts:
                    continue
                key = (parts[0] or "").strip()
                if key and (
                    YOUTUBE_ID_RE.match(key.split(":", 1)[0]) if ":" in key else YOUTUBE_ID_RE.match(key)
                ):
                    ids.add(key)
    except Exception:
        return set()
    return ids


def _derive_channel_key(url: str) -> str:
    s = (url or "").strip()
    if not s:
        return "channel"
    try:
        u = urlparse(s)
        host = (u.netloc or "").lower()
        path = (u.path or "").strip("/")
        if path:
            parts = [p for p in path.split("/") if p]
            for i, part in enumerate(parts):
                if part.startswith("@"):
                    return part
                if part in {"channel", "c", "user"} and i + 1 < len(parts):
                    return f"{part}-{parts[i + 1]}"
            return parts[0]
        if host:
            return host
    except Exception:
        pass

    digest = hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"channel-{digest}"


def _sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for filesystem.

    Keep using underscore for invalid characters to maintain backward compatibility.
    """
    filename = (filename or "").replace("\0", "")

    # Replace control characters with spaces
    for char in ["\n", "\r", "\t"]:
        filename = filename.replace(char, " ")

    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Collapse multiple spaces
    while "  " in filename:
        filename = filename.replace("  ", " ")

    if len(filename) > 200:
        filename = filename[:200]

    return filename.strip()

