"""
Download YouTube videos from a plain text list (video_id or URL).

This is a refactor/port of talkverse_src/download_videos.py into yt-dlp-wrapper.
Key features:
- Rich: overall progress + per-worker current download
- Overall speed: aggregated from active workers
- Optional multi-account cookies pool for rate-limit switching
- Optional Invidious support (single instance, no rotation/fallback)
"""

from __future__ import annotations

import queue
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from rich.console import Console
from rich.filesize import decimal as fmt_bytes
from rich.progress import BarColumn, Progress, ProgressColumn, SpinnerColumn, TextColumn
from rich.text import Text
import yt_dlp
from yt_dlp.utils import DownloadError

from .. import config
from ..auth.pool import YouTubeAccount, YouTubeAccountPool, load_accounts_from_config


console = Console()

YOUTUBE_ID_RE = re.compile(r"^[0-9A-Za-z_-]{11}$")


def _format_eta(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    try:
        seconds = int(seconds)
    except Exception:
        return ""
    if seconds < 0:
        return ""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


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


class CountOrBytesColumn(ProgressColumn):
    """Overall: N/M. Worker: bytes downloaded / total."""

    def render(self, task) -> Text:  # type: ignore[override]
        mode = task.fields.get("mode")
        if mode == "count":
            total = int(task.total or 0)
            return Text(f"{int(task.completed)}/{total}")

        completed = int(task.completed or 0)
        total = task.total
        if total is None:
            return Text(f"{fmt_bytes(completed)}/?")
        return Text(f"{fmt_bytes(completed)}/{fmt_bytes(int(total))}")


class SpeedColumn(ProgressColumn):
    """Overall: sum(worker speeds). Worker: current speed."""

    def __init__(self, worker_speeds: dict[int, float] | None = None):
        super().__init__()
        self.worker_speeds = worker_speeds if worker_speeds is not None else {}

    def render(self, task) -> Text:  # type: ignore[override]
        mode = task.fields.get("mode")
        if mode == "count":
            total_speed = sum(self.worker_speeds.values())
            if total_speed <= 0:
                return Text("0 B/s")
            return Text(f"{fmt_bytes(int(total_speed))}/s")

        if mode != "download":
            return Text("")
        speed = task.fields.get("speed")
        if not speed:
            return Text("")
        try:
            return Text(f"{fmt_bytes(int(speed))}/s")
        except Exception:
            return Text("")


class ETAColumn(ProgressColumn):
    def render(self, task) -> Text:  # type: ignore[override]
        if task.fields.get("mode") != "download":
            return Text("")
        return Text(_format_eta(task.fields.get("eta")))


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


def _select_invidious_instance(*, no_invidious: bool) -> Optional[str]:
    """
    Select a single Invidious instance. No rotation/fallback.
    """
    if no_invidious:
        return None
    inst = getattr(config, "INVIDIOUS_INSTANCE", None)
    if inst:
        return inst
    return None


def _merge_extractor_args(
    extractor_args: dict[str, dict[str, list[str]]],
    raw_args: Optional[object],
) -> None:
    if not raw_args:
        return

    if isinstance(raw_args, str):
        items = [raw_args]
    elif isinstance(raw_args, (list, tuple)):
        items = [str(x) for x in raw_args if str(x).strip()]
    else:
        return

    for raw in items:
        if ":" not in raw:
            continue
        extractor, rest = raw.split(":", 1)
        extractor = extractor.strip()
        if not extractor or not rest:
            continue
        for part in rest.split(";"):
            part = part.strip()
            if not part or "=" not in part:
                continue
            key, val = part.split("=", 1)
            key = key.strip()
            val = val.strip()
            if not key:
                continue
            extractor_args.setdefault(extractor, {}).setdefault(key, []).append(val)


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
                    if YOUTUBE_ID_RE.match(parts[1]):
                        ids.add(parts[1])
                elif len(parts) == 1 and YOUTUBE_ID_RE.match(parts[0]):
                    ids.add(parts[0])
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
                if parts and YOUTUBE_ID_RE.match(parts[0]):
                    ids.add(parts[0])
    except Exception:
        return set()
    return ids


def classify_error(error_msg: str) -> str:
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


@dataclass
class Stats:
    ok: int = 0
    skipped: int = 0
    unavailable: int = 0
    failed: int = 0


def download_from_input_list(
    *,
    input_list_path: Path,
    output_dir: Path,
    workers: int = 8,
    limit: int = 0,
    debug: bool = False,
    no_invidious: bool = False,
    accounts_dir: Optional[Path] = None,
) -> int:
    # #region agent log
    import json as _json; _log_path = Path("/data2/youtube/.cursor/debug.log"); _t0 = time.time()
    def _dbg(msg, data=None, hyp="?"): _log_path.parent.mkdir(parents=True, exist_ok=True); _log_path.open("a").write(_json.dumps({"ts": time.time(), "elapsed": round(time.time()-_t0, 3), "hyp": hyp, "msg": msg, "data": data or {}}) + "\n")
    _dbg("download_from_input_list START", {"input": str(input_list_path), "output": str(output_dir)}, "A")
    # #endregion

    input_list_path = input_list_path.expanduser().resolve()
    if not input_list_path.exists():
        raise FileNotFoundError(f"Input list not found: {input_list_path}")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preload yt-dlp plugins in the main thread to avoid multi-threaded import races.
    # Without this, concurrent YoutubeDL() creation in multiple workers may trigger
    # plugin loading simultaneously and cause spurious ImportError like
    # "cannot import name ... from yt_dlp_plugins...".
    # #region agent log
    _dbg("load_all_plugins START", hyp="B")
    # #endregion
    try:
        from yt_dlp.plugins import load_all_plugins

        load_all_plugins()
    except Exception:
        # Plugins are optional; never block downloads due to plugin import failures.
        pass
    # #region agent log
    _dbg("load_all_plugins END", hyp="B")
    # #endregion

    # Accounts (multi-account dir first; fallback to legacy config cookie)
    # #region agent log
    _dbg("load_accounts_from_config START", hyp="C")
    # #endregion
    accounts, pool = load_accounts_from_config(accounts_dir=accounts_dir)
    # #region agent log
    _dbg("load_accounts_from_config END", {"num_accounts": len(accounts) if accounts else 0}, "C")
    # #endregion
    if accounts:
        console.print(
            f"Account pool enabled: {len(accounts)} account(s) "
            f"(auto-switch={'yes' if pool else 'no'})"
        )

    # Invidious (single instance)
    invidious_instance = _select_invidious_instance(no_invidious=no_invidious)
    if invidious_instance:
        console.print(
            f"Invidious: {invidious_instance}"
        )
    else:
        console.print("Invidious: direct")

    archive_file = output_dir / "download.archive.txt"
    unavailable_file = output_dir / "unavailable.txt"
    failed_file = output_dir / "failed.txt"

    archive_lock = threading.Lock()
    unavailable_lock = threading.Lock()
    failed_lock = threading.Lock()
    processed_lock = threading.Lock()

    # #region agent log
    _dbg("load_archive_ids START", {"file": str(archive_file)}, "D")
    # #endregion
    archived_ids = load_archive_ids(archive_file)
    # #region agent log
    _dbg("load_archive_ids END", {"count": len(archived_ids)}, "D")
    _dbg("load_failed_ids START", {"file": str(unavailable_file)}, "D")
    # #endregion
    unavailable_ids = load_failed_ids(unavailable_file)
    # #region agent log
    _dbg("load_failed_ids END", {"count": len(unavailable_ids)}, "D")
    # #endregion
    processed_ids: set[str] = set(archived_ids) | set(unavailable_ids)

    # Count tasks
    # #region agent log
    _dbg("count_tasks START", {"input_file": str(input_list_path), "limit": limit}, "E")
    # #endregion
    total = 0
    already_done = 0
    with input_list_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            total += 1
            vid = parse_youtube_id(s)
            if vid and vid in processed_ids:
                already_done += 1
            if limit and total >= limit:
                break

    # #region agent log
    _dbg("count_tasks END", {"total": total, "already_done": already_done}, "E")
    _dbg("INIT COMPLETE", {"total_init_time": round(time.time()-_t0, 3)}, "A")
    # #endregion

    if total <= 0:
        console.print("Input list is empty or contains no valid lines")
        return 0

    console.print(
        f"Job summary: total={total} processed={already_done} remaining={total - already_done} "
        f"output={output_dir}"
    )

    # Worker speed tracking (bytes/sec)
    worker_speeds: dict[int, float] = {}
    ui_lock = threading.Lock()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=None),
        CountOrBytesColumn(),
        SpeedColumn(worker_speeds=worker_speeds),
        ETAColumn(),
        console=console,
        refresh_per_second=10,
    )

    overall_task = progress.add_task(
        "Overall", total=total, completed=already_done, mode="count"
    )
    worker_tasks = [
        progress.add_task(
            f"Worker {i+1}: idle",
            total=0,
            mode="download",
            speed=None,
            eta=None,
            visible=False,
        )
        for i in range(max(1, int(workers)))
    ]

    q: queue.Queue[Optional[str]] = queue.Queue(maxsize=max(2, int(workers) * 2))
    stop_event = threading.Event()

    stats = Stats(skipped=already_done)
    stats_lock = threading.Lock()

    def producer() -> None:
        sent = 0
        try:
            with input_list_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if stop_event.is_set():
                        break
                    s = (line or "").strip()
                    if not s:
                        continue
                    q.put(s)
                    sent += 1
                    if limit and sent >= limit:
                        break
        finally:
            for _ in range(max(1, int(workers))):
                q.put(None)

    def make_hook(worker_idx: int, task_id: int, state: dict) -> callable:
        def hook(d: dict) -> None:
            now = time.time()
            if d.get("status") == "downloading" and (now - state.get("last_ui", 0.0) < 0.2):
                return

            with ui_lock:
                status = d.get("status")
                if status == "downloading":
                    downloaded = d.get("downloaded_bytes") or 0
                    total_bytes = d.get("total_bytes") or d.get("total_bytes_estimate")
                    speed = d.get("speed")
                    eta = d.get("eta")
                    worker_speeds[worker_idx] = float(speed) if speed else 0.0

                    acct = state.get("acct") or ""
                    vid = state.get("vid") or ""
                    desc = (
                        f"Worker {worker_idx+1}({acct}) downloading {vid}".strip()
                        if acct
                        else f"Worker {worker_idx+1} downloading {vid}".strip()
                    )

                    progress.update(
                        task_id,
                        description=desc,
                        completed=downloaded,
                        total=total_bytes,
                        speed=speed,
                        eta=eta,
                        visible=True,
                    )
                    state["last_ui"] = now
                elif status in {"finished", "error"}:
                    worker_speeds[worker_idx] = 0.0
                    progress.update(task_id, speed=None, eta=None, visible=False)

        return hook

    def build_ydl(hook: callable, account: Optional[YouTubeAccount]) -> yt_dlp.YoutubeDL:
        extractor_args: dict[str, dict[str, list[str]]] = {
            "youtubetab": {"approximate_date": ["true"]},
        }
        youtube_args: dict[str, list[str]] = {}
        if getattr(config, "YOUTUBE_PLAYER_CLIENT", None):
            youtube_args["player_client"] = [str(getattr(config, "YOUTUBE_PLAYER_CLIENT"))]
        if invidious_instance:
            youtube_args["invidious_instance"] = [f"https://{invidious_instance}"]
        po_token = getattr(config, "YOUTUBE_PO_TOKEN", None)
        pot_provider = getattr(config, "YOUTUBE_POT_PROVIDER", None)
        if po_token:
            youtube_args["po_token"] = [str(po_token)]
        if pot_provider:
            youtube_args["pot_provider"] = [str(pot_provider)]
        if youtube_args:
            extractor_args["youtube"] = youtube_args
        _merge_extractor_args(extractor_args, getattr(config, "YOUTUBE_EXTRACTOR_ARGS", None))

        sleep_interval_requests = float(getattr(config, "YOUTUBE_SLEEP_REQUESTS", 0.0) or 0.0)
        sleep_interval = float(getattr(config, "YOUTUBE_SLEEP_INTERVAL", 0.0) or 0.0)
        max_sleep_interval = float(getattr(config, "YOUTUBE_MAX_SLEEP_INTERVAL", 0.0) or 0.0)

        ydl_opts: dict = {
            "format": getattr(
                config,
                "YOUTUBE_INPUT_LIST_FORMAT",
                "bv[height<=1080][vcodec^=av01]+ba/bv[height<=1080][vcodec^=vp9]+ba/bv[height<=1080]+ba/best[height<=1080]",
            ),
            "format_sort": (
                str(getattr(config, "YOUTUBE_INPUT_LIST_FORMAT_SORT", "res,+tbr")).split(",")
            ),
            "outtmpl": str(output_dir / "%(title).200B [%(id)s].%(ext)s"),
            "noplaylist": True,
            "retries": int(getattr(config, "YOUTUBE_INPUT_LIST_RETRIES", 3)),
            "fragment_retries": int(getattr(config, "YOUTUBE_INPUT_LIST_RETRIES", 3)),
            "retry_sleep_functions": {
                "http": lambda n: _backoff_sleep(
                    n,
                    float(getattr(config, "YOUTUBE_INPUT_LIST_RETRY_BACKOFF_BASE", 1.0)),
                    float(getattr(config, "YOUTUBE_INPUT_LIST_RETRY_BACKOFF_MAX", 60.0)),
                    float(getattr(config, "YOUTUBE_INPUT_LIST_RETRY_BACKOFF_JITTER", 0.1)),
                ),
                "fragment": lambda n: _backoff_sleep(
                    n,
                    float(getattr(config, "YOUTUBE_INPUT_LIST_RETRY_BACKOFF_BASE", 1.0)),
                    float(getattr(config, "YOUTUBE_INPUT_LIST_RETRY_BACKOFF_MAX", 60.0)),
                    float(getattr(config, "YOUTUBE_INPUT_LIST_RETRY_BACKOFF_JITTER", 0.1)),
                ),
                "file_access": lambda n: _backoff_sleep(
                    n,
                    float(getattr(config, "YOUTUBE_INPUT_LIST_RETRY_BACKOFF_BASE", 1.0)),
                    float(getattr(config, "YOUTUBE_INPUT_LIST_RETRY_BACKOFF_MAX", 60.0)),
                    float(getattr(config, "YOUTUBE_INPUT_LIST_RETRY_BACKOFF_JITTER", 0.1)),
                ),
                "extractor": lambda n: _backoff_sleep(
                    n,
                    float(getattr(config, "YOUTUBE_INPUT_LIST_RETRY_BACKOFF_BASE", 1.0)),
                    float(getattr(config, "YOUTUBE_INPUT_LIST_RETRY_BACKOFF_MAX", 60.0)),
                    float(getattr(config, "YOUTUBE_INPUT_LIST_RETRY_BACKOFF_JITTER", 0.1)),
                ),
            },
            "continuedl": True,
            "quiet": not debug,
            "no_warnings": not debug,
            "noprogress": True,
            "progress_hooks": [hook],
            "overwrites": False,
            "ignoreerrors": False,
            "simulate": False,
            "extractor_args": extractor_args,
            # Use Deno runtime by default (better YouTube support)
            "js_runtimes": {"deno": {}},
            # Allow fetching remote EJS component when needed
            "remote_components": ["ejs:github"],
        }

        if sleep_interval_requests and sleep_interval_requests > 0:
            # yt-dlp option: --sleep-requests (seconds between requests during extraction)
            ydl_opts["sleep_interval_requests"] = sleep_interval_requests
        if sleep_interval and sleep_interval > 0:
            # yt-dlp option: --sleep-interval/--min-sleep-interval (seconds before each download)
            ydl_opts["sleep_interval"] = sleep_interval
            if max_sleep_interval and max_sleep_interval > 0:
                ydl_opts["max_sleep_interval"] = max_sleep_interval

        if account is not None and account.cookies_file.exists():
            ydl_opts["cookiefile"] = str(account.cookies_file)

        return yt_dlp.YoutubeDL(ydl_opts)

    def worker(worker_idx: int) -> None:
        task_id = worker_tasks[worker_idx]
        state = {"last_ui": 0.0, "vid": "", "acct": ""}
        hook = make_hook(worker_idx, task_id, state)

        account_idx = (worker_idx % len(accounts)) if accounts else -1
        account = accounts[account_idx] if account_idx >= 0 else None
        if account is not None:
            state["acct"] = account.name
        ydl = build_ydl(hook, account)

        max_switches_cfg = int(getattr(config, "YOUTUBE_ACCOUNT_SWITCH_MAX", 0) or 0)
        cooldown_sleep_after = float(getattr(config, "YOUTUBE_INPUT_LIST_SLEEP", 0.0) or 0.0)

        while True:
            item = q.get()
            try:
                if item is None:
                    return
                if stop_event.is_set():
                    return

                vid = parse_youtube_id(item) or ""
                state["vid"] = vid

                # UI: preparing
                with ui_lock:
                    worker_speeds[worker_idx] = 0.0
                    acct = state.get("acct") or ""
                    desc = (
                        f"Worker {worker_idx+1}({acct}) preparing {vid}".strip()
                        if acct
                        else f"Worker {worker_idx+1} preparing {vid}".strip()
                    )
                    progress.update(task_id, description=desc, completed=0, total=0, speed=None, eta=None, visible=False)

                # skip processed
                if vid:
                    with processed_lock:
                        already = vid in processed_ids
                    if already:
                        with stats_lock:
                            stats.skipped += 1
                        with ui_lock:
                            progress.advance(overall_task, 1)
                        continue

                # If current account is cooling down, switch to an available one
                if pool is not None and account_idx >= 0:
                    now = time.time()
                    cooled_down = False
                    try:
                        cooled_down = accounts[account_idx].cooldown_until > now
                    except Exception:
                        cooled_down = False
                    if cooled_down:
                        next_idx, wait_s = pool.pick_next(account_idx, exclude_current=True)
                        if wait_s and wait_s > 0:
                            time.sleep(wait_s)
                        if 0 <= next_idx < len(accounts) and next_idx != account_idx:
                            account_idx = next_idx
                            account = accounts[account_idx]
                            state["acct"] = account.name
                            ydl = build_ydl(hook, account)

                url = item
                if vid and not item.startswith("http"):
                    url = f"https://www.youtube.com/watch?v={vid}"

                ok = False
                error_msg = ""
                final_error_type = "failed"

                switches_used = 0
                max_switches = (
                    max_switches_cfg
                    if max_switches_cfg and max_switches_cfg > 0
                    else max(0, len(accounts) - 1)
                )

                while True:
                    try:
                        ret = ydl.download([url])
                        ok = (ret == 0)
                    except DownloadError as e:
                        ok = False
                        error_msg = str(e)
                    except Exception as e:
                        ok = False
                        error_msg = str(e)

                    if ok:
                        break

                    final_error_type = classify_error(error_msg) if error_msg else "failed"

                    # Rate limit: switch account if possible
                    if (
                        final_error_type == "rate_limit"
                        and pool is not None
                        and account_idx >= 0
                        and switches_used < max_switches
                    ):
                        pool.mark_rate_limited(account_idx)
                        next_idx, wait_s = pool.pick_next(account_idx, exclude_current=True)
                        if wait_s and wait_s > 0:
                            time.sleep(wait_s)
                        if next_idx < 0 or next_idx >= len(accounts):
                            break
                        account_idx = next_idx
                        account = accounts[account_idx]
                        state["acct"] = account.name
                        ydl = build_ydl(hook, account)
                        switches_used += 1
                        continue

                    break

                if ok:
                    if vid:
                        with processed_lock:
                            archived_ids.add(vid)
                            processed_ids.add(vid)
                        append_archive_id(archive_file, vid, archive_lock)
                    with stats_lock:
                        stats.ok += 1
                    with ui_lock:
                        progress.update(task_id, visible=False)
                        progress.advance(overall_task, 1)
                else:
                    if final_error_type == "unavailable":
                        if vid:
                            with processed_lock:
                                unavailable_ids.add(vid)
                                processed_ids.add(vid)
                            append_failed_id(unavailable_file, vid, error_msg[:200], unavailable_lock)
                        with stats_lock:
                            stats.unavailable += 1
                    else:
                        if vid:
                            append_failed_id(failed_file, vid, error_msg[:200], failed_lock)
                        with stats_lock:
                            stats.failed += 1
                    with ui_lock:
                        progress.update(task_id, visible=False)
                        progress.advance(overall_task, 1)

                if cooldown_sleep_after and cooldown_sleep_after > 0:
                    time.sleep(cooldown_sleep_after)
            finally:
                q.task_done()

    prod = threading.Thread(target=producer, name="producer", daemon=True)
    prod.start()

    threads: list[threading.Thread] = []
    for i in range(max(1, int(workers))):
        t = threading.Thread(target=worker, args=(i,), name=f"worker-{i+1}", daemon=True)
        threads.append(t)

    try:
        with progress:
            for t in threads:
                t.start()
            for t in threads:
                t.join()
    except KeyboardInterrupt:
        stop_event.set()
        console.print("\nInterrupted. Stopping...")
        return 130

    console.print(
        f"Done. ok={stats.ok} skipped={stats.skipped} unavailable={stats.unavailable} "
        f"failed={stats.failed} output_dir={output_dir}"
    )
    console.print(f"Archive file: {archive_file}")
    console.print(f"Unavailable list: {unavailable_file}")
    console.print(f"Failed list: {failed_file}")
    return 0
