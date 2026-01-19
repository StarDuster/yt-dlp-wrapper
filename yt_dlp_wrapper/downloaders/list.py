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

import contextlib
import queue
import random
import re
import threading
import time
import os
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.filesize import decimal as fmt_bytes
from rich.progress import BarColumn, Progress, ProgressColumn, SpinnerColumn, TextColumn
from rich.text import Text
try:
    import yt_dlp  # type: ignore
    from yt_dlp.utils import DownloadError  # type: ignore
except Exception:  # pragma: no cover
    # Keep this import optional so unit tests for helper functions can run in minimal envs.
    class _YtDlpStub:
        class YoutubeDL:  # noqa: N801 (match external library naming)
            def __init__(self, *args, **kwargs):
                raise ModuleNotFoundError("yt_dlp is required. Install with: pip install yt-dlp")

    yt_dlp = _YtDlpStub()  # type: ignore

    class DownloadError(Exception):
        pass

from .. import config
from ..auth.pool import YouTubeAccount, YouTubeAccountPool, load_accounts_from_config
from ..core.diagnostics import classify_list_error, diagnose_ffmpeg_error
from ..core.utils import (
    _FFmpegProgressTail,
    _backoff_sleep,
    _format_eta,
    _format_time,
    _get_physical_cpu_cores,
    _has_nvidia_gpu,
    _to_msec,
    append_archive_id,
    append_failed_id,
    load_archive_ids,
    load_failed_ids,
    parse_youtube_id,
)


console = Console()


class _YtdlpLogBuffer:
    """A small ring buffer to keep recent yt-dlp log lines (per worker)."""

    def __init__(self, max_lines: int = 200) -> None:
        try:
            n = int(max_lines)
        except Exception:
            n = 200
        if n < 20:
            n = 20
        if n > 2000:
            n = 2000
        self._lines: deque[str] = deque(maxlen=n)

    def clear(self) -> None:
        self._lines.clear()

    def append(self, msg: object) -> None:
        s = str(msg or "").strip()
        if not s:
            return
        self._lines.append(s)

    def tail(self, max_chars: int = 6000) -> str:
        s = "\n".join(self._lines)
        if not s:
            return ""
        try:
            cap = int(max_chars)
        except Exception:
            cap = 6000
        if cap > 0 and len(s) > cap:
            return s[-cap:]
        return s


class _YtdlpLogger:
    """yt-dlp Python API logger hook."""

    def __init__(self, buf: _YtdlpLogBuffer) -> None:
        self.buf = buf

    def debug(self, msg) -> None:  # noqa: ANN001
        return

    def warning(self, msg) -> None:  # noqa: ANN001
        self.buf.append(msg)

    def error(self, msg) -> None:  # noqa: ANN001
        self.buf.append(msg)


class CountOrBytesColumn(ProgressColumn):
    """Overall: N/M. Worker: bytes downloaded / total."""

    def render(self, task) -> Text:  # type: ignore[override]
        mode = task.fields.get("mode")
        if mode in {None, "idle", "prepare"}:
            return Text("")
        if mode == "count":
            total = int(task.total or 0)
            return Text(f"{int(task.completed)}/{total}")

        if mode == "segment":
            completed_s = float(task.completed or 0.0)
            total_s = task.total
            if total_s is None:
                return Text(f"{_format_time(completed_s)}/?")
            return Text(f"{_format_time(completed_s)}/{_format_time(float(total_s))}")

        completed = int(task.completed or 0)
        total = task.total
        if total is None:
            return Text(f"{fmt_bytes(completed)}/?")
        return Text(f"{fmt_bytes(completed)}/{fmt_bytes(int(total))}")


class SpeedColumn(ProgressColumn):
    """Overall: sum(worker speeds). Worker: current speed."""

    def __init__(
        self,
        worker_speeds: dict[int, float] | None = None,
        worker_item_rates: dict[int, float] | None = None,
    ):
        super().__init__()
        self.worker_speeds = worker_speeds if worker_speeds is not None else {}
        self.worker_item_rates = worker_item_rates if worker_item_rates is not None else {}
        self._last_count_completed: Optional[float] = None
        self._last_count_ts: Optional[float] = None
        self._last_count_rate: Optional[float] = None

    def render(self, task) -> Text:  # type: ignore[override]
        mode = task.fields.get("mode")
        if mode == "count":
            total_speed = sum(self.worker_speeds.values())
            if total_speed > 0:
                return Text(f"{fmt_bytes(int(total_speed))}/s")

            total_item_rate = sum(self.worker_item_rates.values())
            if total_item_rate > 0:
                return Text(f"{total_item_rate:.2f} it/s")

            now = time.time()
            completed = float(task.completed or 0.0)
            if self._last_count_ts is None:
                self._last_count_ts = now
                self._last_count_completed = completed
                self._last_count_rate = 0.0
                return Text("0 it/s")
            last_ts = float(self._last_count_ts)
            last_completed = float(self._last_count_completed or 0.0)
            dc = completed - last_completed

            if dc > 0:
                dt = now - last_ts
                if dt > 0:
                    self._last_count_rate = max(0.0, dc / dt)
                self._last_count_ts = now
                self._last_count_completed = completed
            elif dc < 0:
                self._last_count_ts = now
                self._last_count_completed = completed
                self._last_count_rate = 0.0

            rate = float(self._last_count_rate or 0.0)
            if rate <= 0:
                return Text("0 it/s")
            return Text(f"{rate:.2f} it/s")

        if mode != "download":
            if mode == "segment":
                speed_text = task.fields.get("speed_text")
                if speed_text:
                    return Text(str(speed_text))
                sf = task.fields.get("speed_factor")
                if sf is None:
                    return Text("")
                try:
                    return Text(f"{float(sf):.2f}x")
                except Exception:
                    return Text("")
            return Text("") if mode not in {"idle", "prepare"} else Text("")
        speed = task.fields.get("speed")
        if not speed:
            return Text("")
        try:
            return Text(f"{fmt_bytes(int(speed))}/s")
        except Exception:
            return Text("")


class ETAColumn(ProgressColumn):
    def render(self, task) -> Text:  # type: ignore[override]
        mode = task.fields.get("mode")
        if mode == "count":
            total = task.total
            if total is None:
                return Text("")
            try:
                completed = float(task.completed or 0.0)
                remaining = float(total) - completed
            except Exception:
                return Text("")
            if remaining <= 0:
                return Text(_format_eta(0))

            start_ts = task.fields.get("start_ts")
            start_completed = task.fields.get("start_completed")
            try:
                start_ts_f = float(start_ts) if start_ts is not None else 0.0
            except Exception:
                start_ts_f = 0.0
            if start_ts_f <= 0:
                return Text("")
            elapsed = time.time() - start_ts_f
            if elapsed <= 0:
                return Text("")
            try:
                start_completed_f = float(start_completed or 0.0)
            except Exception:
                start_completed_f = 0.0
            delta_completed = completed - start_completed_f
            if delta_completed <= 0:
                return Text("")
            avg_rate = delta_completed / elapsed
            if avg_rate <= 0:
                return Text("")
            return Text(_format_eta(remaining / avg_rate))

        if mode not in {"download", "segment"}:
            return Text("")
        return Text(_format_eta(task.fields.get("eta")))

def parse_input_line(raw: str) -> Optional[dict]:
    """
    Parse a single input line.

    Supported formats:
    - video_id_or_url
    - video_id_or_url,start_time,end_time   (seconds; float supported)
    """
    s = (raw or "").strip()
    if not s or s.startswith("#"):
        return None

    parts = [p.strip() for p in s.split(",")]
    if len(parts) == 3:
        vid_or_url, start_s, end_s = parts
        vid = parse_youtube_id(vid_or_url)
        if not vid:
            return None
        try:
            start = float(start_s)
            end = float(end_s)
        except Exception:
            return None
        if start < 0 or end < 0 or end <= start:
            return None

        start_ms = _to_msec(start)
        end_ms = _to_msec(end)
        url = vid_or_url if vid_or_url.startswith("http") else f"https://www.youtube.com/watch?v={vid}"
        return {
            "raw": s,
            "url": url,
            "vid": vid,
            "start": float(start),
            "end": float(end),
            "start_ms": int(start_ms),
            "end_ms": int(end_ms),
            "key": f"{vid}:{start_ms}-{end_ms}",
            "has_range": True,
        }

    vid = parse_youtube_id(s)
    if not vid:
        return None
    url = s if s.startswith("http") else f"https://www.youtube.com/watch?v={vid}"
    return {
        "raw": s,
        "url": url,
        "vid": vid,
        "start": None,
        "end": None,
        "start_ms": None,
        "end_ms": None,
        "key": vid,
        "has_range": False,
    }


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

def classify_error(error_msg: str) -> str:
    return classify_list_error(error_msg)


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
    disable_nvenc: bool = False,
) -> int:
    input_list_path = input_list_path.expanduser().resolve()
    if not input_list_path.exists():
        raise FileNotFoundError(f"Input list not found: {input_list_path}")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preload yt-dlp plugins in the main thread to avoid multi-threaded import races.
    try:
        from yt_dlp.plugins import load_all_plugins

        load_all_plugins()
    except Exception:
        pass

    accounts, pool = load_accounts_from_config(accounts_dir=accounts_dir)
    if accounts:
        console.print(
            f"Account pool enabled: {len(accounts)} account(s) "
            f"(auto-switch={'yes' if pool else 'no'})"
        )

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
    unavailable_this_run_file = output_dir / "unavailable.this_run.txt"
    failed_this_run_file = output_dir / "failed.this_run.txt"

    archive_lock = threading.Lock()
    unavailable_lock = threading.Lock()
    failed_lock = threading.Lock()
    processed_lock = threading.Lock()
    this_run_lock = threading.Lock()
    unavailable_this_run: list[tuple[str, str]] = []
    failed_this_run: list[tuple[str, str]] = []

    archived_ids = load_archive_ids(archive_file)
    unavailable_ids = load_failed_ids(unavailable_file)
    processed_ids: set[str] = set(archived_ids) | set(unavailable_ids)

    total = 0
    already_done = 0
    has_range_tasks = False
    with input_list_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            item = parse_input_line(line)
            if not item:
                continue
            total += 1
            if item.get("has_range"):
                has_range_tasks = True
            key = item.get("key")
            if key and key in processed_ids:
                already_done += 1
            if limit and total >= limit:
                break

    if total <= 0:
        console.print("Input list is empty or contains no valid lines")
        return 0

    console.print(
        f"Job summary: total={total} processed={already_done} remaining={total - already_done} "
        f"output={output_dir}"
    )

    has_nvidia_gpu = False
    nvenc_sem: Optional[threading.Semaphore] = None
    nvenc_concurrency = 0
    if has_range_tasks:
        console.print(
            "Detected timestamp ranges (video_id,start,end). Segment tasks will use precise cutting + transcoding "
            "(--force-keyframes-at-cuts)."
        )
        gpu_available = _has_nvidia_gpu()
        disable_nvenc_effective = bool(disable_nvenc)
        if gpu_available and not disable_nvenc_effective:
            has_nvidia_gpu = True
            console.print("Detected NVIDIA GPU. Using NVENC (h264_nvenc) for transcoding.")
            try:
                nvenc_concurrency = int(getattr(config, "YOUTUBE_NVENC_CONCURRENCY", 3) or 3)
            except Exception:
                nvenc_concurrency = 3
            if nvenc_concurrency <= 0:
                nvenc_concurrency = 3
            nvenc_concurrency = min(nvenc_concurrency, max(1, int(workers)))
            nvenc_sem = threading.BoundedSemaphore(nvenc_concurrency)
            try:
                _eff_workers = max(1, int(workers))
            except Exception:
                _eff_workers = 1
            if _eff_workers > nvenc_concurrency:
                console.print(
                    f"[yellow][bold]NVENC slots: {nvenc_concurrency}. Extra workers will fall back to libx264 (CPU) instead of waiting.[/bold][/yellow]"
                )
            else:
                console.print(f"NVENC concurrency limit: {nvenc_concurrency}")
            try:
                seg_max_h = int(getattr(config, "YOUTUBE_SEGMENT_MAX_HEIGHT", 1080) or 1080)
            except Exception:
                seg_max_h = 1080
            preset = getattr(config, "YOUTUBE_NVENC_PRESET", "p1")
            tune = getattr(config, "YOUTUBE_NVENC_TUNE", "ll")
            rc = getattr(config, "YOUTUBE_NVENC_RC", "constqp")
            qp = getattr(config, "YOUTUBE_NVENC_QP", 18)
            qp_part = ""
            try:
                if str(rc) == "constqp":
                    qp_part = f" qp={int(qp)}"
            except Exception:
                qp_part = ""
            console.print(
                f"NVENC params: preset={preset} tune={tune} rc={rc}{qp_part} | segment_max_height={seg_max_h} | audio=copy"
            )
        else:
            has_nvidia_gpu = False
            if gpu_available and disable_nvenc_effective:
                console.print(
                    "[yellow][bold]Detected NVIDIA GPU, but NVENC is disabled. Forcing libx264 (CPU) transcoding.[/bold][/yellow]"
                )
            else:
                console.print("[red][bold]No NVIDIA GPU detected. Using libx264 transcoding (may be slow).[/bold][/red]")
            try:
                cpu_cores = int(_get_physical_cpu_cores() or 1)
            except Exception:
                cpu_cores = 1
            try:
                eff_workers = max(1, int(workers))
            except Exception:
                eff_workers = 1
            if cpu_cores > 0 and eff_workers >= cpu_cores:
                console.print(
                    f"[red][bold]⚠ You have {eff_workers} workers but only {cpu_cores} physical CPU cores. "
                    f"Running {eff_workers} parallel libx264 encodes will likely max out your CPU! "
                    "Consider using fewer --workers.[/bold][/red]"
                )

    worker_speeds: dict[int, float] = {}
    worker_item_rates: dict[int, float] = {}
    ui_lock = threading.Lock()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=None),
        CountOrBytesColumn(),
        SpeedColumn(worker_speeds=worker_speeds, worker_item_rates=worker_item_rates),
        ETAColumn(),
        console=console,
        refresh_per_second=10,
    )

    overall_task = progress.add_task(
        "Overall",
        total=total,
        completed=already_done,
        mode="count",
        start_ts=time.time(),
        start_completed=already_done,
    )
    worker_tasks = [
        progress.add_task(
            f"Worker {i+1}: idle",
            total=1,
            completed=0,
            mode="idle",
            speed=None,
            eta=None,
            speed_factor=None,
            visible=True,
        )
        for i in range(max(1, int(workers)))
    ]

    q: queue.Queue[Optional[dict]] = queue.Queue(maxsize=max(2, int(workers) * 2))
    stop_event = threading.Event()

    stats = Stats(skipped=already_done)
    stats_lock = threading.Lock()

    def producer() -> None:
        seen = 0
        try:
            with input_list_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if stop_event.is_set():
                        break
                    item = parse_input_line(line)
                    if not item:
                        continue
                    seen += 1
                    key = str(item.get("key") or "")
                    already = False
                    if key:
                        with processed_lock:
                            already = key in processed_ids
                    if not already:
                        q.put(item)
                    if limit and seen >= limit:
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
                    worker_item_rates[worker_idx] = 0.0

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
                        mode="download",
                        speed=speed,
                        eta=eta,
                        visible=True,
                    )
                    state["last_ui"] = now
                elif status in {"finished", "error"}:
                    worker_speeds[worker_idx] = 0.0
                    worker_item_rates[worker_idx] = 0.0
                    acct = state.get("acct") or ""
                    desc = (
                        f"Worker {worker_idx+1}({acct}) idle".strip()
                        if acct
                        else f"Worker {worker_idx+1}: idle"
                    )
                    progress.update(
                        task_id,
                        description=desc,
                        completed=0,
                        total=1,
                        mode="idle",
                        speed=None,
                        eta=None,
                        speed_factor=None,
                        visible=True,
                    )

        return hook

    def build_ydl(
        hook: callable,
        account: Optional[YouTubeAccount],
        segment: Optional[dict] = None,
        force_libx264: bool = False,
        *,
        logger_obj: Optional[object] = None,
    ) -> yt_dlp.YoutubeDL:
        extractor_args: dict[str, dict[str, list[str]]] = {
            "youtubetab": {"approximate_date": ["true"]},
        }
        youtube_args: dict[str, list[str]] = {}
        player_client_raw = os.environ.get("YTDLP_WRAPPER_PLAYER_CLIENT") or getattr(config, "YOUTUBE_PLAYER_CLIENT", None)
        if player_client_raw:
            if isinstance(player_client_raw, (list, tuple)):
                player_clients = [str(x).strip() for x in player_client_raw if str(x).strip()]
            else:
                player_clients = [s.strip() for s in str(player_client_raw).split(",") if s.strip()]
            if player_clients:
                youtube_args["player_client"] = player_clients
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

        if has_range_tasks:
            sleep_interval_requests = 0.0
            sleep_interval = 0.0
            max_sleep_interval = 0.0

        segment_mode = bool(segment and segment.get("has_range"))
        ydl_format = getattr(
            config,
            "YOUTUBE_INPUT_LIST_FORMAT",
            "bv[height<=1080][vcodec^=av01]+ba/bv[height<=1080][vcodec^=vp9]+ba/bv[height<=1080]+ba/best[height<=1080]",
        )
        if segment_mode:
            try:
                seg_max_h = int(getattr(config, "YOUTUBE_SEGMENT_MAX_HEIGHT", 1080) or 1080)
            except Exception:
                seg_max_h = 1080
            if seg_max_h > 0:
                ydl_format = (
                    f"bestvideo[ext=mp4][height<={seg_max_h}]+bestaudio[ext=m4a]/"
                    f"best[ext=mp4][height<={seg_max_h}]/"
                    f"best[height<={seg_max_h}]"
                )
            else:
                ydl_format = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"

        outtmpl = str(output_dir / "%(title).200B [%(id)s].%(ext)s")
        if segment_mode:
            start_ms = int(segment.get("start_ms") or 0)
            end_ms = int(segment.get("end_ms") or 0)
            outtmpl = str(output_dir / f"%(title).200B [%(id)s] [{start_ms}-{end_ms}].mp4")

        ydl_opts: dict = {
            "format": ydl_format,
            "format_sort": (
                str(getattr(config, "YOUTUBE_INPUT_LIST_FORMAT_SORT", "res,+tbr")).split(",")
            ),
            "outtmpl": outtmpl,
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
            "js_runtimes": {"deno": {}},
            "remote_components": ["ejs:github"],
        }
        if logger_obj is not None:
            ydl_opts["logger"] = logger_obj

        if segment_mode:
            start = float(segment.get("start") or 0.0)
            end = float(segment.get("end") or 0.0)

            def _ranges(_info, _ydl):
                return [{"start_time": start, "end_time": end}]

            ydl_opts["download_ranges"] = _ranges
            ydl_opts["force_keyframes_at_cuts"] = True
            # Make sure output timestamps start from 0 (avoid negative/shifted PTS)
            ydl_opts.setdefault("external_downloader_args", {})
            ydl_opts["external_downloader_args"].setdefault("ffmpeg_o", [])
            ydl_opts["external_downloader_args"]["ffmpeg_o"] += [
                "-avoid_negative_ts",
                "make_zero",
            ]
            # Ensure segment outputs are encodable. Prefer NVENC if GPU available and not forced to use libx264.
            use_nvenc = has_nvidia_gpu and not force_libx264
            if use_nvenc:
                ydl_opts["external_downloader_args"]["ffmpeg_o"] += [
                    "-c:v",
                    "h264_nvenc",
                ]
                preset = getattr(config, "YOUTUBE_NVENC_PRESET", None)
                if preset:
                    ydl_opts["external_downloader_args"]["ffmpeg_o"] += ["-preset", str(preset)]
                tune = getattr(config, "YOUTUBE_NVENC_TUNE", None)
                if tune:
                    ydl_opts["external_downloader_args"]["ffmpeg_o"] += ["-tune", str(tune)]
                rc = getattr(config, "YOUTUBE_NVENC_RC", None)
                if rc:
                    ydl_opts["external_downloader_args"]["ffmpeg_o"] += ["-rc", str(rc)]
                    if str(rc) == "constqp":
                        qp = getattr(config, "YOUTUBE_NVENC_QP", None)
                        if qp is not None:
                            with contextlib.suppress(Exception):
                                ydl_opts["external_downloader_args"]["ffmpeg_o"] += ["-qp", str(int(qp))]
                ydl_opts["external_downloader_args"]["ffmpeg_o"] += [
                    "-c:a",
                    "copy",
                ]
            else:
                ydl_opts["external_downloader_args"]["ffmpeg_o"] += [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-c:a",
                    "copy",
                ]
            # ffmpeg -progress output: write to a file so we can show per-worker progress in the UI
            progress_path = segment.get("ffmpeg_progress_path")
            if progress_path:
                ydl_opts["external_downloader_args"]["ffmpeg_o"] += [
                    "-nostats",
                    "-stats_period",
                    "0.5",
                    "-progress",
                    str(progress_path),
                ]

        if sleep_interval_requests and sleep_interval_requests > 0:
            ydl_opts["sleep_interval_requests"] = sleep_interval_requests
        if sleep_interval and sleep_interval > 0:
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
        logbuf = _YtdlpLogBuffer(max_lines=200)
        ydl_logger = _YtdlpLogger(logbuf)
        ydl_base = build_ydl(hook, account, logger_obj=ydl_logger)
        ffmpeg_progress_dir = output_dir / ".ffmpeg-progress"
        try:
            ffmpeg_progress_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        ffmpeg_progress_file = ffmpeg_progress_dir / f"worker-{worker_idx+1}.txt"

        max_switches_cfg = int(getattr(config, "YOUTUBE_ACCOUNT_SWITCH_MAX", 0) or 0)
        cooldown_sleep_after = float(getattr(config, "YOUTUBE_INPUT_LIST_SLEEP", 0.0) or 0.0)
        if has_range_tasks:
            cooldown_sleep_after = 0.0

        while True:
            item = q.get()
            try:
                if item is None:
                    break
                if stop_event.is_set():
                    break

                # Reset per-item yt-dlp log buffer so DownloadError gets useful context.
                logbuf.clear()

                vid = str(item.get("vid") or "")
                url = str(item.get("url") or "")
                item_key = str(item.get("key") or vid)
                state["vid"] = item_key

                with ui_lock:
                    worker_speeds[worker_idx] = 0.0
                    worker_item_rates[worker_idx] = 0.0
                    acct = state.get("acct") or ""
                    desc = (
                        f"Worker {worker_idx+1}({acct}) preparing {vid}".strip()
                        if acct
                        else f"Worker {worker_idx+1} preparing {vid}".strip()
                    )
                    progress.update(
                        task_id,
                        description=desc,
                        completed=0,
                        total=1,
                        mode="prepare",
                        speed=None,
                        eta=None,
                        speed_factor=None,
                        visible=True,
                    )

                if item_key:
                    with processed_lock:
                        already = item_key in processed_ids
                    if already:
                        with stats_lock:
                            stats.skipped += 1
                        with ui_lock:
                            progress.advance(overall_task, 1)
                        continue

                if pool is not None and accounts:
                    while True:
                        next_idx, wait_s = pool.pick_least_used()
                        if wait_s and wait_s > 0:
                            time.sleep(wait_s)
                            continue
                        break
                    if 0 <= next_idx < len(accounts):
                        if next_idx != account_idx or account is None:
                            account_idx = next_idx
                            account = accounts[account_idx]
                            with contextlib.suppress(Exception):
                                ydl_base.close()
                            ydl_base = build_ydl(hook, account, logger_obj=ydl_logger)
                        try:
                            state["acct"] = str(account.name) if account is not None else ""
                        except Exception:
                            state["acct"] = ""

                if not url and vid:
                    url = f"https://www.youtube.com/watch?v={vid}"

                ok = False
                error_msg = ""
                final_error_type = "failed"
                error_hint: str = ""

                switches_used = 0
                max_switches = (
                    max_switches_cfg
                    if max_switches_cfg and max_switches_cfg > 0
                    else max(0, len(accounts) - 1)
                )

                segment_mode = bool(item.get("has_range"))
                if segment_mode:
                    item["ffmpeg_progress_path"] = str(ffmpeg_progress_file)

                nvenc_enabled = bool(segment_mode and has_nvidia_gpu and nvenc_sem is not None)
                segment_duration = None
                if segment_mode:
                    try:
                        segment_duration = float(item.get("end") or 0.0) - float(item.get("start") or 0.0)
                    except Exception:
                        segment_duration = None
                    if segment_duration is not None and segment_duration > 0:
                        with ui_lock:
                            acct = state.get("acct") or ""
                            desc = (
                                f"Worker {worker_idx+1}({acct}) transcoding {item_key}".strip()
                                if acct
                                else f"Worker {worker_idx+1} transcoding {item_key}".strip()
                            )
                            progress.update(
                                task_id,
                                description=desc,
                                completed=0.0,
                                total=segment_duration,
                                mode="segment",
                                speed_factor=None,
                                eta=None,
                                visible=True,
                            )

                attempt = 0
                ffmpeg_failures = 0
                rate_limit_hits = 0
                try:
                    ffmpeg_max_retries = int(getattr(config, "YOUTUBE_FFMPEG_MAX_RETRIES", 3) or 3)
                except Exception:
                    ffmpeg_max_retries = 3
                if ffmpeg_max_retries < 1:
                    ffmpeg_max_retries = 1
                try:
                    ffmpeg_retry_sleep = float(getattr(config, "YOUTUBE_FFMPEG_RETRY_SLEEP", 3.0) or 3.0)
                except Exception:
                    ffmpeg_retry_sleep = 3.0
                if ffmpeg_retry_sleep < 0:
                    ffmpeg_retry_sleep = 0.0
                while True:
                    attempt += 1
                    nvenc_acquired = False
                    use_nvenc_attempt = False
                    if segment_mode and nvenc_enabled and nvenc_sem is not None:
                        acquired_immediately = False
                        try:
                            acquired_immediately = bool(nvenc_sem.acquire(blocking=False))
                        except TypeError:
                            acquired_immediately = False
                        except Exception:
                            acquired_immediately = False
                        if acquired_immediately:
                            nvenc_acquired = True
                            use_nvenc_attempt = True

                    ydl_for_item = (
                        build_ydl(
                            hook,
                            account,
                            segment=item,
                            force_libx264=(segment_mode and not use_nvenc_attempt),
                            logger_obj=ydl_logger,
                        )
                        if segment_mode
                        else ydl_base
                    )

                    if segment_mode and segment_duration and segment_duration > 0:
                        encoder = "NVENC" if use_nvenc_attempt else "libx264"
                        if (not use_nvenc_attempt) and nvenc_enabled:
                            encoder = "libx264 (NVENC busy)"
                        with ui_lock:
                            acct = state.get("acct") or ""
                            desc = (
                                f"Worker {worker_idx+1}({acct}) transcoding {item_key} [{encoder}]".strip()
                                if acct
                                else f"Worker {worker_idx+1} transcoding {item_key} [{encoder}]".strip()
                            )
                            progress.update(
                                task_id,
                                description=desc,
                                mode="segment",
                                speed_factor=None,
                                speed_text=None,
                                eta=None,
                                visible=True,
                            )

                    monitor_stop: Optional[threading.Event] = None
                    monitor_thread: Optional[threading.Thread] = None
                    monitor_started_at = time.time()
                    if segment_mode and segment_duration and segment_duration > 0:
                        try:
                            ffmpeg_progress_file.write_text("", encoding="utf-8")
                        except Exception:
                            pass
                        monitor_stop = threading.Event()
                        tail = _FFmpegProgressTail(ffmpeg_progress_file)

                        def _monitor() -> None:
                            while not monitor_stop.is_set():
                                tail.poll()
                                completed_s = min(float(segment_duration), max(0.0, float(tail.out_time_s or 0.0)))
                                if completed_s <= 0.0:
                                    completed_s = min(float(segment_duration), max(0.0, time.time() - monitor_started_at))
                                sf = tail.speed_factor
                                speed_text = tail.speed_text
                                eta = None
                                if sf is not None and sf > 0 and segment_duration:
                                    eta = max(0.0, (float(segment_duration) - completed_s) / float(sf))
                                item_rate = 0.0
                                try:
                                    if segment_duration and float(segment_duration) > 0:
                                        if sf is not None and float(sf) > 0:
                                            item_rate = max(0.0, float(sf) / float(segment_duration))
                                except Exception:
                                    item_rate = 0.0
                                with ui_lock:
                                    worker_item_rates[worker_idx] = item_rate
                                    progress.update(
                                        task_id,
                                        completed=completed_s,
                                        total=segment_duration,
                                        mode="segment",
                                        speed_factor=sf,
                                        speed_text=speed_text,
                                        eta=eta,
                                        visible=True,
                                    )
                                time.sleep(0.2)

                        monitor_thread = threading.Thread(target=_monitor, name=f"ffmpeg-progress-{worker_idx+1}", daemon=True)
                        monitor_thread.start()

                    try:
                        ret = ydl_for_item.download([url])
                        ok = (ret == 0)
                    except DownloadError as e:
                        ok = False
                        error_msg = str(e)
                        ctx = logbuf.tail()
                        if ctx and ctx not in error_msg:
                            error_msg = f"{error_msg}\n\n--- ytdlp_log_tail ---\n{ctx}"
                    except Exception as e:
                        ok = False
                        error_msg = str(e)
                        ctx = logbuf.tail()
                        if ctx and ctx not in error_msg:
                            error_msg = f"{error_msg}\n\n--- ytdlp_log_tail ---\n{ctx}"
                    finally:
                        if monitor_stop is not None:
                            monitor_stop.set()
                        if monitor_thread is not None:
                            with contextlib.suppress(Exception):
                                monitor_thread.join(timeout=1.0)
                        with ui_lock:
                            worker_item_rates[worker_idx] = 0.0
                        if nvenc_acquired and nvenc_sem is not None:
                            try:
                                nvenc_sem.release()
                            except Exception:
                                pass
                        if segment_mode:
                            with contextlib.suppress(Exception):
                                ydl_for_item.close()

                    if ok:
                        break

                    final_error_type = classify_error(error_msg) if error_msg else "failed"

                    # ffmpeg failures: do not hard-code specific exit codes.
                    # In practice this often means invalid/incomplete input (bad fragments, HTML/403, broken stream, etc.).
                    ffmpeg_exit_code: Optional[int] = None
                    if error_msg:
                        m = re.search(r"ffmpeg exited with code\s+(-?\d+)", error_msg, flags=re.IGNORECASE)
                        if m:
                            with contextlib.suppress(Exception):
                                ffmpeg_exit_code = int(m.group(1))

                    if ffmpeg_exit_code is not None:
                        _diag, _http, _hint = diagnose_ffmpeg_error(error_msg)
                        if _hint:
                            error_hint = _hint
                        hint_for_log = f", {_hint}" if _hint else ""
                        switch_for_log = ""

                        # If the underlying cause looks like rate limiting, do NOT consume the
                        # ffmpeg retry budget. Instead, treat it like a "wait/switch and retry later" event.
                        if _diag == "rate_limit":
                            rate_limit_hits += 1

                            if pool is not None and account_idx >= 0 and accounts:
                                old_idx = int(account_idx)
                                old_name = ""
                                try:
                                    old_name = str(accounts[old_idx].name) if 0 <= old_idx < len(accounts) else str(old_idx)
                                except Exception:
                                    old_name = str(old_idx)

                                with contextlib.suppress(Exception):
                                    pool.mark_rate_limited(old_idx)

                                while True:
                                    next_idx, wait_s = pool.pick_least_used(exclude_idx=old_idx)
                                    if wait_s and wait_s > 0:
                                        time.sleep(wait_s)
                                        continue
                                    break
                                if 0 <= next_idx < len(accounts) and next_idx != old_idx:
                                    account_idx = next_idx
                                    account = accounts[account_idx]
                                    state["acct"] = account.name
                                    with contextlib.suppress(Exception):
                                        ydl_base.close()
                                    ydl_base = build_ydl(hook, account, logger_obj=ydl_logger)
                                    switches_used += 1
                                    try:
                                        new_name = str(accounts[account_idx].name) if 0 <= account_idx < len(accounts) else str(account_idx)
                                    except Exception:
                                        new_name = str(account_idx)
                                    if old_name and new_name and old_name != new_name:
                                        switch_for_log = f", switch account {old_name} -> {new_name}"

                            with ui_lock:
                                progress.console.print(
                                    f"[yellow]Worker {worker_idx+1}: ffmpeg failed (code={ffmpeg_exit_code}{hint_for_log}), "
                                    f"sleep {ffmpeg_retry_sleep:g}s then retry {item_key} "
                                    f"(rate_limited={rate_limit_hits}){switch_for_log}[/yellow]"
                                )

                            if ffmpeg_retry_sleep:
                                time.sleep(ffmpeg_retry_sleep)
                            continue

                        # Non-rate-limit ffmpeg failure: consume retry budget and potentially give up.
                        ffmpeg_failures += 1
                        reached_max = bool(ffmpeg_failures >= ffmpeg_max_retries)

                        with ui_lock:
                            if reached_max:
                                progress.console.print(
                                    f"[red]Worker {worker_idx+1}: ffmpeg failed repeatedly (code={ffmpeg_exit_code}{hint_for_log}) {item_key} "
                                    f"(retries={ffmpeg_failures}/{ffmpeg_max_retries}), giving up[/red]"
                                )
                            else:
                                progress.console.print(
                                    f"[yellow]Worker {worker_idx+1}: ffmpeg failed (code={ffmpeg_exit_code}{hint_for_log}), "
                                    f"sleep {ffmpeg_retry_sleep:g}s then retry {item_key} "
                                    f"(retries={ffmpeg_failures}/{ffmpeg_max_retries})[/yellow]"
                                )

                        if reached_max:
                            # When ffmpeg retries are exhausted, decide whether to permanently skip.
                            # Empirically, ffmpeg exit code 222 often means the media URL is not accessible
                            # (bad fragments/HTML/403-like responses). For large list jobs this is usually
                            # better treated as UNAVAILABLE so we don't repeatedly waste retries and account switches.
                            if ffmpeg_exit_code == 222 and _diag != "rate_limit":
                                final_error_type = "unavailable"
                            break

                        if ffmpeg_retry_sleep:
                            time.sleep(ffmpeg_retry_sleep)
                        continue

                    if (
                        final_error_type == "rate_limit"
                        and pool is not None
                        and account_idx >= 0
                        and switches_used < max_switches
                    ):
                        try:
                            _old_idx = int(account_idx)
                            _old_name = str(accounts[_old_idx].name) if 0 <= _old_idx < len(accounts) else None
                        except Exception:
                            _old_idx = account_idx
                            _old_name = None

                        pool.mark_rate_limited(account_idx)
                        while True:
                            next_idx, wait_s = pool.pick_least_used(exclude_idx=account_idx)
                            if wait_s and wait_s > 0:
                                time.sleep(wait_s)
                                continue
                            break
                        try:
                            _new_name = str(accounts[next_idx].name) if 0 <= next_idx < len(accounts) else None
                        except Exception:
                            _new_name = None
                        if next_idx < 0 or next_idx >= len(accounts):
                            break
                        with ui_lock:
                            progress.console.print(
                                f"[yellow]Worker {worker_idx+1}: rate-limited on {_old_name or _old_idx}, switch -> {_new_name or next_idx}[/yellow]"
                            )
                        account_idx = next_idx
                        account = accounts[account_idx]
                        state["acct"] = account.name
                        with contextlib.suppress(Exception):
                            ydl_base.close()
                        ydl_base = build_ydl(hook, account, logger_obj=ydl_logger)
                        switches_used += 1
                        continue

                    break

                if ok:
                    if vid:
                        with processed_lock:
                            archived_ids.add(item_key)
                            processed_ids.add(item_key)
                        append_archive_id(archive_file, item_key, archive_lock)
                    with stats_lock:
                        stats.ok += 1
                    with ui_lock:
                        worker_item_rates[worker_idx] = 0.0
                        acct = state.get("acct") or ""
                        desc = (
                            f"Worker {worker_idx+1}({acct}) idle".strip()
                            if acct
                            else f"Worker {worker_idx+1}: idle"
                        )
                        progress.update(
                            task_id,
                            description=desc,
                            completed=0,
                            total=1,
                            mode="idle",
                            speed=None,
                            eta=None,
                            speed_factor=None,
                            visible=True,
                        )
                        progress.advance(overall_task, 1)
                else:
                    if final_error_type == "unavailable":
                        if vid:
                            with processed_lock:
                                unavailable_ids.add(item_key)
                                processed_ids.add(item_key)
                            _reason = (error_msg or "")[:200].replace("\n", " ").replace("\r", " ").strip()
                            if error_hint:
                                _reason = f"{error_hint} | {_reason}" if _reason else error_hint
                            append_failed_id(unavailable_file, item_key, _reason, unavailable_lock)
                            with this_run_lock:
                                unavailable_this_run.append((item_key, _reason))
                        with stats_lock:
                            stats.unavailable += 1
                    else:
                        if vid:
                            _reason = (error_msg or "")[:200].replace("\n", " ").replace("\r", " ").strip()
                            if error_hint:
                                _reason = f"{error_hint} | {_reason}" if _reason else error_hint
                            append_failed_id(failed_file, item_key, _reason, failed_lock)
                            with this_run_lock:
                                failed_this_run.append((item_key, _reason))
                        with stats_lock:
                            stats.failed += 1
                    with ui_lock:
                        worker_item_rates[worker_idx] = 0.0
                        acct = state.get("acct") or ""
                        desc = (
                            f"Worker {worker_idx+1}({acct}) idle".strip()
                            if acct
                            else f"Worker {worker_idx+1}: idle"
                        )
                        progress.update(
                            task_id,
                            description=desc,
                            completed=0,
                            total=1,
                            mode="idle",
                            speed=None,
                            eta=None,
                            speed_factor=None,
                            visible=True,
                        )
                        progress.advance(overall_task, 1)

                if cooldown_sleep_after and cooldown_sleep_after > 0:
                    time.sleep(cooldown_sleep_after)
            finally:
                q.task_done()
        with contextlib.suppress(Exception):
            ydl_base.close()

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

    # Write per-run failure snapshots to avoid confusion with the cumulative logs.
    # The cumulative files (failed.txt/unavailable.txt) are append-only across runs,
    # so an item may appear there even if it succeeds later and is in the archive.
    try:
        with unavailable_this_run_file.open("w", encoding="utf-8") as f:
            for k, r in unavailable_this_run:
                f.write(f"{k}\t{r}\n")
    except Exception:
        pass
    try:
        with failed_this_run_file.open("w", encoding="utf-8") as f:
            for k, r in failed_this_run:
                f.write(f"{k}\t{r}\n")
    except Exception:
        pass

    console.print(
        f"Done. ok={stats.ok} skipped={stats.skipped} unavailable={stats.unavailable} "
        f"failed={stats.failed} output_dir={output_dir}"
    )
    console.print(f"Archive file: {archive_file}")
    console.print(f"Unavailable list: {unavailable_file}")
    console.print(f"Failed list: {failed_file}")
    console.print(f"Unavailable this run: {unavailable_this_run_file}")
    console.print(f"Failed this run: {failed_this_run_file}")
    return 0
