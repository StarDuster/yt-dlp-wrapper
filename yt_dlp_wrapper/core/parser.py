"""
yt-dlp stdout parser for channel downloader.

Goal: keep channel_downloader small by moving the heavy line-parsing logic here.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .diagnostics import classify_channel_error_line, DownloadResult


@dataclass
class ParsedProgress:
    percent: float
    speed: str
    eta: str
    current_file: str
    item_index: Optional[int]
    item_total: Optional[int]


class YtDlpOutputParser:
    """
    Parse yt-dlp output lines and update DownloadResult.

    This mirrors the previous logic in channel_downloader._run_ytdlp_with_progress
    with minimal behavior changes.
    """

    def __init__(
        self,
        *,
        channel_name: str,
        progress_callback: Optional[Callable] = None,
        message_callback: Optional[Callable] = None,
        log_callback: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.channel_name = channel_name
        self.progress_callback = progress_callback
        self.message_callback = message_callback
        self.log_callback = log_callback

        # Regex patterns
        self._progress_pattern = re.compile(r"\[download\]\s+(\d+\.\d+)%.*?at\s+(\S+)\s+ETA\s+(\S+)")
        self._dest_pattern = re.compile(r"\[download\]\s+Destination:\s+(.+)")
        self._already_pattern = re.compile(r"\[download\]\s+(.+)\s+has already been downloaded")
        self._item_pattern = re.compile(r"Downloading\s+(?:video|item)\s+(\d+)\s+of\s+(\d+)", re.IGNORECASE)

        self._video_id_pattern = re.compile(r"\[youtube(?:\+invidious)?\]\s+([a-zA-Z0-9_-]{11}):")
        self._archive_skip_pattern = re.compile(
            r"\[download\]\s+([0-9A-Za-z_-]{11}):\s+has already been recorded in (?:the )?archive",
            re.IGNORECASE,
        )
        self._already_id_pattern = re.compile(
            r"\[download\]\s+([0-9A-Za-z_-]{11})\s+has already been downloaded",
            re.IGNORECASE,
        )

        self._last_print_time = 0.0
        self._current_file = ""
        self._current_video_id: Optional[str] = None
        self._current_item_index: Optional[int] = None
        self._current_item_total: Optional[int] = None

        # Track if current video succeeded
        self._current_video_started = False

        self._seen_item_ids: set[str] = set()
        self._last_item_emit_ts = 0.0

    @property
    def current_file(self) -> str:
        return self._current_file

    @property
    def current_item_index(self) -> Optional[int]:
        return self._current_item_index

    @property
    def current_item_total(self) -> Optional[int]:
        return self._current_item_total

    def set_item_total_hint(self, item_total_hint: Optional[int]) -> None:
        if item_total_hint is None:
            return
        try:
            total_i = int(item_total_hint)
            if total_i > 0:
                self._current_item_total = total_i
                self._current_item_index = 0
        except Exception:
            return

    def _emit_progress(self, percent: float, speed: str, eta: str) -> None:
        if self.progress_callback:
            self.progress_callback(
                percent,
                speed,
                eta,
                self._current_file,
                self._current_item_index,
                self._current_item_total,
            )
            return

        # Fallback print if no callback (limited rate)
        now = time.time()
        if (now - self._last_print_time > 5) or (percent > 99):
            item_str = f"[{self._current_item_index}/{self._current_item_total}] " if self._current_item_index else ""
            print(f"[{self.channel_name}] {item_str}{percent}% ({speed}) ETA {eta}")
            self._last_print_time = now

    def _emit_message(self, msg: str, level: str = "info") -> None:
        if self.message_callback:
            self.message_callback(msg)
            return

        # If no UI callback is provided, prefer structured logging via injected callback.
        if self.log_callback:
            self.log_callback(msg, level)
            return

        # Legacy fallback: print only when there is no progress callback (to avoid noisy output).
        if not self.progress_callback:
            print(msg)

    def _advance_item(self, vid: Optional[str], *, force_emit: bool = False) -> None:
        if not vid:
            return
        if vid in self._seen_item_ids:
            return
        self._seen_item_ids.add(vid)

        if self._current_item_index is None:
            self._current_item_index = 0
        try:
            self._current_item_index = int(self._current_item_index) + 1
        except Exception:
            self._current_item_index = 1

        if self._current_item_total is not None:
            try:
                if self._current_item_index > int(self._current_item_total):
                    self._current_item_index = int(self._current_item_total)
            except Exception:
                pass

        if self.progress_callback:
            now = time.time()
            if force_emit or (now - self._last_item_emit_ts >= 0.25):
                self._emit_progress(0, "...", "...")
                self._last_item_emit_ts = now

    def _classify_error(self, error_line: str) -> str:
        return classify_channel_error_line(error_line)

    def handle_line(self, result: DownloadResult, line: str) -> Optional[DownloadResult]:
        """
        Process one line. Returns a DownloadResult if the caller should abort early,
        otherwise returns None.
        """
        line = (line or "").strip()
        if not line:
            return None

        # Batch-file / archive skip case: yt-dlp may not emit [youtube] lines
        m_arch = self._archive_skip_pattern.search(line)
        if m_arch:
            vid = m_arch.group(1)
            result.already_downloaded_count += 1
            self._advance_item(vid, force_emit=True)
            return None

        m_already_id = self._already_id_pattern.search(line)
        if m_already_id:
            vid = m_already_id.group(1)
            result.already_downloaded_count += 1
            self._advance_item(vid, force_emit=True)
            return None

        # Try to extract current video ID
        vid_match = self._video_id_pattern.search(line)
        if vid_match:
            new_vid = vid_match.group(1)
            if new_vid != self._current_video_id:
                # New video started - if previous one had started without error, count as success
                if self._current_video_started:
                    result.success_count += 1
                self._current_video_id = new_vid
                self._current_video_started = True
                self._advance_item(new_vid)

        # Check for item progress (video x of y)
        match_item = self._item_pattern.search(line)
        if match_item:
            self._current_item_index = int(match_item.group(1))
            self._current_item_total = int(match_item.group(2))
            self._emit_progress(0, "...", "...")
            return None

        # Check for progress
        match_progress = self._progress_pattern.search(line)
        if match_progress:
            percent = float(match_progress.group(1))
            speed = match_progress.group(2)
            eta = match_progress.group(3)
            self._emit_progress(percent, speed, eta)
            return None

        # Check for file destination (new download starting)
        match_dest = self._dest_pattern.search(line)
        if match_dest:
            self._current_file = match_dest.group(1).strip()
            msg = f"[{self.channel_name}] Downloading: {self._current_file}"
            self._emit_message(msg)
            self._emit_progress(0, "...", "...")
            return None

        # Check for already downloaded
        match_already = self._already_pattern.search(line)
        if match_already:
            result.already_downloaded_count += 1
            self._current_video_started = False  # Don't count again
            return None

        # Check for errors - classify them
        is_error_line = "ERROR" in line
        lower_line = line.lower()
        is_bot_detection = (
            "not a bot" in lower_line
            or "sign in to confirm" in lower_line
            or "confirm you're not" in lower_line
            or "confirm you’re not" in lower_line
        )

        if is_error_line or is_bot_detection:
            if is_bot_detection:
                error_type = "rate_limit"
            else:
                error_type = self._classify_error(line)

            video_url_part = ""
            id_match = re.search(r" ([a-zA-Z0-9_-]{11}):", line)
            if id_match:
                video_id = id_match.group(1)
                video_url_part = f" (https://www.youtube.com/watch?v={video_id})"

            if error_type == "rate_limit":
                result.rate_limited_count += 1
                result.rate_limit_errors.append(line)

                self._emit_message(
                    f"[yellow][{self.channel_name}] ⚠️ Bot detected, aborting current attempt: {line}{video_url_part}[/yellow]"
                    if self.message_callback
                    else f"[{self.channel_name}] Rate limit/bot detected! {line}{video_url_part}",
                    "warning",
                )

                self._current_video_started = False
                # Signal caller to abort the running process and return immediately.
                result.return_code = -1
                return result

            if error_type == "members_only":
                result.members_only_count += 1
                result.members_only_errors.append(line)
                if self.message_callback:
                    self.message_callback(f"[blue][{self.channel_name}] 🔒 Members only: {line}{video_url_part}[/blue]")
                else:
                    self._emit_message(f"[{self.channel_name}] Members only: {line}{video_url_part}", "info")
            elif error_type == "unavailable":
                result.unavailable_count += 1
                result.unavailable_errors.append(line)
                if self.message_callback:
                    self.message_callback(f"[dim][{self.channel_name}] ⛔ Unavailable: {line}{video_url_part}[/dim]")
                else:
                    self._emit_message(f"[{self.channel_name}] Unavailable: {line}{video_url_part}", "info")
            else:
                result.other_error_count += 1
                result.other_errors.append(line)
                if self.message_callback:
                    self.message_callback(f"[red][{self.channel_name}] {line}{video_url_part}[/red]")
                else:
                    self._emit_message(f"[{self.channel_name}] {line}{video_url_part}", "error")

            self._current_video_started = False
            return None

        return None

    def finalize(self, result: DownloadResult) -> None:
        if self._current_video_started:
            result.success_count += 1
            self._current_video_started = False

