"""
YouTube channel downloader using yt-dlp.
"""

import logging
import hashlib
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

from .. import config
from ..auth.pool import YouTubeAccount, YouTubeAccountPool
from .error_diagnosis import classify_channel_error_line
from .models import DownloadResult

logger = logging.getLogger(__name__)


class YouTubeDownloader:
    """Download audio from YouTube channels/playlists using yt-dlp"""

    def __init__(self, download_dir: Optional[Path] = None, *, skip_dependency_checks: bool = False):
        self.download_dir = download_dir or config.YOUTUBE_DOWNLOAD_DIR
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

        if not skip_dependency_checks:
            if not self._check_ytdlp():
                self.logger.error("yt-dlp not found. Install with: pip install yt-dlp")
                raise RuntimeError("yt-dlp is required but not installed")

            if not self._check_deno():
                self.logger.warning(
                    "Deno not found. Install for better YouTube support: curl -fsSL https://deno.land/install.sh | sh"
                )

    def _check_ytdlp(self) -> bool:
        """Check if yt-dlp is installed"""
        try:
            result = subprocess.run(
                ["yt-dlp", "--version"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def _check_deno(self) -> bool:
        """Check if deno is installed"""
        try:
            result = subprocess.run(
                ["deno", "--version"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def _load_global_archive(self) -> set:
        """
        Load global video IDs from CSV to skip (optional).
        Returns:
            Set of video IDs to skip
        """
        global_ids = set()
        csv_path = getattr(config, "YOUTUBE_GLOBAL_ARCHIVE_CSV", None)
        if not csv_path:
            return global_ids

        csv_path = Path(csv_path).expanduser().resolve()
        if not csv_path.exists():
            self.logger.warning(f"Global archive file not found: {csv_path}")
            return global_ids

        try:
            import csv

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if row and len(row) > 0:
                        global_ids.add(row[0])
            self.logger.info(f"Loaded {len(global_ids)} videos from global archive")
        except Exception as e:
            self.logger.error(f"Error loading global archive: {e}")

        return global_ids

    def _log_msg(self, msg: str, level: str = "info", message_callback: Optional[Callable] = None):
        """Helper to log via callback if available, else standard logger"""
        if message_callback:
            style = ""
            if level == "error":
                style = "[red]"
                end_style = "[/red]"
            elif level == "warning":
                style = "[yellow]"
                end_style = "[/yellow]"
            elif level == "debug":
                style = "[dim]"
                end_style = "[/dim]"
            else:
                style = ""
                end_style = ""
            
            # Remove timestamp if present in msg as rich adds its own or just text
            message_callback(f"{style}{msg}{end_style}")
        else:
            if level == "debug":
                self.logger.debug(msg)
            elif level == "warning":
                self.logger.warning(msg)
            elif level == "error":
                self.logger.error(msg)
            else:
                self.logger.info(msg)

    def _derive_channel_key(self, url: str) -> str:
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

    def _get_channel_video_ids(
        self,
        url: str,
        cache_file: Optional[Path] = None,
        cache_ttl_days: Optional[float] = None,
        debug: bool = False,
        message_callback: Optional[Callable] = None,
    ) -> list[str]:
        """
        Get all video IDs from a channel without downloading.

        If cache_file is provided:
        - Use it if it exists, is non-empty, and not expired (by mtime + cache_ttl_days).
        - Otherwise fetch from network and refresh it (atomic write).
        """
        start_time = time.time()

        yt_id_re = re.compile(r"^[0-9A-Za-z_-]{11}$")
        watch_id_re = re.compile(r"(?:[?&]v=)([0-9A-Za-z_-]{11})")

        def _is_cache_fresh(path: Path) -> bool:
            if cache_ttl_days is None:
                return True
            try:
                ttl = float(cache_ttl_days)
            except Exception:
                return True
            if ttl <= 0:
                return False
            try:
                age_s = max(0.0, time.time() - float(path.stat().st_mtime))
            except Exception:
                return False
            return age_s <= (ttl * 86400.0)

        def _read_ids_from_cache(path: Path) -> list[str]:
            ids: list[str] = []
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for raw in f:
                        line = (raw or "").strip()
                        if not line or line.startswith("#"):
                            continue
                        # Prefer parsing watch URL lines
                        m = watch_id_re.search(line)
                        if m:
                            ids.append(m.group(1))
                            continue
                        # Fallback: allow raw IDs per line
                        if yt_id_re.match(line):
                            ids.append(line)
            except Exception:
                return []
            return ids

        # Try cache first (if provided)
        if cache_file is not None:
            try:
                if cache_file.exists() and cache_file.stat().st_size > 0 and _is_cache_fresh(cache_file):
                    cached_ids = _read_ids_from_cache(cache_file)
                    if cached_ids:
                        duration = time.time() - start_time
                        self._log_msg(
                            f"Using cached video list: {cache_file} ({len(cached_ids)} ids, {duration:.1f}s)",
                            "info",
                            message_callback,
                        )
                        return cached_ids
                    else:
                        self._log_msg(
                            f"Cached list exists but could not be parsed, will refresh: {cache_file}",
                            "warning",
                            message_callback,
                        )
                elif cache_file.exists() and cache_file.stat().st_size > 0 and not _is_cache_fresh(cache_file):
                    self._log_msg(
                        f"Cached list expired (mtime); refreshing: {cache_file}",
                        "info",
                        message_callback,
                    )
            except Exception:
                # Best-effort: fall through to network fetch
                pass

        self._log_msg(f"Fetching channel video list for deduplication: {url}", "info", message_callback)
        try:
            cmd = [
                "yt-dlp",
                "--flat-playlist",
                "--print",
                "id",
            ]
            
            if not debug:
                cmd.extend(["--quiet", "--no-warnings"])
            
            cmd.append(url)
            
            self._log_msg(f"[DEBUG] Running: {' '.join(cmd)}", "debug", message_callback)
            
            ids: list[str] = []
            if debug:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in process.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    if yt_id_re.match(line):
                        ids.append(line)
                    else:
                        self._log_msg(f"[yt-dlp] {line}", "info", message_callback)
                process.wait()
                duration = time.time() - start_time
                if process.returncode == 0:
                    self._log_msg(f"Successfully fetched {len(ids)} video IDs in {duration:.1f}s", "info", message_callback)
                    if cache_file is not None and ids:
                        try:
                            cache_file.parent.mkdir(parents=True, exist_ok=True)
                            tmp_path = cache_file.with_suffix(cache_file.suffix + ".tmp")
                            with open(tmp_path, "w", encoding="utf-8") as f:
                                for vid in ids:
                                    f.write(f"https://www.youtube.com/watch?v={vid}\n")
                            os.replace(tmp_path, cache_file)
                            self._log_msg(
                                f"Saved {len(ids)} ids to list cache: {cache_file}",
                                "info",
                                message_callback,
                            )
                        except Exception as e:
                            self._log_msg(
                                f"Failed to write list cache {cache_file}: {e}",
                                "warning",
                                message_callback,
                            )
                    return ids
                else:
                    self._log_msg(f"Failed to fetch video IDs (code {process.returncode}) after {duration:.1f}s", "error", message_callback)
                    return []
            else:
                # Large channels may take time to enumerate; use a longer timeout here
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                
                duration = time.time() - start_time
                
                if result.returncode == 0:
                    ids = [vid.strip() for vid in result.stdout.split("\n") if vid.strip()]
                    self._log_msg(f"Successfully fetched {len(ids)} video IDs in {duration:.1f}s", "info", message_callback)
                    if cache_file is not None and ids:
                        try:
                            cache_file.parent.mkdir(parents=True, exist_ok=True)
                            tmp_path = cache_file.with_suffix(cache_file.suffix + ".tmp")
                            with open(tmp_path, "w", encoding="utf-8") as f:
                                for vid in ids:
                                    f.write(f"https://www.youtube.com/watch?v={vid}\n")
                            os.replace(tmp_path, cache_file)
                            self._log_msg(
                                f"Saved {len(ids)} ids to list cache: {cache_file}",
                                "info",
                                message_callback,
                            )
                        except Exception as e:
                            self._log_msg(
                                f"Failed to write list cache {cache_file}: {e}",
                                "warning",
                                message_callback,
                            )
                    return ids
                
                # Debug info for failure
                self._log_msg(f"Failed to fetch video IDs (code {result.returncode}) after {duration:.1f}s", "error", message_callback)
                error_preview = result.stderr[-500:] if result.stderr else "No stderr output"
                self._log_msg(f"yt-dlp error output: {error_preview}", "error", message_callback)
                return []

        except subprocess.TimeoutExpired:
            self._log_msg(f"Timeout while fetching video IDs for {url}", "error", message_callback)
            return []
        except Exception as e:
            self._log_msg(f"Error fetching channel video IDs: {e}", "error", message_callback)
            return []

    def _classify_error(self, error_line: str) -> str:
        """
        Classify an error line into categories.
        
        Returns:
            'rate_limit' - Anti-bot / rate limiting
            'members_only' - Requires membership
            'unavailable' - Private/removed video
            'other' - Unknown error
        """
        return classify_channel_error_line(error_line)

    def _determine_final_status(self, result: DownloadResult, channel_name: str, message_callback: Optional[Callable] = None) -> tuple[str, Optional[str]]:
        """Determine final download status based on combined results."""
        total_processed = result.success_count + result.already_downloaded_count
        
        if result.has_rate_limit:
            error_msg = (
                f"Rate limited after {total_processed} videos. "
                f"{result.rate_limited_count} videos blocked. "
                f"Retry later."
            )
            if result.rate_limit_errors:
                error_msg += f" Last error: {result.rate_limit_errors[-1][:200]}"
            self._log_msg(f"[{channel_name}] {error_msg}", "warning", message_callback)
            return ("youtube_rate_limited", error_msg[:500])
        
        if result.return_code != 0 and total_processed == 0:
            error_msg = f"yt-dlp exited with code {result.return_code}, no videos downloaded"
            self._log_msg(f"[{channel_name}] {error_msg}", "error", message_callback)
            return ("youtube_failed", error_msg)
        
        if result.critical_errors > 0 and total_processed == 0:
            # Nothing downloaded and had unknown errors
            error_msg = f"Failed to download any videos. {result.critical_errors} unknown errors."
            if result.other_errors:
                error_msg += f" Last error: {result.other_errors[-1][:200]}"
            self._log_msg(f"[{channel_name}] {error_msg}", "error", message_callback)
            return ("youtube_failed", error_msg[:500])
        
        if result.critical_errors > 0 and total_processed > 0:
            error_msg = (
                f"Partially completed: {total_processed} videos OK, "
                f"{result.critical_errors} unknown errors"
            )
            if result.expected_errors > 0:
                error_msg += f", {result.expected_errors} expected skips (members/unavailable)"
            self._log_msg(f"[{channel_name}] {error_msg}", "warning", message_callback)
            return ("youtube_partial", error_msg[:500])
        
        info_msg = f"Completed: {result.success_count} downloaded, {result.already_downloaded_count} already had"
        if result.expected_errors > 0:
            info_msg += f", {result.expected_errors} skipped (members-only/unavailable)"
        self._log_msg(f"[{channel_name}] {info_msg}", "info", message_callback)
        
        return ("youtube_completed", None)

    def _run_ytdlp_with_progress(
        self,
        cmd: list,
        channel_name: str,
        progress_callback=None,
        message_callback=None,
        debug: bool = False,
        item_total_hint: Optional[int] = None,
    ) -> DownloadResult:
        """
        Run yt-dlp command and parse output.
        Args:
            cmd: Command list
            channel_name: Name for logging
            progress_callback: Function(percent, speed, eta, current_file, item_index, item_total) -> None
            message_callback: Function(str) -> None for non-progress output
            debug: If True, output all unmatched yt-dlp lines for debugging
            
        Returns:
            DownloadResult with counts and error details
        """
        result = DownloadResult()
        
        # Force line buffering and text mode
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Regex patterns
        progress_pattern = re.compile(
            r"\[download\]\s+(\d+\.\d+)%.*?at\s+(\S+)\s+ETA\s+(\S+)"
        )
        dest_pattern = re.compile(r"\[download\]\s+Destination:\s+(.+)")
        already_pattern = re.compile(r"\[download\]\s+(.+)\s+has already been downloaded")
        # Pattern for "Downloading video 1 of 3" or "Downloading item 1 of 3"
        # Relaxed regex to match even if [download] prefix is missing or different
        item_pattern = re.compile(r"Downloading\s+(?:video|item)\s+(\d+)\s+of\s+(\d+)", re.IGNORECASE)
        
        # Pattern for successful extraction (video processed)
        extract_pattern = re.compile(r"\[ExtractAudio\]\s+Destination:")
        
        # Pattern for debug/info messages
        info_pattern = re.compile(r"^\[(debug|info|youtube|youtube:tab|youtube\+invidious)\]\s+(.+)")

        last_print_time = 0
        current_file = ""
        current_video_id = None
        current_item_index = None
        current_item_total = None
        
        # Track if current video succeeded
        current_video_started = False

        # State for handling "Downloading as multiple playlists" (tabs)
        multi_playlist_mode = False
        master_playlist_name = None
        current_playlist_name = None
        
        # Regex for playlist detection
        multi_playlist_pattern = re.compile(r"Downloading as multiple playlists", re.IGNORECASE)
        playlist_pattern = re.compile(r"\[download\]\s+Downloading playlist:\s+(.+)", re.IGNORECASE)
        
        # Pattern to extract video ID from various yt-dlp messages
        video_id_pattern = re.compile(r"\[youtube(?:\+invidious)?\]\s+([a-zA-Z0-9_-]{11}):")
        # Pattern for archive skip (common when using --download-archive / --batch-file)
        archive_skip_pattern = re.compile(
            r"\[download\]\s+([0-9A-Za-z_-]{11}):\s+has already been recorded in (?:the )?archive",
            re.IGNORECASE,
        )
        # Pattern for "already downloaded" that includes an ID prefix
        already_id_pattern = re.compile(
            r"\[download\]\s+([0-9A-Za-z_-]{11})\s+has already been downloaded",
            re.IGNORECASE,
        )

        # Keep track of recent lines for debugging context
        from collections import deque
        recent_lines = deque(maxlen=20)

        # If caller knows total items (e.g. batch-file), initialize item counters.
        if item_total_hint is not None:
            try:
                total_i = int(item_total_hint)
                if total_i > 0:
                    current_item_total = total_i
                    current_item_index = 0
            except Exception:
                pass

        seen_item_ids: set[str] = set()
        last_item_emit_ts = 0.0

        def _advance_item(vid: Optional[str], *, force_emit: bool = False) -> None:
            nonlocal current_item_index, last_item_emit_ts
            if not vid:
                return
            if vid in seen_item_ids:
                return
            seen_item_ids.add(vid)

            if current_item_index is None:
                current_item_index = 0
            try:
                current_item_index = int(current_item_index) + 1
            except Exception:
                current_item_index = 1

            if current_item_total is not None:
                try:
                    if current_item_index > int(current_item_total):
                        current_item_index = int(current_item_total)
                except Exception:
                    pass

            if progress_callback:
                now = time.time()
                # Throttle UI updates for large channels
                if force_emit or (now - last_item_emit_ts >= 0.25):
                    progress_callback(0, "...", "...", current_file, current_item_index, current_item_total)
                    last_item_emit_ts = now

        try:
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                
                recent_lines.append(line)

                # Batch-file / archive skip case: yt-dlp may not emit [youtube] lines
                m_arch = archive_skip_pattern.search(line)
                if m_arch:
                    vid = m_arch.group(1)
                    result.already_downloaded_count += 1
                    _advance_item(vid, force_emit=True)
                    continue

                m_already_id = already_id_pattern.search(line)
                if m_already_id:
                    vid = m_already_id.group(1)
                    result.already_downloaded_count += 1
                    _advance_item(vid, force_emit=True)
                    continue
                
                # Try to extract current video ID
                vid_match = video_id_pattern.search(line)
                if vid_match:
                    new_vid = vid_match.group(1)
                    if new_vid != current_video_id:
                        # New video started - if previous one had started without error, count as success
                        if current_video_started:
                            result.success_count += 1
                        current_video_id = new_vid
                        current_video_started = True
                        _advance_item(new_vid)

                # Check for "multiple playlists" mode (tabs)
                if multi_playlist_pattern.search(line):
                    multi_playlist_mode = True
                    continue

                # Check for playlist name
                match_playlist = playlist_pattern.search(line)
                if match_playlist:
                    new_playlist_name = match_playlist.group(1).strip()
                    current_playlist_name = new_playlist_name
                    if multi_playlist_mode and master_playlist_name is None:
                        master_playlist_name = new_playlist_name
                    continue

                # Check for item progress (video x of y)
                match_item = item_pattern.search(line)
                if match_item:
                    if multi_playlist_mode and current_playlist_name == master_playlist_name:
                        continue

                    current_item_index = int(match_item.group(1))
                    current_item_total = int(match_item.group(2))
                    if progress_callback:
                        progress_callback(0, "...", "...", current_file, current_item_index, current_item_total)
                    continue

                # Check for progress
                match_progress = progress_pattern.search(line)
                if match_progress:
                    percent = float(match_progress.group(1))
                    speed = match_progress.group(2)
                    eta = match_progress.group(3)
                    
                    if progress_callback:
                        progress_callback(percent, speed, eta, current_file, current_item_index, current_item_total)
                    
                    # Fallback logging if no callback (limited rate)
                    elif (time.time() - last_print_time > 5) or (percent > 99):
                        item_str = f"[{current_item_index}/{current_item_total}] " if current_item_index else ""
                        print(f"[{channel_name}] {item_str}{percent}% ({speed}) ETA {eta}")
                        last_print_time = time.time()
                    continue

                # Check for file destination (new download starting)
                match_dest = dest_pattern.search(line)
                if match_dest:
                    current_file = match_dest.group(1).strip()
                    msg = f"[{channel_name}] Downloading: {current_file}"
                    if message_callback:
                        message_callback(msg)
                    elif not progress_callback:
                         print(msg)
                    
                    if progress_callback:
                        # Update callback with 0% and new filename immediately
                        progress_callback(0, "...", "...", current_file, current_item_index, current_item_total)
                    continue

                # Check for already downloaded
                match_already = already_pattern.search(line)
                if match_already:
                    result.already_downloaded_count += 1
                    current_video_started = False  # Don't count again
                    continue

                # Check for errors - classify them
                # Note: yt-dlp may split error messages across multiple lines
                # So we also check for bot detection keywords even without ERROR prefix
                is_error_line = "ERROR" in line
                
                # Quick check for bot detection keywords (may appear on continuation lines)
                lower_line = line.lower()
                is_bot_detection = (
                    "not a bot" in lower_line
                    or "sign in to confirm" in lower_line
                    or "confirm you're not" in lower_line
                    or "confirm you’re not" in lower_line
                )
                
                if is_error_line or is_bot_detection:
                    if is_bot_detection:
                        error_type = 'rate_limit'
                    else:
                        error_type = self._classify_error(line)
                    
                    # Extract video ID if present
                    # Example: ERROR: [youtube+invidious] kwYo-splAVw: Sign in to confirm...
                    # Regex to find video ID (11 chars, base64-like) followed by colon
                    video_url_part = ""
                    # Match [extractor] video_id: 
                    # Try to find standard YouTube ID pattern
                    id_match = re.search(r" ([a-zA-Z0-9_-]{11}):", line)
                    if id_match:
                        video_id = id_match.group(1)
                        video_url_part = f" (https://www.youtube.com/watch?v={video_id})"
                    
                    if error_type == 'rate_limit':
                        result.rate_limited_count += 1
                        result.rate_limit_errors.append(line)
                        
                        # Construct context message
                        context_msg = "\n".join(f"    {l}" for l in recent_lines)
                        self._log_msg(
                            f"[{channel_name}] Rate limit/bot detected! Aborting current attempt.\n"
                            f"Trigger line: {line}\n"
                            f"Recent log context:\n{context_msg}",
                            "warning",
                            message_callback
                        )
                        
                        if message_callback:
                            message_callback(f"[yellow][{channel_name}] ⚠️ Bot detected, aborting current attempt: {line}{video_url_part}[/yellow]")
                        
                        # Immediately terminate the process and switch to next instance
                        current_video_started = False
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        result.return_code = -1  # Mark as interrupted
                        return result
                        
                    elif error_type == 'members_only':
                        result.members_only_count += 1
                        result.members_only_errors.append(line)
                        self._log_msg(f"[{channel_name}] Members-only video: {line}", "info", message_callback)
                        if message_callback:
                            message_callback(f"[blue][{channel_name}] 🔒 Members only: {line}{video_url_part}[/blue]")
                    elif error_type == 'unavailable':
                        # Unavailable videos are expected (private/removed/geo-blocked)
                        result.unavailable_count += 1
                        result.unavailable_errors.append(line)
                        self._log_msg(f"[{channel_name}] Video unavailable: {line}", "info", message_callback)
                        if message_callback:
                            message_callback(f"[dim][{channel_name}] ⛔ Unavailable: {line}{video_url_part}[/dim]")
                    else:
                        result.other_error_count += 1
                        result.other_errors.append(line)
                        self._log_msg(f"[{channel_name}] {line}", "error", message_callback)
                        if message_callback:
                            message_callback(f"[red][{channel_name}] {line}{video_url_part}[/red]")
                    
                    current_video_started = False  # This video failed
                    if not message_callback and not progress_callback:
                        print(f"[{channel_name}] {line}")
                    continue

                # Check for debug/info messages
                match_info = info_pattern.search(line)
                if match_info:
                    msg = f"[{channel_name}] {line}"
                    if message_callback:
                        # Use dim style for debug info to not clutter too much
                        message_callback(f"[dim]{msg}[/dim]")
                    elif not progress_callback:
                        print(msg)
                    continue

                # Fallback: output unmatched lines only in debug mode
                if debug:
                    msg = f"[{channel_name}] {line}"
                    if message_callback:
                        message_callback(f"[dim]{msg}[/dim]")
                    elif not progress_callback:
                        print(msg)

        except Exception as e:
            self._log_msg(f"Error reading output for {channel_name}: {e}", "error", message_callback)
        
        if current_video_started:
            result.success_count += 1
            current_video_started = False

        result.return_code = process.wait()
        
        # Log summary
        self._log_msg(
            f"[{channel_name}] Download result: "
            f"success={result.success_count}, "
            f"already_downloaded={result.already_downloaded_count}, "
            f"rate_limited={result.rate_limited_count}, "
            f"members_only={result.members_only_count}, "
            f"unavailable={result.unavailable_count}, "
            f"other_errors={result.other_error_count}, "
            f"return_code={result.return_code}",
            "info",
            message_callback
        )
        
        return result

    def download_channel(
        self,
        url: str,
        output_dir: Optional[Path] = None,
        channel_name: Optional[str] = None,
        language: str = "ja",
        progress_callback=None,
        message_callback=None,
        debug: bool = False,
        cookies_file_path: Optional[Path] = None,
        no_invidious: bool = False,
        accounts: Optional[list["YouTubeAccount"]] = None,
        account_pool: Optional["YouTubeAccountPool"] = None,
        worker_id: Optional[int] = None,
    ) -> DownloadResult:
        """
        Download entire YouTube channel directly using yt-dlp.

        Args:
            url: Channel URL
            output_dir: Root output directory
            channel_name: Optional display name for logs/output folder
            language: Subtitle language (default: 'ja' for Japanese)
            progress_callback: Optional callback for progress updates
            message_callback: Optional callback for non-progress messages
            debug: If True, output all yt-dlp messages for debugging
            cookies_file_path: Optional path to Netscape format cookies file for auth
            no_invidious: If True, disable Invidious instances and use direct connection only

        Returns:
            DownloadResult with counts and final_status fields
        """
        if not url:
            result = DownloadResult(return_code=1)
            result.final_status = "youtube_failed"
            result.error_message = "Missing channel URL"
            return result

        output_root = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else self.download_dir.expanduser().resolve()
        )
        output_root.mkdir(parents=True, exist_ok=True)

        channel_key = channel_name or self._derive_channel_key(url)
        channel_key = self._sanitize_filename(channel_key)
        channel_dir = output_root / channel_key
        channel_dir.mkdir(parents=True, exist_ok=True)

        archive_file = channel_dir / "download.archive.txt"
        list_file = channel_dir / "video_ids.txt"

        try:
            # --- Global Deduplication Logic ---
            try:
                self._log_msg("Checking global archive for duplicates...", "info", message_callback)

                cache_ttl_days = getattr(config, "YOUTUBE_CHANNEL_LIST_CACHE_TTL_DAYS", 7.0)
                channel_videos = self._get_channel_video_ids(
                    url,
                    cache_file=list_file,
                    cache_ttl_days=cache_ttl_days,
                    debug=debug,
                    message_callback=message_callback,
                )

                if channel_videos:
                    global_ids = self._load_global_archive()

                    local_ids = set()
                    if archive_file.exists():
                        with open(archive_file, "r", encoding="utf-8", errors="ignore") as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 2 and parts[0] == "youtube":
                                    local_ids.add(parts[1])

                    new_skips = []
                    for vid in channel_videos:
                        if vid in global_ids and vid not in local_ids:
                            new_skips.append(vid)

                    if new_skips:
                        self._log_msg(
                            f"Skipping {len(new_skips)} videos found in global archive",
                            "info",
                            message_callback,
                        )
                        with open(archive_file, "a", encoding="utf-8") as f:
                            for vid in new_skips:
                                f.write(f"youtube {vid}\n")
            except Exception as e:
                self._log_msg(f"Error in global deduplication: {e}", "error", message_callback)
                # Continue with download even if deduplication fails
            # ----------------------------------

            sub_langs = language if language else "ja,ja-orig,en"

            # --- Cookies / account pool ---
            # Priority:
            # - explicit accounts list (shared account_pool controls cooldown)
            # - legacy cookies_file_path
            accounts_list: list[YouTubeAccount] = list(accounts or [])
            pool = account_pool if (account_pool is not None and len(accounts_list) > 1) else None

            account_idx = -1
            account_name = ""
            cookies_file: Optional[Path] = None

            if accounts_list:
                # Distribute initial accounts across workers when possible
                base_idx = 0
                if worker_id is not None:
                    try:
                        wid = int(worker_id)
                    except Exception:
                        wid = 0
                    # worker_id from main is 1-based
                    base_idx = (wid - 1) if wid > 0 else wid
                account_idx = base_idx % len(accounts_list)

                # If selected account is cooling down, switch to an available one
                if pool is not None:
                    now = time.time()
                    try:
                        if accounts_list[account_idx].cooldown_until > now:
                            next_idx, wait_s = pool.pick_next(account_idx, exclude_current=True)
                            if wait_s and wait_s > 0:
                                time.sleep(wait_s)
                            if 0 <= next_idx < len(accounts_list):
                                account_idx = next_idx
                    except Exception:
                        pass

                try:
                    account_name = accounts_list[account_idx].name
                    cookies_file = accounts_list[account_idx].cookies_file
                except Exception:
                    account_name = ""
                    cookies_file = None
            else:
                cookies_file = cookies_file_path

            if cookies_file and not cookies_file.exists():
                self._log_msg(f"Cookies file not found: {cookies_file}", "warning", message_callback)
                cookies_file = None

            def build_ytdlp_cmd(
                target_url: str,
                invidious_instance: Optional[str] = None,
                batch_file: Optional[Path] = None,
            ) -> list[str]:
                """Build yt-dlp command for a specific target URL, or a batch file."""
                # Build extractor args
                extractor_args = ["youtubetab:approximate_date=true"]

                # Add YouTube player client if configured
                youtube_extractor_args = []
                if getattr(config, "YOUTUBE_PLAYER_CLIENT", None):
                    youtube_extractor_args.append(f"player_client={config.YOUTUBE_PLAYER_CLIENT}")

                # Add Invidious instance
                if invidious_instance:
                    youtube_extractor_args.append(f"invidious_instance=https://{invidious_instance}")

                # PO token / POT provider (optional)
                po_token = getattr(config, "YOUTUBE_PO_TOKEN", None)
                pot_provider = getattr(config, "YOUTUBE_POT_PROVIDER", None)
                if po_token:
                    youtube_extractor_args.append(f"po_token={po_token}")
                if pot_provider:
                    youtube_extractor_args.append(f"pot_provider={pot_provider}")

                if youtube_extractor_args:
                    extractor_args.append("youtube:" + ";".join(youtube_extractor_args))

                extra_args = getattr(config, "YOUTUBE_EXTRACTOR_ARGS", None)
                if extra_args:
                    if isinstance(extra_args, (list, tuple)):
                        extractor_args.extend([str(x) for x in extra_args if str(x).strip()])
                    elif isinstance(extra_args, str):
                        extractor_args.append(extra_args)

                # Get rate limiting settings from config
                sleep_requests = getattr(config, "YOUTUBE_SLEEP_REQUESTS", 2)
                sleep_interval = getattr(config, "YOUTUBE_SLEEP_INTERVAL", 5)
                max_sleep_interval = getattr(config, "YOUTUBE_MAX_SLEEP_INTERVAL", 8)

                # Start building base command
                cmd: list[str] = ["yt-dlp"]

                # Add JS runtime (Deno) *before* compat options so we don't break
                # the "--compat-options no-youtube-unavailable-videos" pair.
                if self._check_deno():
                    cmd.extend(["--js-runtimes", "deno"])

                # Keep compat options grouped; JS runtime may appear before this.
                cmd.extend(["--compat-options", "no-youtube-unavailable-videos"])

                if batch_file is not None:
                    cmd.extend(["--batch-file", str(batch_file)])
                else:
                    cmd.append(target_url)

                cmd.extend(
                    [
                        "--sleep-requests",
                        str(sleep_requests),
                        "--sleep-interval",
                        str(sleep_interval),
                        "--max-sleep-interval",
                        str(max_sleep_interval),
                        "-x",  # Extract audio
                        "--audio-format",
                        "best",  # Best audio quality
                        "--download-archive",
                        str(archive_file),  # Track downloaded videos
                        "-f",
                        "bestaudio/best[height<=144]/best",  # Format selection
                        "--write-subs",  # Download subtitles
                        "--write-auto-subs",  # Download auto-generated subtitles
                        "--sub-langs",
                        sub_langs,  # Subtitle languages
                        "-o",
                        str(
                            channel_dir / "%(title).50s [%(id)s].%(ext)s"
                        ),  # Output template (50 char limit)
                        "--convert-subs",
                        "srt",  # Convert subtitles to SRT
                        "--sub-format",
                        "srt/best",  # Subtitle format
                        "--newline",  # Force newline for progress
                        "--match-filter",
                        "availability != 'premium' & availability != 'subscriber_only' & availability != 'needs_auth'",
                        "--ignore-errors",  # Skip errors (like members-only if filter fails)
                    ]
                )

                # Add extractor args
                for arg in extractor_args:
                    cmd.extend(["--extractor-args", arg])

                # Add cookies file if available (for browser-based auth)
                if cookies_file:
                    cmd.extend(["--cookies", str(cookies_file)])

                return cmd

            def _select_invidious_instance() -> Optional[str]:
                """
                Select a single Invidious instance.
                Priority:
                - config.INVIDIOUS_INSTANCE
                - None (direct)
                """
                if no_invidious:
                    return None

                inst = getattr(config, "INVIDIOUS_INSTANCE", None)
                if inst:
                    return inst

                return None

            invidious_instance = _select_invidious_instance()
            instance_name = invidious_instance or "direct"

            self._log_msg(f"Downloading YouTube channel: {channel_key}", "info", message_callback)
            self._log_msg(f"URL: {url}", "info", message_callback)
            self._log_msg(f"Output directory: {channel_dir}", "info", message_callback)
            self._log_msg(f"Archive file: {archive_file}", "info", message_callback)
            self._log_msg(f"Invidious instance: {instance_name} (no fallback/rotation)", "info", message_callback)

            # --- Main download (with optional account switching) ---
            switches_used = 0
            max_switches_cfg = int(getattr(config, "YOUTUBE_ACCOUNT_SWITCH_MAX", 0) or 0)
            max_switches = (
                max_switches_cfg
                if max_switches_cfg and max_switches_cfg > 0
                else max(0, len(accounts_list) - 1)
            )

            # Accumulate stats across retries so we don't lose counts when switching accounts.
            combined_result = DownloadResult()
            last_attempt_result: Optional[DownloadResult] = None
            use_batch_file = False
            try:
                use_batch_file = list_file.exists() and list_file.stat().st_size > 0
            except Exception:
                use_batch_file = False
            item_total_hint: Optional[int] = None
            if use_batch_file:
                try:
                    with open(list_file, "r", encoding="utf-8", errors="ignore") as f:
                        item_total_hint = sum(
                            1
                            for line in f
                            if (line or "").strip() and not (line or "").lstrip().startswith("#")
                        )
                    if item_total_hint is not None and item_total_hint <= 0:
                        item_total_hint = None
                except Exception:
                    item_total_hint = None

            if use_batch_file:
                self._log_msg(
                    f"[{channel_key}] Using list cache for download: {list_file}",
                    "info",
                    message_callback,
                )
            else:
                self._log_msg(
                    f"[{channel_key}] List cache not found/empty; downloading from channel URL (may be slow for large channels)",
                    "warning",
                    message_callback,
                )

            while True:
                cmd = build_ytdlp_cmd(
                    url,
                    invidious_instance,
                    batch_file=(list_file if use_batch_file else None),
                )
                if account_name:
                    self._log_msg(
                        f"[{channel_key}] Using account: {account_name}",
                        "info",
                        message_callback,
                    )

                attempt_result = self._run_ytdlp_with_progress(
                    cmd,
                    channel_key,
                    progress_callback=progress_callback,
                    message_callback=message_callback,
                    debug=debug,
                    item_total_hint=item_total_hint,
                )
                last_attempt_result = attempt_result
                combined_result.merge(attempt_result)

                # Decide retry based on this attempt's rate limit state.
                if not attempt_result.has_rate_limit:
                    break

                # Rate limited: switch account (no invidious rotation)
                if pool is None or account_idx < 0 or switches_used >= max_switches:
                    break

                pool.mark_rate_limited(account_idx)
                next_idx, wait_s = pool.pick_next(account_idx, exclude_current=True)
                if wait_s and wait_s > 0:
                    time.sleep(wait_s)

                if next_idx < 0 or next_idx >= len(accounts_list):
                    break

                account_idx = next_idx
                try:
                    account_name = accounts_list[account_idx].name
                    cookies_file = accounts_list[account_idx].cookies_file
                except Exception:
                    account_name = ""
                    cookies_file = None

                if cookies_file and not cookies_file.exists():
                    self._log_msg(
                        f"Cookies file not found (account={account_name}): {cookies_file}",
                        "warning",
                        message_callback,
                    )
                    cookies_file = None

                switches_used += 1
                self._log_msg(
                    f"[{channel_key}] Rate limited; switching account to {account_name or 'no-cookies'}",
                    "warning",
                    message_callback,
                )

            # If we eventually succeeded after one or more retries, clear the transient
            # rate-limit marker so final status reflects recovery, while keeping the
            # accumulated success/skip/error counts.
            if last_attempt_result is not None:
                if not last_attempt_result.has_rate_limit:
                    combined_result.rate_limited_count = 0
                    combined_result.rate_limit_errors.clear()
                # Final return code should reflect the last attempt, not interrupted retries.
                combined_result.return_code = last_attempt_result.return_code

            # Attempt to capture content from the specific 'Podcasts' tab
            base_url = url.rstrip("/")
            # Simple heuristic to identify channel URLs
            is_channel = any(
                x in base_url for x in ["/@", "/channel/", "/c/", "/user/"]
            )

            if (
                is_channel
                and not base_url.endswith("/podcasts")
                and not combined_result.has_rate_limit
            ):
                podcast_tab_url = f"{base_url}/podcasts"
                self._log_msg(
                    f"Attempting to download Podcast tab: {podcast_tab_url}",
                    "info",
                    message_callback,
                )

                # Use the same single invidious instance (no fallback/rotation)
                cmd_podcast = build_ytdlp_cmd(podcast_tab_url, invidious_instance)

                self._log_msg(
                    f"Starting download for Podcast tab: {channel_key}",
                    "info",
                    message_callback,
                )
                # We run this but don't fail the whole process if it fails (tab might not exist)
                podcast_result = self._run_ytdlp_with_progress(
                    cmd_podcast,
                    channel_key,
                    progress_callback=progress_callback,
                    message_callback=message_callback,
                    debug=debug,
                )
                podcast_tab_total_errors = podcast_result.total_errors

                # The "Podcasts" tab is best-effort (many channels don't have it).
                # We should not let its failures downgrade the whole channel status
                # (e.g., turning an otherwise completed channel into youtube_partial).
                podcast_result.return_code = 0
                podcast_result.rate_limited_count = 0
                podcast_result.rate_limit_errors.clear()
                podcast_result.other_error_count = 0
                podcast_result.other_errors.clear()
                combined_result.merge(podcast_result)

                if podcast_tab_total_errors > 0:
                    self._log_msg(
                        "Podcast tab had some errors. Channel might not have a podcasts tab.",
                        "info",
                        message_callback,
                    )
                else:
                    self._log_msg(
                        f"Successfully processed Podcast tab for: {channel_key}",
                        "info",
                        message_callback,
                    )

            # Determine final status based on combined results
            final_status, error_msg = self._determine_final_status(combined_result, channel_key, message_callback)

            self._log_msg(
                f"Channel {channel_key} final status: {final_status} "
                f"(success={combined_result.success_count}, "
                f"already={combined_result.already_downloaded_count}, "
                f"rate_limited={combined_result.rate_limited_count}, "
                f"members_only={combined_result.members_only_count}, "
                f"unavailable={combined_result.unavailable_count}, "
                f"other_errors={combined_result.other_error_count})",
                "info",
                message_callback,
            )

            combined_result.final_status = final_status
            combined_result.error_message = error_msg
            return combined_result

        except subprocess.TimeoutExpired:
            self._log_msg("Download timeout", "error", message_callback)
            result = DownloadResult(return_code=1)
            result.final_status = "youtube_failed"
            result.error_message = "YouTube download timeout"
            return result
        except Exception as e:
            self._log_msg(f"Error downloading channel {url}: {e}", "error", message_callback)
            result = DownloadResult(return_code=1)
            result.final_status = "youtube_failed"
            result.error_message = str(e)[:500]
            return result

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for filesystem

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        # Remove null bytes
        filename = filename.replace("\0", "")

        # Replace control characters with spaces
        for char in ['\n', '\r', '\t']:
            filename = filename.replace(char, " ")

        # Remove or replace invalid characters
        # Keep using underscore for these to maintain backward compatibility
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, "_")

        # Collapse multiple spaces
        while "  " in filename:
            filename = filename.replace("  ", " ")

        if len(filename) > 200:
            filename = filename[:200]

        return filename.strip()


if __name__ == "__main__":
    # Test the downloader
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    downloader = YouTubeDownloader()
    print("YouTube Downloader initialized")
    print(f"Download directory: {downloader.download_dir}")
