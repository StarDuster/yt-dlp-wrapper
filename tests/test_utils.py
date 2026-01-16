"""Tests for yt_dlp_wrapper/core/utils.py"""

import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from yt_dlp_wrapper.core.utils import (
    YOUTUBE_ID_RE,
    _backoff_sleep,
    _derive_channel_key,
    _format_eta,
    _format_time,
    _sanitize_filename,
    append_archive_id,
    append_failed_id,
    load_archive_ids,
    load_failed_ids,
    parse_youtube_id,
)


class TestParseYoutubeId(unittest.TestCase):
    def test_valid_11_char_id(self) -> None:
        self.assertEqual(parse_youtube_id("dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(parse_youtube_id("abcdefghijk"), "abcdefghijk")
        self.assertEqual(parse_youtube_id("12345678901"), "12345678901")
        self.assertEqual(parse_youtube_id("A_-z0123456"), "A_-z0123456")

    def test_youtube_watch_url(self) -> None:
        vid = "dQw4w9WgXcQ"
        self.assertEqual(parse_youtube_id(f"https://www.youtube.com/watch?v={vid}"), vid)
        self.assertEqual(parse_youtube_id(f"http://youtube.com/watch?v={vid}"), vid)
        self.assertEqual(parse_youtube_id(f"https://m.youtube.com/watch?v={vid}"), vid)
        self.assertEqual(parse_youtube_id(f"https://www.youtube.com/watch?v={vid}&t=1"), vid)

    def test_youtu_be_url(self) -> None:
        vid = "dQw4w9WgXcQ"
        self.assertEqual(parse_youtube_id(f"https://youtu.be/{vid}"), vid)
        self.assertEqual(parse_youtube_id(f"http://youtu.be/{vid}"), vid)

    def test_youtube_shorts_url(self) -> None:
        vid = "dQw4w9WgXcQ"
        self.assertEqual(parse_youtube_id(f"https://www.youtube.com/shorts/{vid}"), vid)

    def test_youtube_embed_url(self) -> None:
        vid = "dQw4w9WgXcQ"
        self.assertEqual(parse_youtube_id(f"https://www.youtube.com/embed/{vid}"), vid)

    def test_invalid_input(self) -> None:
        self.assertIsNone(parse_youtube_id(""))
        self.assertIsNone(parse_youtube_id("   "))
        self.assertIsNone(parse_youtube_id("short"))
        self.assertIsNone(parse_youtube_id("toolongforavideoid"))
        self.assertIsNone(parse_youtube_id("https://example.com/video"))

    def test_whitespace_handling(self) -> None:
        self.assertEqual(parse_youtube_id("  dQw4w9WgXcQ  "), "dQw4w9WgXcQ")


class TestFormatEta(unittest.TestCase):
    def test_none(self) -> None:
        self.assertEqual(_format_eta(None), "")

    def test_negative(self) -> None:
        self.assertEqual(_format_eta(-1), "")

    def test_zero(self) -> None:
        self.assertEqual(_format_eta(0), "00:00")

    def test_seconds(self) -> None:
        self.assertEqual(_format_eta(59), "00:59")

    def test_minutes(self) -> None:
        self.assertEqual(_format_eta(60), "01:00")
        self.assertEqual(_format_eta(90), "01:30")

    def test_hours(self) -> None:
        self.assertEqual(_format_eta(3661), "1:01:01")

    def test_days(self) -> None:
        # 1 day + 1 hour + 1 minute + 1 second = 90061 seconds
        self.assertEqual(_format_eta(90061), "1d01:01:01")


class TestFormatTime(unittest.TestCase):
    def test_none(self) -> None:
        self.assertEqual(_format_time(None), "")

    def test_negative(self) -> None:
        self.assertEqual(_format_time(-1), "")

    def test_zero(self) -> None:
        self.assertEqual(_format_time(0), "00:00.00")

    def test_with_decimals(self) -> None:
        self.assertEqual(_format_time(1.5), "00:01.50")
        self.assertEqual(_format_time(65.25), "01:05.25")

    def test_hours(self) -> None:
        self.assertEqual(_format_time(3661.5), "1:01:01.50")


class TestBackoffSleep(unittest.TestCase):
    def test_basic_backoff(self) -> None:
        # attempt 1: base * 2^0 = base
        self.assertEqual(_backoff_sleep(1, 1.0, 60.0, 0.0), 1.0)
        # attempt 2: base * 2^1 = 2
        self.assertEqual(_backoff_sleep(2, 1.0, 60.0, 0.0), 2.0)
        # attempt 3: base * 2^2 = 4
        self.assertEqual(_backoff_sleep(3, 1.0, 60.0, 0.0), 4.0)

    def test_cap(self) -> None:
        # attempt 10 would be 512, but cap is 60
        self.assertEqual(_backoff_sleep(10, 1.0, 60.0, 0.0), 60.0)
        # cap at 3
        self.assertEqual(_backoff_sleep(3, 1.0, 3.0, 0.0), 3.0)

    def test_invalid_inputs(self) -> None:
        self.assertEqual(_backoff_sleep(1, 0.0, 60.0, 0.0), 0.0)
        self.assertEqual(_backoff_sleep(1, 1.0, 0.0, 0.0), 0.0)
        # Invalid attempt defaults to 1
        self.assertEqual(_backoff_sleep("bad", 1.0, 60.0, 0.0), 1.0)


class TestArchiveHelpers(unittest.TestCase):
    def test_load_archive_ids_nonexistent(self) -> None:
        ids = load_archive_ids(Path("/nonexistent/path.txt"))
        self.assertEqual(ids, set())

    def test_archive_round_trip(self) -> None:
        with TemporaryDirectory() as td:
            tmp_dir = Path(td)
            archive = tmp_dir / "subdir" / "download.archive.txt"
            lock = threading.Lock()

            append_archive_id(archive, "dQw4w9WgXcQ", lock)
            append_archive_id(archive, "abcdefghijk", lock)

            ids = load_archive_ids(archive)
            self.assertEqual(ids, {"dQw4w9WgXcQ", "abcdefghijk"})

    def test_archive_with_segment_key(self) -> None:
        with TemporaryDirectory() as td:
            tmp_dir = Path(td)
            archive = tmp_dir / "download.archive.txt"
            lock = threading.Lock()

            seg_key = "dQw4w9WgXcQ:1500-2750"
            append_archive_id(archive, seg_key, lock)

            ids = load_archive_ids(archive)
            self.assertIn(seg_key, ids)


class TestFailedHelpers(unittest.TestCase):
    def test_load_failed_ids_nonexistent(self) -> None:
        ids = load_failed_ids(Path("/nonexistent/path.txt"))
        self.assertEqual(ids, set())

    def test_failed_round_trip(self) -> None:
        with TemporaryDirectory() as td:
            tmp_dir = Path(td)
            failed = tmp_dir / "subdir" / "failed.txt"
            lock = threading.Lock()

            append_failed_id(failed, "dQw4w9WgXcQ", "some error", lock)

            ids = load_failed_ids(failed)
            self.assertEqual(ids, {"dQw4w9WgXcQ"})

    def test_failed_with_segment_key(self) -> None:
        with TemporaryDirectory() as td:
            tmp_dir = Path(td)
            failed = tmp_dir / "failed.txt"
            lock = threading.Lock()

            seg_key = "dQw4w9WgXcQ:1500-2750"
            append_failed_id(failed, seg_key, "error", lock)

            ids = load_failed_ids(failed)
            self.assertIn(seg_key, ids)


class TestDeriveChannelKey(unittest.TestCase):
    def test_handle_url(self) -> None:
        self.assertEqual(_derive_channel_key("https://www.youtube.com/@SomeChannel"), "@SomeChannel")

    def test_channel_path(self) -> None:
        self.assertEqual(_derive_channel_key("https://www.youtube.com/channel/UC12345"), "channel-UC12345")

    def test_user_path(self) -> None:
        self.assertEqual(_derive_channel_key("https://www.youtube.com/user/SomeUser"), "user-SomeUser")

    def test_empty(self) -> None:
        self.assertEqual(_derive_channel_key(""), "channel")

    def test_fallback_hash(self) -> None:
        # For URLs that don't match known patterns, should return a hash-based key
        result = _derive_channel_key("https://example.com/something")
        self.assertTrue(result.startswith("something") or result.startswith("channel-"))


class TestSanitizeFilename(unittest.TestCase):
    def test_basic(self) -> None:
        self.assertEqual(_sanitize_filename("normal_file.txt"), "normal_file.txt")

    def test_invalid_chars(self) -> None:
        self.assertEqual(_sanitize_filename('file<>:"/\\|?*.txt'), "file_________.txt")

    def test_control_chars(self) -> None:
        # Control chars are replaced with spaces, then trailing spaces are stripped
        result = _sanitize_filename("file\nwith\rnewlines\t")
        self.assertEqual(result, "file with newlines")

    def test_null_char(self) -> None:
        self.assertEqual(_sanitize_filename("file\0name"), "filename")

    def test_long_filename(self) -> None:
        long_name = "a" * 300
        result = _sanitize_filename(long_name)
        self.assertEqual(len(result), 200)

    def test_collapse_spaces(self) -> None:
        self.assertEqual(_sanitize_filename("file   with   spaces"), "file with spaces")


if __name__ == "__main__":
    unittest.main()
