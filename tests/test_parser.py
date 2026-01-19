"""Tests for yt_dlp_wrapper/core/parser.py"""

import unittest
from unittest.mock import MagicMock

from yt_dlp_wrapper.core.diagnostics import DownloadResult
from yt_dlp_wrapper.core.parser import YtDlpOutputParser


class TestYtDlpOutputParser(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = YtDlpOutputParser(channel_name="TestChannel")
        self.result = DownloadResult()

    def test_progress_line(self) -> None:
        line = "[download]  50.0% of 100.00MiB at 1.00MiB/s ETA 00:50"
        self.parser.handle_line(self.result, line)

    def test_destination_line(self) -> None:
        line = "[download] Destination: /path/to/file.mp4"
        self.parser.handle_line(self.result, line)
        self.assertEqual(self.parser.current_file, "/path/to/file.mp4")

    def test_already_downloaded_line(self) -> None:
        line = "[download] /path/to/file.mp4 has already been downloaded"
        self.parser.handle_line(self.result, line)
        self.assertEqual(self.result.already_downloaded_count, 1)

    def test_archive_skip_line(self) -> None:
        line = "[download] dQw4w9WgXcQ: has already been recorded in the archive"
        self.parser.handle_line(self.result, line)
        self.assertEqual(self.result.already_downloaded_count, 1)

    def test_already_id_pattern(self) -> None:
        line = "[download] dQw4w9WgXcQ has already been downloaded"
        self.parser.handle_line(self.result, line)
        self.assertEqual(self.result.already_downloaded_count, 1)

    def test_video_id_extraction(self) -> None:
        line = "[youtube] dQw4w9WgXcQ: Downloading webpage"
        self.parser.handle_line(self.result, line)

    def test_item_progress(self) -> None:
        line = "Downloading video 5 of 100"
        self.parser.handle_line(self.result, line)
        self.assertEqual(self.parser.current_item_index, 5)
        self.assertEqual(self.parser.current_item_total, 100)

    def test_rate_limit_error(self) -> None:
        line = "ERROR: Sign in to confirm you're not a bot"
        abort = self.parser.handle_line(self.result, line)
        self.assertEqual(self.result.rate_limited_count, 1)
        self.assertIsNotNone(abort)

    def test_members_only_error(self) -> None:
        line = "ERROR: Join this channel to get access to members-only content"
        self.parser.handle_line(self.result, line)
        self.assertEqual(self.result.members_only_count, 1)

    def test_unavailable_error(self) -> None:
        line = "ERROR: This video is private"
        self.parser.handle_line(self.result, line)
        self.assertEqual(self.result.unavailable_count, 1)

    def test_other_error(self) -> None:
        line = "ERROR: Some unknown error occurred"
        self.parser.handle_line(self.result, line)
        self.assertEqual(self.result.other_error_count, 1)

    def test_finalize_counts_current_video(self) -> None:
        line = "[youtube] dQw4w9WgXcQ: Downloading webpage"
        self.parser.handle_line(self.result, line)
        self.parser.finalize(self.result)
        self.assertEqual(self.result.success_count, 1)

    def test_set_item_total_hint(self) -> None:
        self.parser.set_item_total_hint(50)
        self.assertEqual(self.parser.current_item_total, 50)
        self.assertEqual(self.parser.current_item_index, 0)

    def test_set_item_total_hint_invalid(self) -> None:
        self.parser.set_item_total_hint(None)
        self.assertIsNone(self.parser.current_item_total)

        self.parser.set_item_total_hint(-5)
        self.assertIsNone(self.parser.current_item_total)

    def test_with_callbacks(self) -> None:
        progress_cb = MagicMock()
        message_cb = MagicMock()
        parser = YtDlpOutputParser(
            channel_name="Test",
            progress_callback=progress_cb,
            message_callback=message_cb,
        )
        result = DownloadResult()

        parser.handle_line(result, "[download] Destination: /path/to/file.mp4")
        message_cb.assert_called()

        parser.handle_line(result, "[download]  50.0% of 100.00MiB at 1.00MiB/s ETA 00:50")
        progress_cb.assert_called()

    def test_empty_line(self) -> None:
        result = self.parser.handle_line(self.result, "")
        self.assertIsNone(result)

        result = self.parser.handle_line(self.result, "   ")
        self.assertIsNone(result)


class TestDownloadResult(unittest.TestCase):
    def test_total_errors(self) -> None:
        result = DownloadResult(
            rate_limited_count=1,
            members_only_count=2,
            unavailable_count=3,
            other_error_count=4,
        )
        self.assertEqual(result.total_errors, 10)

    def test_expected_errors(self) -> None:
        result = DownloadResult(
            members_only_count=2,
            unavailable_count=3,
        )
        self.assertEqual(result.expected_errors, 5)

    def test_critical_errors(self) -> None:
        result = DownloadResult(other_error_count=5)
        self.assertEqual(result.critical_errors, 5)

    def test_has_rate_limit(self) -> None:
        result = DownloadResult()
        self.assertFalse(result.has_rate_limit)

        result.rate_limited_count = 1
        self.assertTrue(result.has_rate_limit)

    def test_has_members_only(self) -> None:
        result = DownloadResult()
        self.assertFalse(result.has_members_only)

        result.members_only_count = 1
        self.assertTrue(result.has_members_only)

    def test_merge(self) -> None:
        result1 = DownloadResult(
            success_count=5,
            rate_limited_count=1,
            members_only_count=2,
            unavailable_count=3,
            other_error_count=4,
            already_downloaded_count=10,
            last_rate_limit_error="error1",
        )

        result2 = DownloadResult(
            success_count=3,
            rate_limited_count=2,
            members_only_count=1,
            unavailable_count=1,
            other_error_count=1,
            already_downloaded_count=5,
            return_code=1,
            last_rate_limit_error="error2",
        )

        result1.merge(result2)

        self.assertEqual(result1.success_count, 8)
        self.assertEqual(result1.rate_limited_count, 3)
        self.assertEqual(result1.members_only_count, 3)
        self.assertEqual(result1.unavailable_count, 4)
        self.assertEqual(result1.other_error_count, 5)
        self.assertEqual(result1.already_downloaded_count, 15)
        self.assertEqual(result1.return_code, 1)
        self.assertEqual(result1.last_rate_limit_error, "error2")


if __name__ == "__main__":
    unittest.main()
