import unittest

from yt_dlp_wrapper.core.diagnostics import DownloadResult
from yt_dlp_wrapper.core.parser import YtDlpOutputParser


class TestYtDlpOutputParserLogging(unittest.TestCase):
    def _parse_one(self, line: str):
        logs: list[tuple[str, str]] = []

        def log_cb(msg: str, level: str) -> None:
            logs.append((level, msg))

        # Important: simulate the common "progress_callback provided, message_callback missing" case.
        parser = YtDlpOutputParser(
            channel_name="chan",
            progress_callback=lambda *_: None,
            message_callback=None,
            log_callback=log_cb,
        )
        result = DownloadResult()
        abort = parser.handle_line(result, line)
        return result, abort, logs

    def test_rate_limit_logs_warning(self) -> None:
        result, abort, logs = self._parse_one("ERROR: HTTP Error 429: Too Many Requests")
        self.assertIs(abort, result)
        self.assertEqual(result.rate_limited_count, 1)
        self.assertTrue(any(level == "warning" for level, _ in logs))

    def test_members_only_logs_info(self) -> None:
        result, abort, logs = self._parse_one("ERROR: Join this channel to get access")
        self.assertIsNone(abort)
        self.assertEqual(result.members_only_count, 1)
        self.assertTrue(any(level == "info" for level, _ in logs))

    def test_unavailable_logs_info(self) -> None:
        result, abort, logs = self._parse_one("ERROR: This video is private")
        self.assertIsNone(abort)
        self.assertEqual(result.unavailable_count, 1)
        self.assertTrue(any(level == "info" for level, _ in logs))

    def test_other_logs_error(self) -> None:
        result, abort, logs = self._parse_one("ERROR: Some unknown failure")
        self.assertIsNone(abort)
        self.assertEqual(result.other_error_count, 1)
        self.assertTrue(any(level == "error" for level, _ in logs))

