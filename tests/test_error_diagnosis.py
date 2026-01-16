import unittest

from yt_dlp_wrapper.core.error_diagnosis import (
    classify_channel_error_line,
    diagnose_ffmpeg_error,
    extract_http_status_from_text,
)


class TestErrorDiagnosis(unittest.TestCase):
    def test_extract_http_status_from_text(self) -> None:
        self.assertEqual(extract_http_status_from_text("ERROR: HTTP Error 429: Too Many Requests"), 429)
        self.assertEqual(extract_http_status_from_text("Server returned 403 Forbidden"), 403)
        self.assertEqual(extract_http_status_from_text("returned error: 410"), 410)
        self.assertIsNone(extract_http_status_from_text("ffmpeg exited with code 183"))

    def test_diagnose_ffmpeg_error(self) -> None:
        diag, http, hint = diagnose_ffmpeg_error("ERROR: HTTP Error 429: Too Many Requests\nffmpeg exited with code 183")
        self.assertEqual(diag, "rate_limit")
        self.assertEqual(http, 429)
        self.assertTrue(hint and "Suspected rate limit" in hint and "429" in hint)

        diag, http, hint = diagnose_ffmpeg_error("ERROR: This video is private\nffmpeg exited with code 183")
        self.assertEqual(diag, "access")
        self.assertIsNone(http)
        self.assertEqual(hint, "Suspected access restriction")

        diag, http, hint = diagnose_ffmpeg_error("ERROR: Server returned 403 Forbidden\nffmpeg exited with code 183")
        self.assertEqual(diag, "access")
        self.assertEqual(http, 403)
        self.assertTrue(hint and "Suspected access restriction" in hint and "403" in hint)

    def test_classify_channel_error_line(self) -> None:
        self.assertEqual(classify_channel_error_line("ERROR: HTTP Error 429: Too Many Requests"), "rate_limit")
        self.assertEqual(classify_channel_error_line("Join this channel to get access"), "members_only")
        self.assertEqual(classify_channel_error_line("This video is private"), "unavailable")
        self.assertEqual(classify_channel_error_line("Some unknown failure"), "other")


if __name__ == "__main__":
    unittest.main()

