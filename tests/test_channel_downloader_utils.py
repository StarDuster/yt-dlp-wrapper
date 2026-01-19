import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from yt_dlp_wrapper.downloaders.channel import YouTubeDownloader
from yt_dlp_wrapper.auth.pool import YouTubeAccount
from yt_dlp_wrapper.core.diagnostics import DownloadResult


class _FakePopen:
    def __init__(self, lines: list[str], *, returncode: int = 0):
        self.stdout = iter(lines)
        self._returncode = returncode

    def terminate(self) -> None:
        return None

    def kill(self) -> None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return self._returncode


class TestChannelDownloaderUtils(unittest.TestCase):
    def test_sanitize_filename(self) -> None:
        with TemporaryDirectory() as td:
            d = YouTubeDownloader(download_dir=Path(td), skip_dependency_checks=True)
            self.assertEqual(d._sanitize_filename("a/b:c*?\n"), "a_b_c__")
            self.assertEqual(d._sanitize_filename("a  b\tc"), "a b c")

    def test_derive_channel_key(self) -> None:
        with TemporaryDirectory() as td:
            d = YouTubeDownloader(download_dir=Path(td), skip_dependency_checks=True)
            self.assertEqual(d._derive_channel_key("https://www.youtube.com/@example/videos"), "@example")
            self.assertEqual(d._derive_channel_key("https://www.youtube.com/channel/UC123"), "channel-UC123")
            self.assertEqual(d._derive_channel_key("https://www.youtube.com/c/MyChannel"), "c-MyChannel")
            self.assertEqual(d._derive_channel_key(""), "channel")

    def test_classify_error(self) -> None:
        with TemporaryDirectory() as td:
            d = YouTubeDownloader(download_dir=Path(td), skip_dependency_checks=True)
            self.assertEqual(d._classify_error("ERROR: HTTP Error 429: Too Many Requests"), "rate_limit")
            self.assertEqual(d._classify_error("Join this channel to get access"), "members_only")
            self.assertEqual(d._classify_error("This video is private"), "unavailable")
            self.assertEqual(d._classify_error("Some unknown failure"), "other")

    def test_determine_final_status(self) -> None:
        with TemporaryDirectory() as td:
            d = YouTubeDownloader(download_dir=Path(td), skip_dependency_checks=True)

            status, msg = d._determine_final_status(
                DownloadResult(rate_limited_count=1, success_count=2), "chan", message_callback=lambda _: None
            )
            self.assertEqual(status, "youtube_rate_limited")
            self.assertTrue(msg and "Rate limited" in msg)

            status, msg = d._determine_final_status(
                DownloadResult(return_code=1), "chan", message_callback=lambda _: None
            )
            self.assertEqual(status, "youtube_failed")
            self.assertTrue(msg)

            status, msg = d._determine_final_status(
                DownloadResult(success_count=1, other_error_count=1), "chan", message_callback=lambda _: None
            )
            self.assertEqual(status, "youtube_partial")
            self.assertTrue(msg)

            status, msg = d._determine_final_status(
                DownloadResult(success_count=1, already_downloaded_count=1), "chan", message_callback=lambda _: None
            )
            self.assertEqual(status, "youtube_completed")
            self.assertIsNone(msg)

    def test_get_channel_video_ids_refreshes_expired_cache(self) -> None:
        with TemporaryDirectory() as td:
            tmp_dir = Path(td)
            cache_file = tmp_dir / "video_ids.txt"
            cache_file.write_text("https://www.youtube.com/watch?v=AAAAAAAAAAA\n", encoding="utf-8")

            d = YouTubeDownloader(download_dir=tmp_dir, skip_dependency_checks=True)

            cp = subprocess.CompletedProcess(
                args=["yt-dlp"], returncode=0, stdout="CCCCCCCCCCC\nDDDDDDDDDDD\n", stderr=""
            )
            with patch("yt_dlp_wrapper.downloaders.channel.subprocess.run", return_value=cp) as run_mock:
                ids = d._get_channel_video_ids(
                    "https://www.youtube.com/@example",
                    cache_file=cache_file,
                    cache_ttl_days=0,
                    debug=False,
                    message_callback=None,
                )

            self.assertEqual(ids, ["CCCCCCCCCCC", "DDDDDDDDDDD"])
            run_mock.assert_called_once()
            written = cache_file.read_text(encoding="utf-8")
            self.assertIn("https://www.youtube.com/watch?v=CCCCCCCCCCC\n", written)
            self.assertIn("https://www.youtube.com/watch?v=DDDDDDDDDDD\n", written)

    def test_run_ytdlp_with_progress_counts_single_video_success(self) -> None:
        with TemporaryDirectory() as td:
            d = YouTubeDownloader(download_dir=Path(td), skip_dependency_checks=True)
            fake = _FakePopen(
                [
                    "[youtube] AAAAAAAAAAA: Downloading webpage\n",
                    "[download] Destination: file.webm\n",
                    "[download] 100.0% of 1.00MiB at 1.00MiB/s ETA 00:00\n",
                ],
                returncode=0,
            )
            with patch("yt_dlp_wrapper.downloaders.channel.subprocess.Popen", return_value=fake):
                result = d._run_ytdlp_with_progress(
                    ["yt-dlp", "--version"],
                    "chan",
                    progress_callback=lambda *_: None,
                    message_callback=lambda *_: None,
                    debug=False,
                )

            self.assertEqual(result.return_code, 0)
            self.assertEqual(result.success_count, 1)
            self.assertEqual(result.total_errors, 0)

    def test_download_channel_includes_cookies_arg_when_account_provided(self) -> None:
        with TemporaryDirectory() as td:
            tmp_dir = Path(td)
            d = YouTubeDownloader(download_dir=tmp_dir, skip_dependency_checks=True)

            cookies_file = tmp_dir / "acc1.cookies.txt"
            cookies_file.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
            account = YouTubeAccount(name="acc1", cookies_file=cookies_file)

            seen_cmds: list[list[str]] = []

            def _fake_run(cmd: list, *_args, **_kwargs) -> DownloadResult:
                seen_cmds.append(list(cmd))
                return DownloadResult(return_code=0)

            with (
                patch.object(d, "_get_channel_video_ids", return_value=[]),
                patch.object(d, "_run_ytdlp_with_progress", side_effect=_fake_run),
            ):
                result = d.download_channel(
                    url="https://www.youtube.com/@example",
                    output_dir=tmp_dir,
                    debug=False,
                    no_invidious=True,
                    accounts=[account],
                    account_pool=None,
                )

            self.assertIsNotNone(result.final_status)
            self.assertGreaterEqual(len(seen_cmds), 1)

            for cmd in seen_cmds:
                with self.subTest(cmd=cmd):
                    self.assertIn("--cookies", cmd)
                    idx = cmd.index("--cookies")
                    self.assertEqual(cmd[idx + 1], str(cookies_file))