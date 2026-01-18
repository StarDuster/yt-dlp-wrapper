import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from yt_dlp_wrapper.downloaders import list as list_downloader


class TestListDownloaderUtils(unittest.TestCase):
    def test_parse_youtube_id(self) -> None:
        vid = "dQw4w9WgXcQ"
        cases = {
            vid: vid,
            f"https://youtu.be/{vid}": vid,
            f"https://www.youtube.com/watch?v={vid}&t=1": vid,
            f"https://m.youtube.com/watch?v={vid}": vid,
            f"https://www.youtube.com/shorts/{vid}": vid,
            f"https://www.youtube.com/embed/{vid}": vid,
            "not-a-video-id": None,
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(list_downloader.parse_youtube_id(raw), expected)

    def test_format_eta(self) -> None:
        self.assertEqual(list_downloader._format_eta(None), "")
        self.assertEqual(list_downloader._format_eta(-1), "")
        self.assertEqual(list_downloader._format_eta(0), "00:00")
        self.assertEqual(list_downloader._format_eta(59), "00:59")
        self.assertEqual(list_downloader._format_eta(60), "01:00")
        self.assertEqual(list_downloader._format_eta(3661), "1:01:01")

    def test_speed_column_item_per_second_is_stable_between_completions(self) -> None:
        col = list_downloader.SpeedColumn(worker_speeds={})

        class _Task:
            def __init__(self, completed: float):
                self.completed = completed
                self.total = 100
                self.fields = {"mode": "count"}

        t = _Task(0.0)
        with patch("yt_dlp_wrapper.downloaders.list.time.time") as mock_time:
            mock_time.return_value = 0.0
            self.assertEqual(str(col.render(t)), "0 it/s")

            mock_time.return_value = 20.0
            t.completed = 1.0
            self.assertEqual(str(col.render(t)), "0.05 it/s")

            mock_time.return_value = 25.0
            t.completed = 1.0
            self.assertEqual(str(col.render(t)), "0.05 it/s")

    def test_speed_column_overall_uses_segment_item_rates_when_available(self) -> None:
        col = list_downloader.SpeedColumn(worker_speeds={}, worker_item_rates={0: 0.10, 1: 0.25})

        class _Task:
            def __init__(self):
                self.completed = 0.0
                self.total = 100
                self.fields = {"mode": "count"}

        t = _Task()
        self.assertEqual(str(col.render(t)), "0.35 it/s")

    def test_backoff_sleep_deterministic(self) -> None:
        self.assertEqual(list_downloader._backoff_sleep(1, 1.0, 60.0, 0.0), 1.0)
        self.assertEqual(list_downloader._backoff_sleep(2, 1.0, 60.0, 0.0), 2.0)
        self.assertEqual(list_downloader._backoff_sleep(3, 1.0, 3.0, 0.0), 3.0)
        self.assertEqual(list_downloader._backoff_sleep("bad", 1.0, 60.0, 0.0), 1.0)
        self.assertEqual(list_downloader._backoff_sleep(1, 0.0, 60.0, 0.0), 0.0)

    def test_merge_extractor_args(self) -> None:
        args: dict[str, dict[str, list[str]]] = {}
        list_downloader._merge_extractor_args(args, "youtube:po_token=aaa;player_client=web")
        list_downloader._merge_extractor_args(args, ["youtube:po_token=bbb", "bad", "youtube:pot_provider=x"])

        self.assertEqual(args["youtube"]["po_token"], ["aaa", "bbb"])
        self.assertEqual(args["youtube"]["player_client"], ["web"])
        self.assertEqual(args["youtube"]["pot_provider"], ["x"])

    def test_archive_and_failed_file_helpers(self) -> None:
        with TemporaryDirectory() as td:
            tmp_dir = Path(td)
            archive = tmp_dir / "subdir" / "download.archive.txt"
            failed = tmp_dir / "subdir" / "failed.txt"
            lock = threading.Lock()

            list_downloader.append_archive_id(archive, "dQw4w9WgXcQ", lock)
            self.assertEqual(list_downloader.load_archive_ids(archive), {"dQw4w9WgXcQ"})

            list_downloader.append_failed_id(failed, "dQw4w9WgXcQ", "oops", lock)
            self.assertEqual(list_downloader.load_failed_ids(failed), {"dQw4w9WgXcQ"})

            seg_key = "dQw4w9WgXcQ:1500-2750"
            list_downloader.append_archive_id(archive, seg_key, lock)
            self.assertIn(seg_key, list_downloader.load_archive_ids(archive))

            list_downloader.append_failed_id(failed, seg_key, "oops2", lock)
            self.assertIn(seg_key, list_downloader.load_failed_ids(failed))

    def test_parse_input_line(self) -> None:
        vid = "dQw4w9WgXcQ"
        self.assertIsNone(list_downloader.parse_input_line(""))
        self.assertIsNone(list_downloader.parse_input_line("# comment"))
        self.assertIsNone(list_downloader.parse_input_line("not-a-video-id,1,2"))

        item = list_downloader.parse_input_line(vid)
        assert item is not None
        self.assertEqual(item["vid"], vid)
        self.assertFalse(item["has_range"])
        self.assertEqual(item["key"], vid)

        item2 = list_downloader.parse_input_line(f"{vid},1.5,2.75")
        assert item2 is not None
        self.assertEqual(item2["vid"], vid)
        self.assertTrue(item2["has_range"])
        self.assertEqual(item2["start"], 1.5)
        self.assertEqual(item2["end"], 2.75)
        self.assertEqual(item2["start_ms"], 1500)
        self.assertEqual(item2["end_ms"], 2750)
        self.assertEqual(item2["key"], "dQw4w9WgXcQ:1500-2750")

    def test_download_from_input_list_mixed_items_builds_segment_opts(self) -> None:
        vid = "dQw4w9WgXcQ"
        created_opts: list[dict] = []
        downloaded_urls: list[str] = []
        sleep_calls: list[float] = []

        class _FakeYDL:
            def __init__(self, opts: dict):
                created_opts.append(opts)

            def download(self, urls: list[str]) -> int:
                downloaded_urls.extend(urls)
                return 0

        def _fake_sleep(s: float) -> None:
            try:
                sleep_calls.append(float(s))
            except Exception:
                pass

        with TemporaryDirectory() as td:
            tmp_dir = Path(td)
            input_list = tmp_dir / "input.txt"
            input_list.write_text(f"{vid}\n{vid},1.5,2.75\n", encoding="utf-8")

            with (
                patch("yt_dlp_wrapper.downloaders.list.yt_dlp.YoutubeDL", _FakeYDL),
                patch("yt_dlp_wrapper.downloaders.list._has_nvidia_gpu", return_value=False),
                patch.object(list_downloader.config, "YOUTUBE_SLEEP_REQUESTS", 2.0),
                patch.object(list_downloader.config, "YOUTUBE_SLEEP_INTERVAL", 5.0),
                patch.object(list_downloader.config, "YOUTUBE_MAX_SLEEP_INTERVAL", 8.0),
                patch.object(list_downloader.config, "YOUTUBE_INPUT_LIST_SLEEP", 7.0),
                patch("yt_dlp_wrapper.downloaders.list.time.sleep", _fake_sleep),
            ):
                rc = list_downloader.download_from_input_list(
                    input_list_path=input_list,
                    output_dir=tmp_dir,
                    workers=1,
                    limit=0,
                    debug=False,
                    no_invidious=True,
                    accounts_dir=None,
                )

        self.assertEqual(rc, 0)
        self.assertEqual(len(downloaded_urls), 2)

        for opts in created_opts:
            self.assertNotIn("sleep_interval_requests", opts)
            self.assertNotIn("sleep_interval", opts)
            self.assertNotIn("max_sleep_interval", opts)

        self.assertNotIn(7.0, sleep_calls)

        seg_opts = next((o for o in created_opts if "download_ranges" in o), None)
        self.assertIsNotNone(seg_opts)
        assert seg_opts is not None
        self.assertTrue(seg_opts.get("force_keyframes_at_cuts"))

        self.assertIn("[1500-2750]", str(seg_opts.get("outtmpl")))

        eda = seg_opts.get("external_downloader_args") or {}
        ffmpeg_o = eda.get("ffmpeg_o") or []
        self.assertIn("libx264", ffmpeg_o)

    def test_classify_error(self) -> None:
        self.assertEqual(
            list_downloader.classify_error("ERROR: HTTP Error 429: Too Many Requests"),
            "rate_limit",
        )
        self.assertEqual(
            list_downloader.classify_error("ERROR: This video is private"),
            "unavailable",
        )
        self.assertEqual(
            list_downloader.classify_error("ERROR: HTTP Error 503: Service Unavailable"),
            "retry",
        )
        self.assertEqual(
            list_downloader.classify_error("ERROR: Something else"),
            "failed",
        )

    def test_download_from_input_list_sets_cookiefile_when_account_available(self) -> None:
        """
        Ensure we pass cookiefile to yt-dlp when an account cookies file exists.

        This is important for avoiding YouTube anti-bot flows ("confirm you're not a bot")
        and for accessing member-only content.
        """
        vid = "dQw4w9WgXcQ"
        created_opts: list[dict] = []

        class _FakeYDL:
            def __init__(self, opts: dict):
                created_opts.append(opts)

            def download(self, urls: list[str]) -> int:
                return 0

        with TemporaryDirectory() as td:
            tmp_dir = Path(td)
            input_list = tmp_dir / "input.txt"
            input_list.write_text(f"{vid}\n", encoding="utf-8")

            accounts_dir = tmp_dir / "accounts"
            cookies_file = accounts_dir / "acc1" / "youtube_cookies.txt"
            cookies_file.parent.mkdir(parents=True, exist_ok=True)
            cookies_file.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")

            with (
                patch("yt_dlp_wrapper.downloaders.list.yt_dlp.YoutubeDL", _FakeYDL),
                patch("yt_dlp_wrapper.downloaders.list._has_nvidia_gpu", return_value=False),
            ):
                rc = list_downloader.download_from_input_list(
                    input_list_path=input_list,
                    output_dir=tmp_dir,
                    workers=1,
                    limit=0,
                    debug=False,
                    no_invidious=True,
                    accounts_dir=accounts_dir,
                )

        self.assertEqual(rc, 0)
        self.assertTrue(created_opts, "Expected at least one YoutubeDL instance to be created")
        for opts in created_opts:
            with self.subTest(opts_keys=sorted(list(opts.keys()))):
                self.assertEqual(opts.get("cookiefile"), str(cookies_file))
