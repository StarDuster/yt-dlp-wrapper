import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from yt_dlp_wrapper.core import list_downloader


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

            # segment key should also be supported
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

