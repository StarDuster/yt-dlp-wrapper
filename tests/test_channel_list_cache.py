import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from yt_dlp_wrapper.downloaders.channel import YouTubeDownloader


class TestChannelListCache(unittest.TestCase):
    def test_parse_ids_from_cache_file(self) -> None:
        with TemporaryDirectory() as td:
            tmp_dir = Path(td)
            cache_file = tmp_dir / "video_ids.txt"
            cache_file.write_text(
                "\n".join(
                    [
                        "# comment",
                        "https://www.youtube.com/watch?v=AAAAAAAAAAA",
                        "BBBBBBBBBBB",
                        "not-an-id",
                        "",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            downloader = YouTubeDownloader(download_dir=tmp_dir, skip_dependency_checks=True)
            ids = downloader._get_channel_video_ids(
                "https://www.youtube.com/@example",
                cache_file=cache_file,
                cache_ttl_days=None,
                debug=False,
                message_callback=None,
            )
            self.assertEqual(ids, ["AAAAAAAAAAA", "BBBBBBBBBBB"])
