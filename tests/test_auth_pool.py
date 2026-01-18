import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from yt_dlp_wrapper.auth.pool import (
    YouTubeAccount,
    YouTubeAccountPool,
    discover_accounts,
    get_account_paths,
    load_accounts_from_config,
)


class TestYouTubeAccountPool(unittest.TestCase):
    def test_pick_next_no_accounts(self) -> None:
        pool = YouTubeAccountPool([])
        idx, wait_s = pool.pick_next(current_idx=None)
        self.assertEqual(idx, -1)
        self.assertEqual(wait_s, 0.0)

    def test_pick_least_used_no_accounts(self) -> None:
        pool = YouTubeAccountPool([])
        idx, wait_s = pool.pick_least_used()
        self.assertEqual(idx, -1)
        self.assertEqual(wait_s, 0.0)

    def test_pick_next_round_robin(self) -> None:
        accounts = [
            YouTubeAccount(name="a", cookies_file=Path("/tmp/a.txt")),
            YouTubeAccount(name="b", cookies_file=Path("/tmp/b.txt")),
            YouTubeAccount(name="c", cookies_file=Path("/tmp/c.txt")),
        ]
        pool = YouTubeAccountPool(accounts, cooldown_seconds=10.0)
        self.assertEqual(pool.pick_next(0)[0], 1)
        self.assertEqual(pool.pick_next(1)[0], 2)
        self.assertEqual(pool.pick_next(2)[0], 0)

    def test_pick_least_used_balances_by_video_count(self) -> None:
        accounts = [
            YouTubeAccount(name="a", cookies_file=Path("/tmp/a.txt")),
            YouTubeAccount(name="b", cookies_file=Path("/tmp/b.txt")),
            YouTubeAccount(name="c", cookies_file=Path("/tmp/c.txt")),
        ]
        pool = YouTubeAccountPool(accounts, cooldown_seconds=10.0, autosave=False)
        picks = [pool.pick_least_used()[0] for _ in range(6)]
        self.assertEqual(picks, [0, 1, 2, 0, 1, 2])
        self.assertEqual([a.usage_count for a in accounts], [2, 2, 2])

    def test_pick_least_used_respects_exclude_idx(self) -> None:
        accounts = [
            YouTubeAccount(name="a", cookies_file=Path("/tmp/a.txt")),
            YouTubeAccount(name="b", cookies_file=Path("/tmp/b.txt")),
            YouTubeAccount(name="c", cookies_file=Path("/tmp/c.txt")),
        ]
        pool = YouTubeAccountPool(accounts, cooldown_seconds=10.0, autosave=False)
        idx, wait_s = pool.pick_least_used(exclude_idx=0)
        self.assertEqual(wait_s, 0.0)
        self.assertNotEqual(idx, 0)

    def test_pick_least_used_all_cooling_down_returns_wait(self) -> None:
        accounts = [
            YouTubeAccount(name="a", cookies_file=Path("/tmp/a.txt"), cooldown_until=200.0),
            YouTubeAccount(name="b", cookies_file=Path("/tmp/b.txt"), cooldown_until=150.0),
        ]
        pool = YouTubeAccountPool(accounts, cooldown_seconds=10.0, autosave=False)
        with patch("yt_dlp_wrapper.auth.pool.time.time", return_value=100.0):
            idx, wait_s = pool.pick_least_used()
        self.assertEqual(idx, 1)
        self.assertAlmostEqual(wait_s, 50.0, places=6)
        self.assertEqual([a.usage_count for a in accounts], [0, 0])

    def test_pick_least_used_thread_safe_reservation(self) -> None:
        accounts = [
            YouTubeAccount(name="a", cookies_file=Path("/tmp/a.txt")),
            YouTubeAccount(name="b", cookies_file=Path("/tmp/b.txt")),
            YouTubeAccount(name="c", cookies_file=Path("/tmp/c.txt")),
        ]
        pool = YouTubeAccountPool(accounts, cooldown_seconds=10.0, autosave=False)

        results: list[int] = []
        results_lock = threading.Lock()
        barrier = threading.Barrier(4, timeout=1.0)

        def _runner() -> None:
            barrier.wait()
            idx, _wait = pool.pick_least_used()
            with results_lock:
                results.append(idx)

        threads = [threading.Thread(target=_runner, daemon=True) for _ in range(3)]
        for t in threads:
            t.start()
        barrier.wait()
        for t in threads:
            t.join(timeout=1.0)

        self.assertEqual(set(results), {0, 1, 2})
        self.assertEqual([a.usage_count for a in accounts], [1, 1, 1])

    def test_save_state_persists_usage_counters(self) -> None:
        with TemporaryDirectory() as td:
            tmp_dir = Path(td)
            state_file = tmp_dir / "state.json"
            accounts = [
                YouTubeAccount(name="a", cookies_file=Path("/tmp/a.txt")),
                YouTubeAccount(name="b", cookies_file=Path("/tmp/b.txt")),
            ]
            pool = YouTubeAccountPool(accounts, cooldown_seconds=10.0, state_file=state_file, autosave=False)
            with patch("yt_dlp_wrapper.auth.pool.time.time", return_value=100.0):
                idx, wait_s = pool.pick_least_used()
            self.assertEqual(wait_s, 0.0)
            self.assertEqual(idx, 0)

            ok = pool.save_state()
            self.assertTrue(ok)
            self.assertTrue(state_file.exists())

            data = json.loads(state_file.read_text(encoding="utf-8"))
            self.assertEqual(data.get("version"), 1)
            acc = data.get("accounts") or {}
            self.assertEqual(acc["a"]["usage_count"], 1)
            self.assertEqual(acc["a"]["usage_updated_at"], 100.0)
            self.assertEqual(acc["b"]["usage_count"], 0)
            self.assertEqual(acc["b"]["usage_updated_at"], 0.0)

    def test_pick_next_all_cooling_down_returns_wait(self) -> None:
        accounts = [
            YouTubeAccount(name="a", cookies_file=Path("/tmp/a.txt"), cooldown_until=200.0),
            YouTubeAccount(name="b", cookies_file=Path("/tmp/b.txt"), cooldown_until=150.0),
        ]
        pool = YouTubeAccountPool(accounts, cooldown_seconds=10.0)
        with patch("yt_dlp_wrapper.auth.pool.time.time", return_value=100.0):
            idx, wait_s = pool.pick_next(current_idx=0)
        self.assertEqual(idx, 1)
        self.assertAlmostEqual(wait_s, 50.0, places=6)

    def test_mark_rate_limited_sets_cooldown(self) -> None:
        accounts = [YouTubeAccount(name="a", cookies_file=Path("/tmp/a.txt"))]
        pool = YouTubeAccountPool(accounts, cooldown_seconds=10.0)
        with patch("yt_dlp_wrapper.auth.pool.time.time", return_value=100.0):
            until = pool.mark_rate_limited(0)
        self.assertAlmostEqual(until, 110.0, places=6)
        self.assertAlmostEqual(accounts[0].cooldown_until, 110.0, places=6)
        self.assertEqual(accounts[0].rate_limited, 1)



class TestAccountDiscovery(unittest.TestCase):
    def test_get_account_paths(self) -> None:
        with TemporaryDirectory() as td:
            base = Path(td)
            profile_dir, cookies_file = get_account_paths("acc_1", accounts_dir=base)
            self.assertEqual(profile_dir, base / "acc_1" / "browser-profile")
            self.assertEqual(cookies_file, base / "acc_1" / "youtube_cookies.txt")

    def test_discover_accounts(self) -> None:
        with TemporaryDirectory() as td:
            base = Path(td)
            (base / "b").mkdir(parents=True)
            (base / "a").mkdir(parents=True)
            (base / "a" / "youtube_cookies.txt").write_text("cookies", encoding="utf-8")
            (base / "b" / "youtube_cookies.txt").write_text("cookies", encoding="utf-8")

            accounts = discover_accounts(accounts_dir=base)
            self.assertEqual([a.name for a in accounts], ["a", "b"])
            self.assertTrue(all(a.cookies_file.exists() for a in accounts))

    def test_load_accounts_from_config_applies_persisted_usage(self) -> None:
        with TemporaryDirectory() as td:
            base = Path(td)
            (base / "a").mkdir(parents=True)
            (base / "b").mkdir(parents=True)
            (base / "a" / "youtube_cookies.txt").write_text("cookies", encoding="utf-8")
            (base / "b" / "youtube_cookies.txt").write_text("cookies", encoding="utf-8")

            state_file = base / ".account_pool_state.json"
            state_file.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "updated_at": 123.0,
                        "accounts": {
                            "a": {"usage_count": 10, "usage_updated_at": 111.0},
                            "b": {"usage_count": 3, "usage_updated_at": 222.0},
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            accounts, pool = load_accounts_from_config(accounts_dir=base)
            self.assertEqual([a.name for a in accounts], ["a", "b"])
            self.assertEqual([a.usage_count for a in accounts], [10, 3])
            self.assertEqual([a.usage_updated_at for a in accounts], [111.0, 222.0])
            self.assertIsNotNone(pool)

