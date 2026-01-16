#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from . import config
from .auth.pool import get_account_paths, load_accounts_from_config
from .downloaders.channel import YouTubeDownloader


def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


def _non_negative_float(value: str) -> float:
    try:
        n = float(value)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid number: {value}") from e
    if n < 0:
        raise argparse.ArgumentTypeError("Must be >= 0")
    return n


def _apply_download_overrides(args: argparse.Namespace) -> None:
    if getattr(args, "sleep", None) is not None:
        config.YOUTUBE_SLEEP_REQUESTS = float(args.sleep)


def _read_list(path: Path) -> list[str]:
    items: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = (raw or "").strip()
            if not line or line.startswith("#"):
                continue
            items.append(line)
    return items


def _handle_channel_list(args: argparse.Namespace) -> int:
    input_path = Path(args.channel_list).expanduser().resolve()
    if not input_path.exists():
        print(f"Channel list not found: {input_path}")
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else config.YOUTUBE_DOWNLOAD_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    channels = _read_list(input_path)
    if not channels:
        print("Channel list is empty.")
        return 0

    downloader = YouTubeDownloader(download_dir=output_dir)
    accounts, pool = load_accounts_from_config(
        accounts_dir=Path(args.accounts_dir).expanduser().resolve() if args.accounts_dir else None
    )

    for idx, url in enumerate(channels, start=1):
        print(f"\n[{idx}/{len(channels)}] {url}")
        result = downloader.download_channel(
            url=url,
            output_dir=output_dir,
            channel_name=None,
            language=args.lang,
            debug=args.debug,
            no_invidious=args.no_invidious,
            accounts=accounts,
            account_pool=pool,
        )
        status = result.final_status or "unknown"
        print(
            f"Done: status={status} success={result.success_count} "
            f"already={result.already_downloaded_count} "
            f"rate_limited={result.rate_limited_count} "
            f"members_only={result.members_only_count} "
            f"unavailable={result.unavailable_count} "
            f"other_errors={result.other_error_count}"
        )

    return 0


def _handle_video_list(args: argparse.Namespace) -> int:
    from .downloaders.list import download_from_input_list

    input_path = Path(args.video_list).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (config.YOUTUBE_DOWNLOAD_DIR / "input_list")
    )
    workers = int(args.workers or 8)
    limit = int(args.limit or 0)
    disable_nvenc = bool(getattr(args, "disable_nvenc", False))

    return download_from_input_list(
        input_list_path=input_path,
        output_dir=output_dir,
        workers=workers,
        limit=limit,
        debug=bool(args.debug),
        no_invidious=bool(args.no_invidious),
        accounts_dir=Path(args.accounts_dir).expanduser().resolve() if args.accounts_dir else None,
        disable_nvenc=disable_nvenc,
    )


def _account_login(args: argparse.Namespace) -> int:
    from .auth.browser import YouTubeBrowserAuth

    if args.account:
        profile_dir, cookies_file = get_account_paths(
            args.account,
            accounts_dir=Path(args.accounts_dir).expanduser().resolve() if args.accounts_dir else None,
        )
        auth = YouTubeBrowserAuth(profile_dir=profile_dir, cookies_file=cookies_file)
        print(f"Account: {args.account}")
    else:
        auth = YouTubeBrowserAuth()

    print(f"Browser profile: {auth.profile_dir}")
    print(f"Cookies file: {auth.cookies_file}")
    success = auth.login(headless=bool(args.headless))
    print("Login successful." if success else "Login failed or timed out.")
    return 0 if success else 1


def _account_refresh(args: argparse.Namespace) -> int:
    from .auth.browser import YouTubeBrowserAuth

    if args.account:
        profile_dir, cookies_file = get_account_paths(
            args.account,
            accounts_dir=Path(args.accounts_dir).expanduser().resolve() if args.accounts_dir else None,
        )
        auth = YouTubeBrowserAuth(profile_dir=profile_dir, cookies_file=cookies_file)
        print(f"Account: {args.account}")
    else:
        auth = YouTubeBrowserAuth()

    print(f"Browser profile: {auth.profile_dir}")
    print(f"Cookies file: {auth.cookies_file}")
    success = auth.refresh_cookies()
    print("Cookies refreshed." if success else "Refresh failed.")
    return 0 if success else 1


def _account_clear(args: argparse.Namespace) -> int:
    from .auth.browser import YouTubeBrowserAuth

    if args.account:
        profile_dir, cookies_file = get_account_paths(
            args.account,
            accounts_dir=Path(args.accounts_dir).expanduser().resolve() if args.accounts_dir else None,
        )
        auth = YouTubeBrowserAuth(profile_dir=profile_dir, cookies_file=cookies_file)
        print(f"Account: {args.account}")
    else:
        auth = YouTubeBrowserAuth()

    print(f"Browser profile: {auth.profile_dir}")
    print(f"Cookies file: {auth.cookies_file}")

    response = input("Clear authentication data? [y/N]: ").strip().lower()
    if response != "y":
        print("Cancelled.")
        return 1

    success = auth.clear_auth()
    print("Authentication data cleared." if success else "Failed to clear auth data.")
    return 0 if success else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yt-dlp-wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="A yt-dlp wrapper for batch downloading YouTube channels and video lists.",
        epilog="""\
Examples:
  # Download from a channel list
  yt-dlp-wrapper download --channel-list channels.txt --output-dir ~/videos

  # Download from a video list (with 8 concurrent workers)
  yt-dlp-wrapper download --video-list videos.txt --workers 8

  # Login to an account (for member-only videos or rate limit bypass)
  yt-dlp-wrapper account login --account myaccount

  # Refresh cookies
  yt-dlp-wrapper account refresh --account myaccount

For more information: https://github.com/your-repo/yt-dlp-wrapper
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, title="commands", metavar="COMMAND")

    download = subparsers.add_parser(
        "download",
        help="Batch download from channel list or video list",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Batch download YouTube videos from a channel list or video list.",
        epilog="""\
Examples:
  # Download all videos from channels (default subtitle language: ja)
  yt-dlp-wrapper download --channel-list channels.txt

  # Specify output directory and subtitle language
  yt-dlp-wrapper download --channel-list channels.txt --output-dir ~/videos --lang zh

  # Download from video list with limit
  yt-dlp-wrapper download --video-list videos.txt --limit 100 --workers 4

  # Disable NVENC hardware encoding (use CPU encoding)
  yt-dlp-wrapper download --video-list videos.txt --disable-nvenc

  # Disable Invidious proxy
  yt-dlp-wrapper download --channel-list channels.txt --no-invidious

List file format:
  - One URL or video ID per line
  - Lines starting with # are comments
  - Empty lines are ignored
""",
    )
    group = download.add_mutually_exclusive_group(required=True)
    group.add_argument("--channel-list", metavar="FILE", help="Path to channel list file (one channel URL per line)")
    group.add_argument("--video-list", metavar="FILE", help="Path to video list file (one video ID or URL per line)")
    download.add_argument("--output-dir", "-o", metavar="DIR", help="Output directory (default: YOUTUBE_DOWNLOAD_DIR from config)")
    download.add_argument("--workers", "-w", type=int, default=8, metavar="N", help="Number of concurrent workers (video list mode only, default: 8)")
    download.add_argument(
        "--disable-nvenc",
        action="store_true",
        help="Disable NVENC hardware encoding; force libx264 (CPU) for segment transcoding",
    )
    download.add_argument("--limit", "-n", type=int, default=0, metavar="N", help="Limit number of downloads (video list mode only, 0=unlimited)")
    download.add_argument("--lang", "-l", default="ja", metavar="LANG", help="Subtitle language code (channel mode only, default: ja)")
    download.add_argument(
        "--sleep",
        type=_non_negative_float,
        metavar="SEC",
        help="Seconds to sleep between requests (overrides config.YOUTUBE_SLEEP_REQUESTS)",
    )
    download.add_argument("--no-invidious", action="store_true", help="Disable Invidious proxy for fetching video lists")
    download.add_argument("--debug", "-d", action="store_true", help="Enable debug output")
    download.add_argument("--accounts-dir", metavar="DIR", help="Override accounts directory path")

    account = subparsers.add_parser(
        "account",
        help="Account management (login, refresh cookies, clear auth)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Manage YouTube account authentication with multi-account pool support.",
        epilog="""\
Examples:
  # Login to default account (opens browser)
  yt-dlp-wrapper account login

  # Login to a specific account
  yt-dlp-wrapper account login --account work

  # Headless login (requires existing valid session)
  yt-dlp-wrapper account login --account work --headless

  # Refresh cookies (no password required)
  yt-dlp-wrapper account refresh --account work

  # Clear account authentication data
  yt-dlp-wrapper account clear-auth --account work

Accounts directory structure:
  accounts/
  ├── account1/
  │   ├── profile/      # Browser profile directory
  │   └── cookies.txt   # Netscape format cookies
  └── account2/
      ├── profile/
      └── cookies.txt
""",
    )
    account_sub = account.add_subparsers(dest="action", required=True, title="actions", metavar="ACTION")

    login = account_sub.add_parser(
        "login",
        help="Login via browser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Open browser to login to YouTube and automatically save cookies.",
    )
    login.add_argument("--account", "-a", metavar="NAME", help="Account name (for multi-account pool)")
    login.add_argument("--headless", action="store_true", help="Headless mode (no browser window)")
    login.add_argument("--accounts-dir", metavar="DIR", help="Override accounts directory path")

    refresh = account_sub.add_parser(
        "refresh",
        help="Refresh account cookies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Refresh cookies using saved browser profile without re-login.",
    )
    refresh.add_argument("--account", "-a", metavar="NAME", help="Account name (for multi-account pool)")
    refresh.add_argument("--accounts-dir", metavar="DIR", help="Override accounts directory path")

    clear = account_sub.add_parser(
        "clear-auth",
        help="Clear account authentication data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Delete browser profile and cookies file (requires confirmation).",
    )
    clear.add_argument("--account", "-a", metavar="NAME", help="Account name (for multi-account pool)")
    clear.add_argument("--accounts-dir", metavar="DIR", help="Override accounts directory path")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "download":
        _setup_logging(bool(args.debug))
        _apply_download_overrides(args)
        if args.channel_list:
            return _handle_channel_list(args)
        if args.video_list:
            return _handle_video_list(args)

    if args.command == "account":
        if args.action == "login":
            return _account_login(args)
        if args.action == "refresh":
            return _account_refresh(args)
        if args.action == "clear-auth":
            return _account_clear(args)
        print("Missing account action (login/refresh/clear-auth).")
        return 1

    print("Unknown command.")
    return 1


def cli() -> None:
    sys.exit(main())


if __name__ == "__main__":
    cli()
