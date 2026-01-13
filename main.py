#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

import config
from auth.pool import get_account_paths, load_accounts_from_config
from core.channel_downloader import YouTubeDownloader


def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


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
    from core.list_downloader import download_from_input_list

    input_path = Path(args.video_list).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (config.YOUTUBE_DOWNLOAD_DIR / "input_list")
    )
    workers = int(args.workers or 8)
    limit = int(args.limit or 0)

    return download_from_input_list(
        input_list_path=input_path,
        output_dir=output_dir,
        workers=workers,
        limit=limit,
        debug=bool(args.debug),
        no_invidious=bool(args.no_invidious),
        accounts_dir=Path(args.accounts_dir).expanduser().resolve() if args.accounts_dir else None,
    )


def _account_login(args: argparse.Namespace) -> int:
    from auth.browser import YouTubeBrowserAuth

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
    from auth.browser import YouTubeBrowserAuth

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
    from auth.browser import YouTubeBrowserAuth

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
    parser = argparse.ArgumentParser(prog="yt-dlp-wrapper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download from channel list or video list")
    group = download.add_mutually_exclusive_group(required=True)
    group.add_argument("--channel-list", help="File containing channel URLs")
    group.add_argument("--video-list", help="File containing video IDs/URLs")
    download.add_argument("--output-dir", help="Output directory")
    download.add_argument("--workers", type=int, help="Worker count (video list mode)")
    download.add_argument("--limit", type=int, help="Limit items (video list mode)")
    download.add_argument("--lang", default="ja", help="Subtitle language (channel mode)")
    download.add_argument("--no-invidious", action="store_true", help="Disable Invidious")
    download.add_argument("--debug", action="store_true", help="Debug output")
    download.add_argument("--accounts-dir", help="Override accounts directory")

    account = subparsers.add_parser("account", help="Account management")
    account_sub = account.add_subparsers(dest="action", required=True)

    login = account_sub.add_parser("login", help="Login via browser")
    login.add_argument("--account", help="Account name (for multi-account pool)")
    login.add_argument("--headless", action="store_true", help="Headless login")
    login.add_argument("--accounts-dir", help="Override accounts directory")

    refresh = account_sub.add_parser("refresh", help="Refresh cookies via browser profile")
    refresh.add_argument("--account", help="Account name (for multi-account pool)")
    refresh.add_argument("--accounts-dir", help="Override accounts directory")

    clear = account_sub.add_parser("clear-auth", help="Clear auth data")
    clear.add_argument("--account", help="Account name (for multi-account pool)")
    clear.add_argument("--accounts-dir", help="Override accounts directory")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "download":
        _setup_logging(bool(args.debug))
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
