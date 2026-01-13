"""
Configuration for yt-dlp-wrapper.

Copy and edit as needed to match your environment.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent

# YouTube download output
YOUTUBE_DOWNLOAD_DIR = Path.home() / "yt-dlp-wrapper" / "downloads"
YOUTUBE_CHANNEL_LIST_CACHE_TTL_DAYS = 7.0

# yt-dlp Invidious configuration (single instance)
INVIDIOUS_INSTANCE = "127.0.0.1:3000"  # Local instance via Docker

# yt-dlp player client setting (optional)
# Options: "web", "android", "ios", "mweb", "tv_embedded"
YOUTUBE_PLAYER_CLIENT = None

# yt-dlp rate limiting settings (optional)
YOUTUBE_SLEEP_REQUESTS = 2
YOUTUBE_SLEEP_INTERVAL = 5
YOUTUBE_MAX_SLEEP_INTERVAL = 8

# YouTube browser authentication (Playwright)
YOUTUBE_BROWSER_PROFILE = Path.home() / ".config/yt-dlp-wrapper/browser-profile"
YOUTUBE_COOKIES_FILE = Path.home() / ".config/yt-dlp-wrapper/youtube_cookies.txt"

# Multi-account cookies pool
YOUTUBE_ACCOUNTS_DIR = Path.home() / ".config/yt-dlp-wrapper/youtube-accounts"
YOUTUBE_ACCOUNT_COOLDOWN_SECONDS = 900.0
YOUTUBE_ACCOUNT_SWITCH_MAX = 0  # 0 = auto (try each account at most once)

# PO token / POT provider (optional)
YOUTUBE_PO_TOKEN = None
YOUTUBE_POT_PROVIDER = None
# Extra extractor args passed as raw strings, e.g. ["youtube:po_token=xxx"]
YOUTUBE_EXTRACTOR_ARGS = []

# Optional global archive CSV for dedup across channels
YOUTUBE_GLOBAL_ARCHIVE_CSV = None

# Input-list downloader overrides
YOUTUBE_INPUT_LIST_FORMAT = (
    "bv[height<=1080][vcodec^=av01]+ba/"
    "bv[height<=1080][vcodec^=vp9]+ba/"
    "bv[height<=1080]+ba/"
    "best[height<=1080]"
)
YOUTUBE_INPUT_LIST_FORMAT_SORT = "res,+tbr"
YOUTUBE_INPUT_LIST_RETRIES = 3
YOUTUBE_INPUT_LIST_RETRY_BACKOFF_BASE = 1.0
YOUTUBE_INPUT_LIST_RETRY_BACKOFF_MAX = 60.0
YOUTUBE_INPUT_LIST_RETRY_BACKOFF_JITTER = 0.1
YOUTUBE_INPUT_LIST_SLEEP = 0.0

# Ensure directories exist (best effort)
try:
    YOUTUBE_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass
try:
    YOUTUBE_BROWSER_PROFILE.mkdir(parents=True, exist_ok=True)
except OSError:
    pass
try:
    YOUTUBE_COOKIES_FILE.parent.mkdir(parents=True, exist_ok=True)
except OSError:
    pass
try:
    YOUTUBE_ACCOUNTS_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass
