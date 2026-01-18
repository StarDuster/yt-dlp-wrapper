"""
Configuration for yt-dlp-wrapper.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent

# Output directory
YOUTUBE_DOWNLOAD_DIR = Path.home() / "yt-dlp-wrapper" / "downloads"
YOUTUBE_CHANNEL_LIST_CACHE_TTL_DAYS = 7.0

# Invidious instance (local Docker recommended)
INVIDIOUS_INSTANCE = "127.0.0.1:3000"

# Player client (avoid SABR-affected clients like web_safari)
YOUTUBE_PLAYER_CLIENT = "android,ios,tv_embedded"

# Rate limiting
YOUTUBE_SLEEP_REQUESTS = 2.0
YOUTUBE_SLEEP_INTERVAL = 5
YOUTUBE_MAX_SLEEP_INTERVAL = 8

# Browser authentication (Playwright)
YOUTUBE_BROWSER_PROFILE = Path.home() / ".config/yt-dlp-wrapper/browser-profile"
YOUTUBE_COOKIES_FILE = Path.home() / ".config/yt-dlp-wrapper/youtube_cookies.txt"

# Multi-account pool
YOUTUBE_ACCOUNTS_DIR = Path.home() / ".config/yt-dlp-wrapper/youtube-accounts"
YOUTUBE_ACCOUNT_COOLDOWN_SECONDS = 900.0
YOUTUBE_ACCOUNT_SWITCH_MAX = 0  # 0 = auto (try each account at most once)

# PO token / POT provider
YOUTUBE_PO_TOKEN = None
YOUTUBE_POT_PROVIDER = None
YOUTUBE_EXTRACTOR_ARGS = []

# Global archive CSV for dedup across channels
YOUTUBE_GLOBAL_ARCHIVE_CSV = None

# Input-list downloader format
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

# NVENC concurrency for segment downloads
YOUTUBE_NVENC_CONCURRENCY = 8

# Segment download settings
YOUTUBE_SEGMENT_MAX_HEIGHT = 1080

# NVENC encoding (speed priority, high quality)
YOUTUBE_NVENC_PRESET = "p1"    # p1=fastest, p7=slowest
YOUTUBE_NVENC_TUNE = "ll"      # ll=low latency
YOUTUBE_NVENC_RC = "constqp"   # constqp=fastest
YOUTUBE_NVENC_QP = 18          # lower=better quality, larger file

# Ensure directories exist
for _dir in [YOUTUBE_DOWNLOAD_DIR, YOUTUBE_BROWSER_PROFILE, YOUTUBE_ACCOUNTS_DIR]:
    try:
        _dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
try:
    YOUTUBE_COOKIES_FILE.parent.mkdir(parents=True, exist_ok=True)
except OSError:
    pass
