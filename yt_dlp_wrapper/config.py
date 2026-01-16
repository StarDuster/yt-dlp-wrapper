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
# --sleep-requests: seconds to sleep between requests during data extraction
YOUTUBE_SLEEP_REQUESTS = 2.0
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

# NVENC concurrency (segment download with --force-keyframes-at-cuts)
# GeForce cards usually have a limited number of concurrent NVENC sessions.
# Keep this small (e.g. 3-4). If you run multiple yt-dlp-wrapper processes,
# you may need to lower this value.
YOUTUBE_NVENC_CONCURRENCY = 8

# Segment download/transcode tuning (speed vs quality)
# - Keep timestamps accurate; these only affect encoded video quality/speed.
# - If you want maximum speed, keep preset at p1/p2 and use constqp.
YOUTUBE_SEGMENT_MAX_HEIGHT = 1080  # set to 720/480 to speed up significantly

# NVENC video encoding options used for segment tasks
# 当前配置：体积无所谓，速度优先，画质尽量高
YOUTUBE_NVENC_PRESET = "p1"        # p1..p7 (p1=最快, p7=最慢但质量最好)
YOUTUBE_NVENC_TUNE = "ll"          # hq/ll/ull/lossless (ll=低延迟, 快)
YOUTUBE_NVENC_RC = "constqp"       # constqp/vbr/cbr/vbr_hq... (constqp=最快)
YOUTUBE_NVENC_QP = 18              # QP 越低画质越好，体积越大 (18=高画质)

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
