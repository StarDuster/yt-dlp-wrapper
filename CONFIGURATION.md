# 配置参考

本文档包含 yt-dlp-wrapper 的完整 CLI 参数和配置项说明。

## CLI 参数

### download 命令

核心下载功能。

```bash
yt-dlp-wrapper download [OPTIONS]
```

**必选（二选一）：**

| 参数 | 说明 |
|------|------|
| `--channel-list FILE` | 频道列表文件路径 |
| `--video-list FILE` | 视频列表文件路径 |

**通用参数：**

| 参数 | 说明 |
|------|------|
| `--output-dir DIR` | 输出目录 |
| `--sleep SECONDS` | 覆盖 `YOUTUBE_SLEEP_REQUESTS` 配置 |
| `--no-invidious` | 禁用 Invidious 实例 |
| `--accounts-dir DIR` | 覆盖账号目录路径 |
| `--debug` | 输出调试信息 |

**Channel List 模式专用：**

| 参数 | 说明 |
|------|------|
| `--lang LANG` | 字幕语言（默认 `ja`） |

**Video List 模式专用：**

| 参数 | 说明 |
|------|------|
| `--workers N` | 并发下载数（默认 `8`） |
| `--limit N` | 限制下载数量 |
| `--disable-nvenc` | 禁用 NVENC，强制使用 libx264（CPU） |

### account 命令

账号管理功能（用于绕过年龄限制或会员验证）。

```bash
yt-dlp-wrapper account <action> [OPTIONS]
```

**Actions：**

| Action | 说明 |
|--------|------|
| `login` | 登录账号（弹出浏览器窗口） |
| `refresh` | 刷新 Cookies |
| `clear-auth` | 清除账号认证信息 |

**参数：**

| 参数 | 说明 |
|------|------|
| `--account NAME` | 指定账号名（用于账号池） |
| `--accounts-dir DIR` | 覆盖账号目录路径 |
| `--headless` | 无头模式（仅 login，需已有有效 profile） |

---

## 配置项

配置文件位于 `yt_dlp_wrapper/config.py`，支持通过环境变量覆盖。

### 核心配置

最基本的运行参数。

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `YOUTUBE_DOWNLOAD_DIR` | 视频下载目录 | `~/yt-dlp-wrapper/downloads` |
| `YOUTUBE_ACCOUNTS_DIR` | 账号数据存储目录 | `~/.config/yt-dlp-wrapper/youtube-accounts` |
| `YOUTUBE_CHANNEL_LIST_CACHE_TTL_DAYS` | 频道视频列表缓存有效期（天） | `7.0` |

### 反限流配置 (Anti-Rate-Limit)

关键配置，用于绕过 YouTube 的反爬虫和限流机制。

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `INVIDIOUS_INSTANCE` | Invidious 实例地址（用于加速解析） | `127.0.0.1:3000` |
| `YOUTUBE_PO_TOKEN` | PO Token（用于绕过反爬验证，**推荐配置**） | `None` |
| `YOUTUBE_POT_PROVIDER` | PO Token 自动获取程序 | `None` |
| `YOUTUBE_PLAYER_CLIENT` | 播放器客户端类型（`web`/`android`/`ios`/`mweb`/`tv_embedded`） | `None` |
| `YOUTUBE_EXTRACTOR_ARGS` | 额外的 extractor 参数（列表） | `[]` |

### 请求限流 (Throttling)

控制请求频率，避免 IP 被封禁。

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `YOUTUBE_SLEEP_REQUESTS` | 请求间隔（秒），对应 `yt-dlp --sleep-requests` | `2.0` |
| `YOUTUBE_SLEEP_INTERVAL` | 下载前最小等待时间（秒） | `5` |
| `YOUTUBE_MAX_SLEEP_INTERVAL` | 下载前最大等待时间（秒） | `8` |

### 账号池 (Account Pool)

多账号轮询配置，进一步降低单账号风险。

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `YOUTUBE_ACCOUNT_COOLDOWN_SECONDS` | 账号被限流后的冷却时间（秒） | `900.0` |
| `YOUTUBE_ACCOUNT_SWITCH_MAX` | 最大账号切换次数，`0` 表示自动（每个账号最多尝试一次） | `0` |

### Video List 模式专用

仅在指定 `--video-list` 时生效的配置。

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `YOUTUBE_INPUT_LIST_FORMAT` | 视频格式选择 | `bv[height<=1080][vcodec^=av01]+ba/bv[height<=1080][vcodec^=vp9]+ba/bv[height<=1080]+ba/best[height<=1080]` |
| `YOUTUBE_INPUT_LIST_FORMAT_SORT` | 格式排序规则 | `res,+tbr` |
| `YOUTUBE_INPUT_LIST_RETRIES` | 重试次数 | `3` |
| `YOUTUBE_INPUT_LIST_RETRY_BACKOFF_BASE` | 重试退避基数（秒） | `1.0` |
| `YOUTUBE_INPUT_LIST_RETRY_BACKOFF_MAX` | 重试退避上限（秒） | `60.0` |
| `YOUTUBE_INPUT_LIST_RETRY_BACKOFF_JITTER` | 重试退避抖动系数 | `0.1` |
| `YOUTUBE_INPUT_LIST_SLEEP` | 每个视频下载完成后的等待时间（秒） | `0.0` |

### 切片下载 / NVENC

高级功能：仅在 Video List 输入包含时间范围（`video_id,start,end`）时生效。

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `YOUTUBE_NVENC_CONCURRENCY` | NVENC 并发上限（超出则回退到 CPU） | `8` |
| `YOUTUBE_SEGMENT_MAX_HEIGHT` | 切片下载的视频最大高度 | `1080` |
| `YOUTUBE_NVENC_PRESET` | NVENC 预设（p1..p7，p1 最快） | `p1` |
| `YOUTUBE_NVENC_TUNE` | NVENC tune（`hq`/`ll`/`ull`/`lossless`） | `ll` |
| `YOUTUBE_NVENC_RC` | NVENC 码率控制（`constqp`/`vbr`/`cbr` 等） | `constqp` |
| `YOUTUBE_NVENC_QP` | constqp 质量参数（数值越低画质越高） | `18` |

> 备注：当输入列表包含切片任务时，`YOUTUBE_SLEEP_*` 与 `YOUTUBE_INPUT_LIST_SLEEP` 会被强制为 0，以避免额外等待拖慢转码流水线。

#### 并发控制
GeForce 显卡的 NVENC 并发会话数有限（通常为 3-8 路）。当 worker 数超过 `YOUTUBE_NVENC_CONCURRENCY` 时，多余任务会回退到 `libx264` (CPU)。

#### 编码质量预设参考

**h264_nvenc (GPU)**

| 用途 | preset | tune | rc | qp/crf | 说明 |
|------|--------|------|----|--------|------|
| 速度优先 | `p1` | `ll` | `constqp` | 23-28 | 最快，适合大批量转码 |
| 平衡（默认） | `p1` | `ll` | `constqp` | 18 | 速度与画质折中 |
| 高画质 | `p4` | `hq` | `constqp` | 15-18 | 较慢，画质更好 |
| 最高画质 | `p7` | `hq` | `constqp` | 12-15 | 最慢，适合存档 |

**libx264 (CPU)**

libx264 使用 `-preset` 和 `-crf` 控制质量。常用预设：

| 用途 | preset | crf | 说明 |
|------|--------|-----|------|
| 速度优先 | `veryfast` | 23-28 | 快速转码，体积较大 |
| 平衡 | `medium` | 18-23 | 默认预设，速度与质量折中 |
| 高画质 | `slow` | 15-18 | 较慢，压缩效率更高 |
| 最高画质 | `veryslow` | 12-18 | 最慢，适合存档 |

> CRF/QP 数值越低，画质越高，文件体积越大。通常 18 左右为"视觉无损"参考点。

### 其他

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `YOUTUBE_GLOBAL_ARCHIVE_CSV` | 全局去重 CSV 文件路径 | `None` |
