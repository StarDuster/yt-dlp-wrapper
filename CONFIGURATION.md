# 配置参考

本文档包含 yt-dlp-wrapper 的完整配置项和 CLI 参数说明。

配置文件位于 `yt_dlp_wrapper/config.py`，支持通过环境变量覆盖。

## 配置项

### 核心配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `YOUTUBE_DOWNLOAD_DIR` | 视频下载目录 | `~/yt-dlp-wrapper/downloads` |
| `YOUTUBE_CHANNEL_LIST_CACHE_TTL_DAYS` | 频道视频列表缓存有效期（天） | `7.0` |
| `YOUTUBE_ACCOUNTS_DIR` | 账号数据存储目录 | `~/.config/yt-dlp-wrapper/youtube-accounts` |

### 请求限流

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `YOUTUBE_SLEEP_REQUESTS` | 请求间隔（秒），对应 `yt-dlp --sleep-requests` | `2.0` |
| `YOUTUBE_SLEEP_INTERVAL` | 下载前最小等待时间（秒） | `5` |
| `YOUTUBE_MAX_SLEEP_INTERVAL` | 下载前最大等待时间（秒） | `8` |

### 账号池

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `YOUTUBE_ACCOUNT_COOLDOWN_SECONDS` | 账号被限流后的冷却时间（秒） | `900.0` |
| `YOUTUBE_ACCOUNT_SWITCH_MAX` | 最大账号切换次数，`0` 表示自动（每个账号最多尝试一次） | `0` |

### 反限流配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `INVIDIOUS_INSTANCE` | Invidious 实例地址 | `127.0.0.1:3000` |
| `YOUTUBE_PLAYER_CLIENT` | 播放器客户端类型（`web`/`android`/`ios`/`mweb`/`tv_embedded`） | `None` |
| `YOUTUBE_PO_TOKEN` | PO Token（用于绕过反爬验证） | `None` |
| `YOUTUBE_POT_PROVIDER` | PO Token 自动获取程序 | `None` |
| `YOUTUBE_EXTRACTOR_ARGS` | 额外的 extractor 参数（列表） | `[]` |

### Video List 模式专用

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `YOUTUBE_INPUT_LIST_FORMAT` | 视频格式选择 | `bv[height<=1080]+ba/best[height<=1080]` |
| `YOUTUBE_INPUT_LIST_FORMAT_SORT` | 格式排序规则 | `res,+tbr` |
| `YOUTUBE_INPUT_LIST_RETRIES` | 重试次数 | `3` |
| `YOUTUBE_INPUT_LIST_RETRY_BACKOFF_BASE` | 重试退避基数（秒） | `1.0` |
| `YOUTUBE_INPUT_LIST_RETRY_BACKOFF_MAX` | 重试退避上限（秒） | `60.0` |
| `YOUTUBE_INPUT_LIST_RETRY_BACKOFF_JITTER` | 重试退避抖动系数 | `0.1` |
| `YOUTUBE_INPUT_LIST_SLEEP` | 每个视频下载完成后的等待时间（秒） | `0.0` |

### 其他

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `YOUTUBE_GLOBAL_ARCHIVE_CSV` | 全局去重 CSV 文件路径 | `None` |

## CLI 参数

### download 命令

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

### account 命令

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
