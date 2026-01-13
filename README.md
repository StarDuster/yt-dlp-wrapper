# yt-dlp-wrapper

基于 `yt-dlp` 的封装工具，针对批量下载与长期归档场景进行优化。

## 特性

- **本地缓存视频列表**：首次运行时将频道内所有视频 ID 展开并缓存至本地文件，后续运行无需重新请求 YouTube API 翻页，提升重启时的恢复速度
- **账号池与自动轮换**：支持配置多个 YouTube 账号，当某账号触发请求限制 (HTTP 429) 时自动切换至其他账号并设置冷却时间
- **基于 Playwright 的认证**：通过真实浏览器环境完成登录，自动导出 Cookies

## 安装

### 系统依赖

需要 `ffmpeg` 处理音视频合并：

- **Ubuntu/Debian**: `sudo apt-get install -y ffmpeg`
- **macOS**: `brew install ffmpeg`

### 安装项目

推荐使用 [uv](https://github.com/astral-sh/uv)：

```bash
uv sync
uv run playwright install chromium
```

## 快速开始

### 1. 登录账号

```bash
yt-dlp-wrapper account login
```

运行后会弹出浏览器窗口，完成登录并关闭窗口即可。

### 2. 下载频道

```bash
# 准备 channels.txt，每行一个频道链接
yt-dlp-wrapper download --channel-list channels.txt --output-dir /data/youtube
```

### 3. 下载视频列表

```bash
# 准备 videos.txt，每行一个链接或视频 ID
yt-dlp-wrapper download --video-list videos.txt --workers 4
```

## 账号池

配置多个账号后，工具会在检测到限流时自动切换。

```bash
# 登录多个账号
yt-dlp-wrapper account login --account acc_1
yt-dlp-wrapper account login --account acc_2
```

账号目录结构：

```
~/.config/yt-dlp-wrapper/youtube-accounts/
├── acc_1/
│   ├── browser-profile/
│   └── youtube_cookies.txt
└── acc_2/
    └── ...
```

## 配置

配置文件位于 `yt_dlp_wrapper/config.py`，主要配置项：

| 配置项 | 说明 |
|--------|------|
| `YOUTUBE_DOWNLOAD_DIR` | 下载目录 |
| `YOUTUBE_SLEEP_REQUESTS` | 请求间隔（秒） |
| `INVIDIOUS_INSTANCE` | Invidious 实例地址 |
| `YOUTUBE_PO_TOKEN` | PO Token |

完整配置和 CLI 参数说明见 [CONFIGURATION.md](./CONFIGURATION.md)。

## 进阶：Invidious 与 PO Token

### 什么是 Invidious？

[Invidious](https://invidious.io/) 是一个开源的、注重隐私的 YouTube 前端替代品。

**为什么需要 Invidious？**

除了作为观看界面，Invidious 常被用作 `yt-dlp` 的备用 API 来源。当 YouTube 官方接口对当前 IP 限流或屏蔽时，配置 Invidious 实例可以帮助获取视频元数据，绕过部分限制。

需要注意的是，`pip` 安装的只是客户端插件，你还需要配置一个可用的 Invidious 服务端实例。

**方案 A：使用公共实例**

在配置中填入公共实例地址（如 `https://inv.tux.pizza`），实例列表见 [Invidious Instances](https://docs.invidious.io/instances/)。

公共实例可能因负载问题导致不稳定，建议仅用于测试。

**方案 B：本地部署（推荐）**

使用 Docker 在本地启动服务，最稳定且无网络延迟：

```bash
git clone https://github.com/iv-org/invidious.git
cd invidious
docker compose up -d
```

然后在配置中设置：`INVIDIOUS_INSTANCE = "127.0.0.1:3000"`

### 什么是 PO Token？

PO Token (Proof of Origin) 是 YouTube 引入的一种反爬虫验证机制，由其 BotGuard (Web) 或 DroidGuard (Android) 组件生成。

**为什么需要 PO Token？**

这个 Token 向 YouTube 证明当前的请求来自于一个真实的浏览器或合法的客户端环境，而非简单的脚本。提供有效的 Token 可以解除限速和 403 禁止访问。

**典型症状**：下载速度被严重限制（如卡在几十 KB/s），或者直接报错 `Sign in to confirm you're not a bot`。

**如何配置？**

1. **手动填入**：从浏览器中提取 Token，填入 `YOUTUBE_PO_TOKEN`
2. **自动获取**：配置 `YOUTUBE_POT_PROVIDER` 使用第三方脚本动态生成

详见 [yt-dlp Wiki - "Sign in to confirm you're not a bot"](https://github.com/yt-dlp/yt-dlp/wiki/Extractor-Interactions#im-getting-sign-in-to-confirm-youre-not-a-bot-errors)。

## FAQ

**下载中断后如何恢复？**

直接重新运行命令。频道模式会读取本地缓存的 `video_ids.txt`，跳过已下载视频继续处理。

**Members-only 视频如何处理？**

工具默认会跳过需要会员订阅的视频，并在日志中标记。
