# yt-dlp-wrapper

一个基于 `yt-dlp` 的封装工具，主要为了解决**批量下载**和**长期归档**时的几个痛点：

1.  **避免重复扫描**：下载大频道（几千个视频）时，如果中途断了，普通工具重启时又要从头扫描一遍列表。本工具会先“展开”所有视频 ID 到本地缓存，下次直接读文件继续下，不用再去请求 YouTube 接口翻页。
2.  **更稳的账号管理**：内置了浏览器模拟（Playwright），你可以像平时上网一样登录 YouTube，工具会自动导出 Cookies 给下载器用。这比手动抓 Cookie 更稳，也能支持账号轮换（账号池），防止单号下载太猛被限流。

## 1. 安装与环境

### 系统依赖
需要 `ffmpeg` 来处理音视频合并。

*   **Ubuntu/Debian**: `sudo apt-get install -y ffmpeg`
*   **macOS**: `brew install ffmpeg`

### Python 环境
建议用 Conda 或 venv 隔离环境，防止污染系统。

**方式 A: Conda (推荐)**
```bash
conda create -n yt-dlp-wrapper python=3.10 -y
conda activate yt-dlp-wrapper
```

**方式 B: venv**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 安装工具
```bash
# 安装依赖
python -m pip install -r requirements.txt
# 以编辑模式安装本项目
python -m pip install -e .
```

### 初始化浏览器组件
这是为了后面能弹窗登录账号。
```bash
python -m playwright install chromium
# Linux 如果缺依赖报错，运行这个:
# python -m playwright install-deps chromium
```

## 2. 配置 (Config)

项目根目录下的 `config.py` 包含了默认配置。你可以直接修改它，或者通过环境变量覆盖。

主要关注这几个：
*   `YOUTUBE_DOWNLOAD_DIR`: 视频下载保存到哪。
*   `YOUTUBE_ACCOUNTS_DIR`: 账号数据存哪（默认在 `~/.config/...`）。
*   `INVIDIOUS_INSTANCE`: 如果你想用 Invidious 实例来辅助解析，可以在这里填。
*   `YOUTUBE_PO_TOKEN` / `YOUTUBE_POT_PROVIDER`: 需要过 PO Token 验证时配置。

## 3. 账号准备

虽然不登录也能下，但为了高画质和抗限流，强烈建议登录。

### 登录账号
运行命令后，会弹出一个浏览器窗口。你在里面登录 YouTube，关掉窗口，Cookies 就自动保存好了。

```bash
# 登录默认账号
yt-dlp-wrapper account login

# 如果你有多个账号（账号池），可以指定名字登录
yt-dlp-wrapper account login --account backup_acc_1
```

### 维护账号
```bash
# 刷新 Cookies (打开浏览器重新走一遍流程)
yt-dlp-wrapper account refresh --account backup_acc_1

# 清除某个账号的认证信息
yt-dlp-wrapper account clear-auth --account backup_acc_1
```

**关于账号池的设计：**
工具会在下载时自动检测。如果当前账号被 YouTube 暂时限制（比如 429 Too Many Requests），它会自动切换到下一个配置好的账号继续下，并让当前账号“冷却”一会儿。

## 4. 下载指南

### 场景 A：下载整个频道 (Channel List)
**这是本工具最擅长的场景。**

1.  准备一个 `channels.txt`，一行一个链接：
    ```text
    https://www.youtube.com/@somechannel
    https://www.youtube.com/channel/UCxxxxxxxxxxxxxxx
    ```

2.  运行：
    ```bash
    yt-dlp-wrapper download \
      --channel-list channels.txt \
      --output-dir /data/youtube
    ```

**它会做什么？**
1.  **展开列表**：先扫一遍频道，把所有 Video ID 存到 `video_ids.txt`。
    *   *好处：* 哪怕频道有 5000 个视频，下次运行也只读这个文本文件，瞬间启动，不用再去请求 YouTube 翻页。
2.  **增量下载**：对比已下载记录 (`download.archive.txt`)，只下新的。
3.  **自动归档**：按频道名自动分文件夹存放。

### 场景 B：下载指定视频 (Video List)
适合下一些散乱的视频。

1.  准备 `videos.txt`，一行一个链接或 ID。
2.  运行（支持多线程）：
    ```bash
    yt-dlp-wrapper download \
      --video-list videos.txt \
      --workers 4
    ```

## 5. 进阶：关于 Invidious 和 PO Token

yt-dlp 支持的反限流插件：

### 什么是 Invidious？
Invidious 是一个开源的、注重隐私的 YouTube 前端替代品。
*   **作用**：除了作为观看界面，它常被用作 `yt-dlp` 的备用 API 来源（Instance API）。当 YouTube 官方接口对当前 IP 限流或屏蔽时，配置 Invidious 实例可以帮助获取视频元数据，规避部分访问限制。
*   **如何使用**：在 `config.py` 中配置 `INVIDIOUS_INSTANCE`（填入一个有效的 Invidious 实例 URL，如 `https://yewtu.be`）。

### 什么是 PO Token (Proof of Origin)？
PO Token（来源证明）是 YouTube 引入的一种反爬虫验证机制，由其 BotGuard (Web) 或 DroidGuard (Android) 组件生成。
*   **典型症状**：下载速度被严重限制（如卡在几十 KB/s），或者直接报错 `Sign in to confirm you're not a bot`。
*   **作用**：这个 Token 向 YouTube 证明当前的请求来自于一个真实的浏览器或合法的客户端环境，而非简单的脚本。提供有效的 Token 可以解除限速和 403 禁止访问。
*   **相关文档**：详见 [yt-dlp Wiki - "Sign in to confirm you're not a bot"](https://github.com/yt-dlp/yt-dlp/wiki/Extractor-Interactions#im-getting-sign-in-to-confirm-youre-not-a-bot-errors)。
*   **如何使用**：
    1.  **手动填入**：从浏览器中提取 Token，填入 `config.py` 的 `YOUTUBE_PO_TOKEN`。
    2.  **自动获取**：配置 `YOUTUBE_POT_PROVIDER` 使用第三方脚本动态生成。
    *原理上，这对应 `yt-dlp` 的参数：`--extractor-args "youtube:player_client=web;po_token=..."`。*

## 常见问题

*   **为什么要用 Playwright？**
现在 YouTube 对纯 HTTP 请求抓取的 Cookies 查得很严。用真实的浏览器环境登录并导出 Netscape 格式 Cookies，是目前最稳定、被限流概率最低的方法。
*   **下载到一半中断了怎么办？**
直接重新运行命令即可。
如果是**频道模式**，它会读取本地的 `video_ids.txt` 缓存，直接跳过已下载的，继续处理剩下的。
*   **如何手动干预下载列表？**
频道模式生成的 `video_ids.txt` 是纯文本。你可以手动编辑它，删掉你不想下的视频 ID，工具就会跳过它们。