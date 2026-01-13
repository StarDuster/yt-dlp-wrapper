# yt-dlp-wrapper 计划

目标：将 `podcast-agent` 中的 yt-dlp 相关能力抽离为独立工具 `yt-dlp-wrapper`，仅保留两种输入（频道列表 / 视频 ID 列表），并保留账号池、浏览器导入 cookies、PO token、Invidious。

## 现状分析（源项目 podcast-agent）

- 频道下载：`crawler/youtube_downloader.py`（yt-dlp CLI + 进度解析 + 限流检测 + 账号切换 + Invidious + 列表缓存/去重），但耦合 DB（PodcastSource / status）。
- 视频列表下载：`crawler/youtube_list_downloader.py`（yt_dlp API + 多线程 + 账号池 + Invidious + 插件预加载），已无 DB 依赖。
- 认证/账号池：`crawler/browser_auth.py` + `crawler/youtube_account_pool.py`（Playwright 登录 + cookies 导出 + 冷却轮换）。
- PO token：依赖层包含 `bgutil-ytdlp-pot-provider`；新工具保留插件依赖与 extractor-args 配置位。

## 当前实现（本仓库）

目录结构：
```text
yt-dlp-wrapper/
├── main.py          # CLI（yt-dlp-wrapper download / account ...）
├── config.py        # 配置
├── pyproject.toml   # 包/命令行入口（yt-dlp-wrapper）
├── requirements.txt # 依赖（可用于快速 pip 安装）
├── core/
│   ├── channel_downloader.py  # 频道下载（去 DB 化，channel -> list -> batch-file）
│   ├── list_downloader.py     # 视频列表下载（yt_dlp API）
│   └── models.py              # 下载结果/错误分类模型
└── auth/
    ├── browser.py             # Playwright 登录导出 cookies
    └── pool.py                # 账号池发现/冷却切换
```

核心设计点：
- **DB 全移除**：频道下载从 `podcast_id` 改为 `url/output_dir` 驱动。
- **频道展开成 list**：频道模式会先枚举视频 ID 并写入 `video_ids.txt`，然后用 `--batch-file` 下载（失败才回退到直接用频道 URL）。
- **账号池保留**：默认目录 `~/.config/yt-dlp-wrapper/youtube-accounts`，并可用 `--accounts-dir` 或 `config.YOUTUBE_ACCOUNTS_DIR` 覆盖。
- **Invidious**：仅支持单实例（无轮换/回退）。
- **PO token**：通过 `config.YOUTUBE_PO_TOKEN / YOUTUBE_POT_PROVIDER / YOUTUBE_EXTRACTOR_ARGS` 注入 extractor-args。

## 下一步（可选增强）

- `download` 增加并发下载多个频道（按频道粒度并发）。
- 增加 `expand` 子命令：仅做频道→列表展开（不下载），便于调试/复用。
- 增加轻量测试：验证频道列表缓存解析逻辑（不依赖网络）。
