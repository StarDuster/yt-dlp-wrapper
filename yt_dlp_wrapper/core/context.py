"""
Shared yt-dlp configuration builder.

This module centralizes common extractor-args/invidious settings so both the list and
channel downloaders stay small and consistent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .. import config


def _coerce_str_list(raw: object | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        return [s] if s else []
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for x in raw:
            sx = str(x).strip()
            if sx:
                out.append(sx)
        return out
    return []


def select_invidious_instance(*, no_invidious: bool) -> Optional[str]:
    """
    Select a single Invidious instance. No rotation/fallback.
    """
    if no_invidious:
        return None
    inst = getattr(config, "INVIDIOUS_INSTANCE", None)
    return inst or None


def _get_player_clients_from_env_or_config() -> list[str]:
    raw = os.environ.get("YTDLP_WRAPPER_PLAYER_CLIENT")
    if raw is None or not str(raw).strip():
        raw = getattr(config, "YOUTUBE_PLAYER_CLIENT", None)

    if raw is None:
        return []

    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]

    # Support comma-separated values like: "tv,-web_safari"
    return [s.strip() for s in str(raw).split(",") if s.strip()]


def merge_extractor_args_dict(
    extractor_args: dict[str, dict[str, list[str]]],
    raw_args: object | None,
) -> None:
    """
    Merge raw yt-dlp --extractor-args style strings into the Python API dict form.

    Supported input:
    - "youtube:po_token=aaa;player_client=web"
    - ["youtube:po_token=bbb", "youtubetab:approximate_date=true"]
    """
    items = _coerce_str_list(raw_args)
    if not items:
        return

    for raw in items:
        if ":" not in raw:
            continue
        extractor, rest = raw.split(":", 1)
        extractor = extractor.strip()
        if not extractor or not rest:
            continue
        for part in rest.split(";"):
            part = part.strip()
            if not part or "=" not in part:
                continue
            key, val = part.split("=", 1)
            key = key.strip()
            val = val.strip()
            if not key:
                continue
            extractor_args.setdefault(extractor, {}).setdefault(key, []).append(val)


def build_extractor_args_dict(*, no_invidious: bool) -> dict[str, dict[str, list[str]]]:
    extractor_args: dict[str, dict[str, list[str]]] = {
        "youtubetab": {"approximate_date": ["true"]},
    }

    youtube_args: dict[str, list[str]] = {}

    player_clients = _get_player_clients_from_env_or_config()
    if player_clients:
        youtube_args["player_client"] = player_clients

    invidious_instance = select_invidious_instance(no_invidious=no_invidious)
    if invidious_instance:
        youtube_args["invidious_instance"] = [f"https://{invidious_instance}"]

    po_token = getattr(config, "YOUTUBE_PO_TOKEN", None)
    pot_provider = getattr(config, "YOUTUBE_POT_PROVIDER", None)
    if po_token:
        youtube_args["po_token"] = [str(po_token)]
    if pot_provider:
        youtube_args["pot_provider"] = [str(pot_provider)]

    if youtube_args:
        extractor_args["youtube"] = youtube_args

    merge_extractor_args_dict(extractor_args, getattr(config, "YOUTUBE_EXTRACTOR_ARGS", None))
    return extractor_args


def build_extractor_args_cli(*, no_invidious: bool) -> list[str]:
    """
    Build a list of strings suitable for passing to yt-dlp CLI via repeated:
      --extractor-args <arg>
    """
    args: list[str] = ["youtubetab:approximate_date=true"]

    youtube_parts: list[str] = []

    player_clients = _get_player_clients_from_env_or_config()
    for pc in player_clients:
        youtube_parts.append(f"player_client={pc}")

    invidious_instance = select_invidious_instance(no_invidious=no_invidious)
    if invidious_instance:
        youtube_parts.append(f"invidious_instance=https://{invidious_instance}")

    po_token = getattr(config, "YOUTUBE_PO_TOKEN", None)
    pot_provider = getattr(config, "YOUTUBE_POT_PROVIDER", None)
    if po_token:
        youtube_parts.append(f"po_token={po_token}")
    if pot_provider:
        youtube_parts.append(f"pot_provider={pot_provider}")

    if youtube_parts:
        args.append("youtube:" + ";".join(youtube_parts))

    # Preserve any extra args as raw strings (they may include non key=value syntax).
    args.extend(_coerce_str_list(getattr(config, "YOUTUBE_EXTRACTOR_ARGS", None)))
    return args


@dataclass
class YtDlpContext:
    """
    A small wrapper around extractor args & Invidious selection, derived from config/env.
    """

    no_invidious: bool = False

    @property
    def invidious_instance(self) -> Optional[str]:
        return select_invidious_instance(no_invidious=self.no_invidious)

    @property
    def extractor_args_dict(self) -> dict[str, dict[str, list[str]]]:
        return build_extractor_args_dict(no_invidious=self.no_invidious)

    @property
    def extractor_args_cli(self) -> list[str]:
        return build_extractor_args_cli(no_invidious=self.no_invidious)

    def apply_common_python_api_opts(self, opts: dict) -> None:
        """
        Apply common yt_dlp.YoutubeDL options shared by our tools.
        """
        opts["extractor_args"] = self.extractor_args_dict
        # Prefer Deno runtime by default (better YouTube support).
        opts.setdefault("js_runtimes", {"deno": {}})
        # Allow fetching remote EJS component when needed.
        opts.setdefault("remote_components", ["ejs:github"])

    def extend_cli_cmd(self, cmd: list[str]) -> None:
        """
        Extend a yt-dlp CLI command list with extractor args.
        """
        for arg in self.extractor_args_cli:
            cmd.extend(["--extractor-args", str(arg)])

