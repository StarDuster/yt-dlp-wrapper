"""Tests for yt_dlp_wrapper/core/context.py"""

import unittest
from unittest.mock import patch

from yt_dlp_wrapper.core.context import (
    YtDlpContext,
    build_extractor_args_cli,
    build_extractor_args_dict,
    merge_extractor_args_dict,
    select_invidious_instance,
    _coerce_str_list,
    _get_player_clients_from_env_or_config,
)


class TestCoerceStrList(unittest.TestCase):
    def test_none(self) -> None:
        self.assertEqual(_coerce_str_list(None), [])

    def test_string(self) -> None:
        self.assertEqual(_coerce_str_list("foo"), ["foo"])
        self.assertEqual(_coerce_str_list("  bar  "), ["bar"])
        self.assertEqual(_coerce_str_list(""), [])
        self.assertEqual(_coerce_str_list("   "), [])

    def test_list(self) -> None:
        self.assertEqual(_coerce_str_list(["a", "b"]), ["a", "b"])
        self.assertEqual(_coerce_str_list(["a", "", "b"]), ["a", "b"])

    def test_tuple(self) -> None:
        self.assertEqual(_coerce_str_list(("x", "y")), ["x", "y"])


class TestSelectInvidiousInstance(unittest.TestCase):
    def test_no_invidious_flag(self) -> None:
        self.assertIsNone(select_invidious_instance(no_invidious=True))

    @patch("yt_dlp_wrapper.core.context.config")
    def test_from_config(self, mock_config) -> None:
        mock_config.INVIDIOUS_INSTANCE = "invidious.example.com"
        self.assertEqual(select_invidious_instance(no_invidious=False), "invidious.example.com")

    @patch("yt_dlp_wrapper.core.context.config")
    def test_no_config(self, mock_config) -> None:
        mock_config.INVIDIOUS_INSTANCE = None
        self.assertIsNone(select_invidious_instance(no_invidious=False))


class TestGetPlayerClientsFromEnvOrConfig(unittest.TestCase):
    @patch.dict("os.environ", {"YTDLP_WRAPPER_PLAYER_CLIENT": "tv,web"}, clear=False)
    def test_from_env(self) -> None:
        result = _get_player_clients_from_env_or_config()
        self.assertEqual(result, ["tv", "web"])

    @patch.dict("os.environ", {}, clear=False)
    @patch("yt_dlp_wrapper.core.context.config")
    def test_from_config_string(self, mock_config) -> None:
        mock_config.YOUTUBE_PLAYER_CLIENT = "ios,-web_safari"
        result = _get_player_clients_from_env_or_config()
        self.assertEqual(result, ["ios", "-web_safari"])

    @patch.dict("os.environ", {}, clear=False)
    @patch("yt_dlp_wrapper.core.context.config")
    def test_from_config_list(self, mock_config) -> None:
        mock_config.YOUTUBE_PLAYER_CLIENT = ["android", "web"]
        result = _get_player_clients_from_env_or_config()
        self.assertEqual(result, ["android", "web"])

    @patch.dict("os.environ", {}, clear=False)
    @patch("yt_dlp_wrapper.core.context.config")
    def test_empty(self, mock_config) -> None:
        mock_config.YOUTUBE_PLAYER_CLIENT = None
        result = _get_player_clients_from_env_or_config()
        self.assertEqual(result, [])


class TestMergeExtractorArgsDict(unittest.TestCase):
    def test_merge_single_string(self) -> None:
        args: dict[str, dict[str, list[str]]] = {}
        merge_extractor_args_dict(args, "youtube:po_token=abc;player_client=web")
        self.assertEqual(args["youtube"]["po_token"], ["abc"])
        self.assertEqual(args["youtube"]["player_client"], ["web"])

    def test_merge_list(self) -> None:
        args: dict[str, dict[str, list[str]]] = {}
        merge_extractor_args_dict(args, ["youtube:po_token=aaa", "youtubetab:approximate_date=true"])
        self.assertEqual(args["youtube"]["po_token"], ["aaa"])
        self.assertEqual(args["youtubetab"]["approximate_date"], ["true"])

    def test_merge_appends(self) -> None:
        args: dict[str, dict[str, list[str]]] = {"youtube": {"po_token": ["existing"]}}
        merge_extractor_args_dict(args, "youtube:po_token=new")
        self.assertEqual(args["youtube"]["po_token"], ["existing", "new"])

    def test_ignore_invalid(self) -> None:
        args: dict[str, dict[str, list[str]]] = {}
        merge_extractor_args_dict(args, "invalid_no_colon")
        merge_extractor_args_dict(args, "youtube:no_equals")
        self.assertEqual(args, {})


class TestBuildExtractorArgsDict(unittest.TestCase):
    @patch("yt_dlp_wrapper.core.context.config")
    @patch("yt_dlp_wrapper.core.context._get_player_clients_from_env_or_config")
    def test_basic(self, mock_get_pc, mock_config) -> None:
        mock_get_pc.return_value = []
        mock_config.INVIDIOUS_INSTANCE = None
        mock_config.YOUTUBE_PO_TOKEN = None
        mock_config.YOUTUBE_POT_PROVIDER = None
        mock_config.YOUTUBE_EXTRACTOR_ARGS = None

        result = build_extractor_args_dict(no_invidious=True)
        self.assertIn("youtubetab", result)
        self.assertEqual(result["youtubetab"]["approximate_date"], ["true"])

    @patch("yt_dlp_wrapper.core.context.config")
    @patch("yt_dlp_wrapper.core.context._get_player_clients_from_env_or_config")
    def test_with_player_client(self, mock_get_pc, mock_config) -> None:
        mock_get_pc.return_value = ["tv", "web"]
        mock_config.INVIDIOUS_INSTANCE = None
        mock_config.YOUTUBE_PO_TOKEN = None
        mock_config.YOUTUBE_POT_PROVIDER = None
        mock_config.YOUTUBE_EXTRACTOR_ARGS = None

        result = build_extractor_args_dict(no_invidious=True)
        self.assertEqual(result["youtube"]["player_client"], ["tv", "web"])

    @patch("yt_dlp_wrapper.core.context.config")
    @patch("yt_dlp_wrapper.core.context._get_player_clients_from_env_or_config")
    def test_with_invidious(self, mock_get_pc, mock_config) -> None:
        mock_get_pc.return_value = []
        mock_config.INVIDIOUS_INSTANCE = "invidious.example.com"
        mock_config.YOUTUBE_PO_TOKEN = None
        mock_config.YOUTUBE_POT_PROVIDER = None
        mock_config.YOUTUBE_EXTRACTOR_ARGS = None

        result = build_extractor_args_dict(no_invidious=False)
        self.assertEqual(result["youtube"]["invidious_instance"], ["https://invidious.example.com"])


class TestBuildExtractorArgsCli(unittest.TestCase):
    @patch("yt_dlp_wrapper.core.context.config")
    @patch("yt_dlp_wrapper.core.context._get_player_clients_from_env_or_config")
    def test_basic(self, mock_get_pc, mock_config) -> None:
        mock_get_pc.return_value = []
        mock_config.INVIDIOUS_INSTANCE = None
        mock_config.YOUTUBE_PO_TOKEN = None
        mock_config.YOUTUBE_POT_PROVIDER = None
        mock_config.YOUTUBE_EXTRACTOR_ARGS = None

        result = build_extractor_args_cli(no_invidious=True)
        self.assertIn("youtubetab:approximate_date=true", result)

    @patch("yt_dlp_wrapper.core.context.config")
    @patch("yt_dlp_wrapper.core.context._get_player_clients_from_env_or_config")
    def test_with_po_token(self, mock_get_pc, mock_config) -> None:
        mock_get_pc.return_value = []
        mock_config.INVIDIOUS_INSTANCE = None
        mock_config.YOUTUBE_PO_TOKEN = "my_token"
        mock_config.YOUTUBE_POT_PROVIDER = "my_provider"
        mock_config.YOUTUBE_EXTRACTOR_ARGS = None

        result = build_extractor_args_cli(no_invidious=True)
        youtube_arg = [a for a in result if a.startswith("youtube:")]
        self.assertTrue(len(youtube_arg) > 0)
        self.assertIn("po_token=my_token", youtube_arg[0])
        self.assertIn("pot_provider=my_provider", youtube_arg[0])


class TestYtDlpContext(unittest.TestCase):
    @patch("yt_dlp_wrapper.core.context.config")
    @patch("yt_dlp_wrapper.core.context._get_player_clients_from_env_or_config")
    def test_properties(self, mock_get_pc, mock_config) -> None:
        mock_get_pc.return_value = []
        mock_config.INVIDIOUS_INSTANCE = "inv.example.com"
        mock_config.YOUTUBE_PO_TOKEN = None
        mock_config.YOUTUBE_POT_PROVIDER = None
        mock_config.YOUTUBE_EXTRACTOR_ARGS = None

        ctx = YtDlpContext(no_invidious=False)
        self.assertEqual(ctx.invidious_instance, "inv.example.com")
        self.assertIn("youtubetab", ctx.extractor_args_dict)
        self.assertTrue(any("youtubetab" in a for a in ctx.extractor_args_cli))

    @patch("yt_dlp_wrapper.core.context.config")
    @patch("yt_dlp_wrapper.core.context._get_player_clients_from_env_or_config")
    def test_apply_common_python_api_opts(self, mock_get_pc, mock_config) -> None:
        mock_get_pc.return_value = []
        mock_config.INVIDIOUS_INSTANCE = None
        mock_config.YOUTUBE_PO_TOKEN = None
        mock_config.YOUTUBE_POT_PROVIDER = None
        mock_config.YOUTUBE_EXTRACTOR_ARGS = None

        ctx = YtDlpContext(no_invidious=True)
        opts: dict = {}
        ctx.apply_common_python_api_opts(opts)

        self.assertIn("extractor_args", opts)
        self.assertIn("js_runtimes", opts)
        self.assertIn("remote_components", opts)

    @patch("yt_dlp_wrapper.core.context.config")
    @patch("yt_dlp_wrapper.core.context._get_player_clients_from_env_or_config")
    def test_extend_cli_cmd(self, mock_get_pc, mock_config) -> None:
        mock_get_pc.return_value = []
        mock_config.INVIDIOUS_INSTANCE = None
        mock_config.YOUTUBE_PO_TOKEN = None
        mock_config.YOUTUBE_POT_PROVIDER = None
        mock_config.YOUTUBE_EXTRACTOR_ARGS = None

        ctx = YtDlpContext(no_invidious=True)
        cmd: list[str] = ["yt-dlp"]
        ctx.extend_cli_cmd(cmd)

        self.assertIn("--extractor-args", cmd)


if __name__ == "__main__":
    unittest.main()
