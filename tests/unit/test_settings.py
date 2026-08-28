"""Offline tests for the central, secret-safe configuration loader."""

from __future__ import annotations

import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from hyscript.config import (
    PROJECT_ROOT,
    SettingsError,
    get_settings,
    load_settings,
    reset_settings_cache,
    settings,
)


SETTINGS_MODULE = importlib.import_module("hyscript.config.settings")


def valid_environment(**overrides: str) -> dict[str, str]:
    values = {
        "HY3_BASE_URL": "https://hy3.example/v1",
        "HY3_API_KEY": "hy3-test-secret",
        "TAVILY_API_KEY": "tavily-test-secret",
    }
    values.update(overrides)
    return values


class SettingsTests(unittest.TestCase):
    def test_loads_explicit_mapping_with_safe_defaults(self) -> None:
        loaded = load_settings(env_file=None, environ=valid_environment())

        self.assertEqual(loaded.hy3.openai_base_url, "https://hy3.example/v1")
        self.assertEqual(loaded.hy3.model, "hy3")
        self.assertEqual(
            loaded.topic_recommendation.embedding_model,
            "kinfra-text-embedding-4b",
        )
        self.assertEqual(loaded.topic_recommendation.similarity_threshold, 0.72)
        self.assertEqual(loaded.topic_recommendation.max_generation_concurrency, 4)
        self.assertEqual(loaded.tavily.base_url, "https://api.tavily.com")
        self.assertEqual(loaded.tavily.max_results, 20)
        self.assertEqual(loaded.newsnow.base_url, "https://newsnow.busiyi.world")
        self.assertEqual(loaded.newsnow.source_ids[0], "weibo")
        self.assertEqual(loaded.hotlist_provider, "newsnow")
        self.assertEqual(loaded.runtime.log_level, "INFO")
        self.assertFalse(loaded.runtime.run_live_tests)
        self.assertNotIn("hy3-test-secret", repr(loaded))
        self.assertNotIn("tavily-test-secret", repr(loaded))

    def test_process_mapping_overrides_dotenv_file(self) -> None:
        with TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "\n".join(
                    (
                        "HY3_BASE_URL=https://file.example/v1",
                        "HY3_API_KEY=file-hy3-key",
                        "HY3_MODEL=file-model",
                        "TAVILY_API_KEY=file-tavily-key",
                        "TAVILY_MAX_RESULTS=20",
                    )
                ),
                encoding="utf-8",
            )

            loaded = load_settings(
                env_file,
                environ={"HY3_MODEL": "process-model", "TAVILY_MAX_RESULTS": "5"},
            )

        self.assertEqual(loaded.hy3.model, "process-model")
        self.assertEqual(loaded.tavily.max_results, 5)
        self.assertEqual(loaded.env_file, env_file.resolve())

    def test_explicit_empty_override_does_not_fall_back_to_dotenv(self) -> None:
        with TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "\n".join(f"{key}={value}" for key, value in valid_environment().items()),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SettingsError, "HY3_API_KEY"):
                load_settings(env_file, environ={"HY3_API_KEY": ""})

    def test_missing_required_values_raise_settings_error(self) -> None:
        with self.assertRaisesRegex(
            SettingsError,
            "HY3_BASE_URL, HY3_API_KEY, TAVILY_API_KEY",
        ):
            load_settings(env_file=None, environ={})

    def test_rejects_invalid_numeric_and_provider_values(self) -> None:
        invalid_values = {
            "HY3_TEMPERATURE": "2.1",
            "TAVILY_MAX_RESULTS": "21",
            "SEARCH_PROVIDER": "unsupported",
            "HOTLIST_PROVIDER": "unsupported",
            "NEWSNOW_SOURCE_IDS": "bad/source",
            "NEWSNOW_MAX_CONCURRENCY": "11",
            "TOPIC_EMBEDDING_MODEL": "",
            "TOPIC_SIMILARITY_THRESHOLD": "1.01",
            "TOPIC_MAX_GENERATION_CONCURRENCY": "0",
            "HYSCRIPT_RUN_LIVE_TESTS": "sometimes",
        }
        for name, value in invalid_values.items():
            with self.subTest(name=name):
                with self.assertRaises(SettingsError):
                    load_settings(
                        env_file=None,
                        environ=valid_environment(**{name: value}),
                    )

    def test_runtime_paths_and_boolean_are_normalized(self) -> None:
        loaded = load_settings(
            env_file=None,
            environ=valid_environment(
                HYSCRIPT_LOG_LEVEL="debug",
                HYSCRIPT_RUNS_DIR="eval/custom-runs",
                HYSCRIPT_RUN_LIVE_TESTS="yes",
            ),
        )

        self.assertEqual(loaded.runtime.log_level, "DEBUG")
        self.assertEqual(
            loaded.runtime.runs_dir,
            (PROJECT_ROOT / "eval/custom-runs").resolve(),
        )
        self.assertTrue(loaded.runtime.run_live_tests)

    def test_full_tavily_search_url_is_normalized_for_sdk(self) -> None:
        loaded = load_settings(
            env_file=None,
            environ=valid_environment(
                TAVILY_BASE_URL=(
                    "https://tavily.sharyuke.com/api/proxy/search"
                ),
            ),
        )

        self.assertEqual(
            loaded.tavily.sdk_base_url,
            "https://tavily.sharyuke.com/api/proxy",
        )
        self.assertEqual(
            loaded.tavily.search_url,
            "https://tavily.sharyuke.com/api/proxy/search",
        )

    def test_newsnow_settings_are_normalized(self) -> None:
        loaded = load_settings(
            env_file=None,
            environ=valid_environment(
                NEWSNOW_BASE_URL="https://newsnow.example/",
                NEWSNOW_SOURCE_IDS="weibo,baidu,weibo",
                NEWSNOW_MAX_ITEMS_PER_SOURCE="12",
                NEWSNOW_MAX_CONCURRENCY="2",
            ),
        )

        self.assertEqual(loaded.newsnow.api_url, "https://newsnow.example/api/s")
        self.assertEqual(loaded.newsnow.source_ids, ("weibo", "baidu"))
        self.assertEqual(loaded.newsnow.max_items_per_source, 12)
        self.assertEqual(loaded.newsnow.max_concurrency, 2)

    def test_topic_recommendation_settings_are_normalized(self) -> None:
        loaded = load_settings(
            env_file=None,
            environ=valid_environment(
                TOPIC_EMBEDDING_MODEL="custom-embedding",
                TOPIC_SIMILARITY_THRESHOLD="0.68",
                TOPIC_MAX_GENERATION_CONCURRENCY="3",
            ),
        )

        self.assertEqual(
            loaded.topic_recommendation.embedding_model,
            "custom-embedding",
        )
        self.assertEqual(loaded.topic_recommendation.similarity_threshold, 0.68)
        self.assertEqual(loaded.topic_recommendation.max_generation_concurrency, 3)

    def test_cached_settings_can_be_reset(self) -> None:
        first = load_settings(env_file=None, environ=valid_environment(HY3_MODEL="first"))
        second = load_settings(env_file=None, environ=valid_environment(HY3_MODEL="second"))
        reset_settings_cache()
        self.addCleanup(reset_settings_cache)

        with patch.object(SETTINGS_MODULE, "load_settings", side_effect=[first, second]) as loader:
            self.assertIs(get_settings(), first)
            self.assertIs(get_settings(), first)
            self.assertEqual(loader.call_count, 1)

            reset_settings_cache()
            self.assertIs(get_settings(), second)
            self.assertEqual(loader.call_count, 2)

    def test_lazy_proxy_repr_does_not_load_configuration(self) -> None:
        with patch.object(SETTINGS_MODULE, "get_settings") as loader:
            self.assertEqual(repr(settings), "<lazy hyscript settings>")
            loader.assert_not_called()

    def test_lazy_proxy_exposes_topic_recommendation_settings(self) -> None:
        loaded = load_settings(env_file=None, environ=valid_environment())
        with patch.object(SETTINGS_MODULE, "get_settings", return_value=loaded):
            self.assertIs(
                settings.topic_recommendation,
                loaded.topic_recommendation,
            )


if __name__ == "__main__":
    unittest.main()
