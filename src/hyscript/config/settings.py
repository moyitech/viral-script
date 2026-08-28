"""Load and validate all application settings from the project-root ``.env``.

This is the only production module that reads environment variables.  Values
already present in the process environment take precedence over values in the
dotenv file, which makes deployment overrides predictable without mutating
``os.environ`` during local development.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import os
from pathlib import Path
import re
from typing import Final, Literal, Mapping, cast
from urllib.parse import urlparse

from dotenv import dotenv_values

SearchDepth = Literal["basic", "advanced", "fast", "ultra-fast"]
SearchTopic = Literal["general", "news", "finance"]
SearchProviderName = Literal["tavily"]
HotlistProviderName = Literal["newsnow"]

DEFAULT_NEWSNOW_SOURCE_IDS: Final[tuple[str, ...]] = (
    "weibo",
    "baidu",
    "zhihu",
    "douyin",
    "bilibili-hot-search",
    "toutiao",
    "thepaper",
)
DEFAULT_NEWSNOW_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 Chrome/140.0.0.0 Safari/537.36"
)
_SOURCE_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class SettingsError(RuntimeError):
    """Raised when application configuration is missing or invalid."""


def _find_project_root(start: Path) -> Path:
    """Find the nearest parent containing ``pyproject.toml``."""

    current = start if start.is_dir() else start.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise SettingsError("Could not locate the project root containing pyproject.toml.")


PROJECT_ROOT: Final[Path] = _find_project_root(Path(__file__).resolve())
DEFAULT_ENV_FILE: Final[Path] = PROJECT_ROOT / ".env"


@dataclass(frozen=True, slots=True)
class Hy3Config:
    """Hy3 endpoint, credentials, and sampling defaults."""

    base_url: str
    api_key: str = field(repr=False)
    model: str = "hy3"
    timeout_seconds: float = 180.0
    temperature: float = 0.9
    top_p: float = 1.0

    @property
    def openai_base_url(self) -> str:
        """Return the API base expected by the OpenAI client."""

        normalized = self.base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized.removesuffix("/chat/completions")
        if normalized.endswith("/v1"):
            return normalized
        return f"{normalized}/v1"


@dataclass(frozen=True, slots=True)
class TopicRecommendationConfig:
    """Embedding deduplication and parallel topic-generation settings."""

    embedding_model: str = "kinfra-text-embedding-4b"
    similarity_threshold: float = 0.72
    max_generation_concurrency: int = 4


@dataclass(frozen=True, slots=True)
class TavilyConfig:
    """Tavily credentials and request defaults."""

    api_key: str = field(repr=False)
    base_url: str = "https://api.tavily.com"
    search_depth: SearchDepth = "basic"
    topic: SearchTopic = "general"
    max_results: int = 20
    timeout_seconds: float = 30.0

    @property
    def sdk_base_url(self) -> str:
        """Return the base expected by the SDK, which appends ``/search``."""

        normalized = self.base_url.rstrip("/")
        if normalized.endswith("/search"):
            return normalized.removesuffix("/search")
        return normalized

    @property
    def search_url(self) -> str:
        """Return the effective search endpoint for traces and diagnostics."""

        return f"{self.sdk_base_url}/search"


@dataclass(frozen=True, slots=True)
class NewsNowConfig:
    """NewsNow endpoint and bounded hot-list request defaults."""

    base_url: str = "https://newsnow.busiyi.world"
    source_ids: tuple[str, ...] = DEFAULT_NEWSNOW_SOURCE_IDS
    max_items_per_source: int = 20
    max_concurrency: int = 4
    timeout_seconds: float = 20.0
    user_agent: str = DEFAULT_NEWSNOW_USER_AGENT

    @property
    def api_url(self) -> str:
        """Return the source-list API endpoint."""

        return f"{self.base_url.rstrip('/')}/api/s"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Provider-independent runtime settings."""

    log_level: str = "INFO"
    runs_dir: Path = PROJECT_ROOT / "eval/traces/runs"
    run_live_tests: bool = False


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated configuration snapshot used by the whole application."""

    hy3: Hy3Config
    topic_recommendation: TopicRecommendationConfig
    tavily: TavilyConfig
    newsnow: NewsNowConfig
    runtime: RuntimeConfig
    search_provider: SearchProviderName = "tavily"
    hotlist_provider: HotlistProviderName = "newsnow"
    project_root: Path = PROJECT_ROOT
    env_file: Path | None = None


def _text(values: Mapping[str, str], name: str, default: str = "") -> str:
    value = values.get(name, default)
    return value.strip()


def _required(values: Mapping[str, str], *names: str) -> None:
    missing = [name for name in names if not _text(values, name)]
    if missing:
        raise SettingsError(f"Missing configuration: {', '.join(missing)}")


def _float(values: Mapping[str, str], name: str, default: str) -> float:
    try:
        return float(_text(values, name, default))
    except ValueError:
        raise SettingsError(f"{name} must be numeric.") from None


def _integer(values: Mapping[str, str], name: str, default: str) -> int:
    try:
        return int(_text(values, name, default))
    except ValueError:
        raise SettingsError(f"{name} must be an integer.") from None


def _boolean(values: Mapping[str, str], name: str, default: str) -> bool:
    value = _text(values, name, default).lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise SettingsError(
        f"{name} must be one of: 1, 0, true, false, yes, no, on, off."
    )


def _source_ids(values: Mapping[str, str]) -> tuple[str, ...]:
    raw_value = _text(
        values,
        "NEWSNOW_SOURCE_IDS",
        ",".join(DEFAULT_NEWSNOW_SOURCE_IDS),
    )
    source_ids = tuple(dict.fromkeys(part.strip() for part in raw_value.split(",")))
    if not source_ids or any(not source_id for source_id in source_ids):
        raise SettingsError("NEWSNOW_SOURCE_IDS must contain at least one source ID.")
    if len(source_ids) > 50:
        raise SettingsError("NEWSNOW_SOURCE_IDS must contain at most 50 source IDs.")
    if any(not _SOURCE_ID_PATTERN.fullmatch(source_id) for source_id in source_ids):
        raise SettingsError("NEWSNOW_SOURCE_IDS contains an invalid source ID.")
    return source_ids


def _http_url(values: Mapping[str, str], name: str, default: str = "") -> str:
    value = _text(values, name, default).rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SettingsError(f"{name} must be an HTTP(S) URL.")
    return value


def _resolve_env_file(env_file: str | os.PathLike[str] | None) -> Path | None:
    if env_file is None:
        return None
    path = Path(env_file).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _read_values(
    env_file: Path | None,
    environ: Mapping[str, str],
) -> dict[str, str]:
    file_values: dict[str, str] = {}
    if env_file is not None:
        try:
            parsed_values = dotenv_values(
                dotenv_path=env_file,
                encoding="utf-8",
                interpolate=False,
            )
        except (OSError, UnicodeError) as exc:
            raise SettingsError(f"Could not read dotenv file: {env_file}") from exc
        file_values = {
            key: value for key, value in parsed_values.items() if value is not None
        }

    # Explicit process values win, including an explicit empty value.  This is
    # useful for detecting deployment mistakes rather than silently falling
    # back to a developer credential from .env.
    return {**file_values, **dict(environ)}


def load_settings(
    env_file: str | os.PathLike[str] | None = DEFAULT_ENV_FILE,
    *,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Build an isolated configuration snapshot.

    Passing ``environ`` is primarily useful for deterministic tests.  When it
    is omitted, the current process environment is merged over ``env_file``.
    Passing ``env_file=None`` disables dotenv loading entirely.
    """

    resolved_env_file = _resolve_env_file(env_file)
    values = _read_values(
        resolved_env_file,
        os.environ if environ is None else environ,
    )

    _required(values, "HY3_BASE_URL", "HY3_API_KEY", "TAVILY_API_KEY")

    search_provider = _text(values, "SEARCH_PROVIDER", "tavily").lower()
    if search_provider != "tavily":
        raise SettingsError("SEARCH_PROVIDER must be 'tavily'.")
    hotlist_provider = _text(values, "HOTLIST_PROVIDER", "newsnow").lower()
    if hotlist_provider != "newsnow":
        raise SettingsError("HOTLIST_PROVIDER must be 'newsnow'.")

    hy3_timeout = _float(values, "HY3_TIMEOUT_SECONDS", "180")
    hy3_temperature = _float(values, "HY3_TEMPERATURE", "0.9")
    hy3_top_p = _float(values, "HY3_TOP_P", "1.0")
    if hy3_timeout <= 0:
        raise SettingsError("HY3_TIMEOUT_SECONDS must be greater than zero.")
    if not 0 <= hy3_temperature <= 2:
        raise SettingsError("HY3_TEMPERATURE must be between 0 and 2.")
    if not 0 < hy3_top_p <= 1:
        raise SettingsError("HY3_TOP_P must be greater than 0 and at most 1.")

    topic_embedding_model = _text(
        values,
        "TOPIC_EMBEDDING_MODEL",
        "kinfra-text-embedding-4b",
    )
    topic_similarity_threshold = _float(
        values,
        "TOPIC_SIMILARITY_THRESHOLD",
        "0.72",
    )
    topic_max_generation_concurrency = _integer(
        values,
        "TOPIC_MAX_GENERATION_CONCURRENCY",
        "4",
    )
    if not topic_embedding_model:
        raise SettingsError("TOPIC_EMBEDDING_MODEL must not be empty.")
    if not 0 < topic_similarity_threshold <= 1:
        raise SettingsError(
            "TOPIC_SIMILARITY_THRESHOLD must be greater than 0 and at most 1."
        )
    if not 1 <= topic_max_generation_concurrency <= 10:
        raise SettingsError(
            "TOPIC_MAX_GENERATION_CONCURRENCY must be between 1 and 10."
        )

    search_depth = _text(values, "TAVILY_SEARCH_DEPTH", "basic")
    if search_depth not in {"basic", "advanced", "fast", "ultra-fast"}:
        raise SettingsError("TAVILY_SEARCH_DEPTH is unsupported.")
    topic = _text(values, "TAVILY_TOPIC", "general")
    if topic not in {"general", "news", "finance"}:
        raise SettingsError("TAVILY_TOPIC is unsupported.")
    tavily_max_results = _integer(values, "TAVILY_MAX_RESULTS", "20")
    tavily_timeout = _float(values, "TAVILY_TIMEOUT_SECONDS", "30")
    if not 1 <= tavily_max_results <= 20:
        raise SettingsError("TAVILY_MAX_RESULTS must be between 1 and 20.")
    if tavily_timeout <= 0:
        raise SettingsError("TAVILY_TIMEOUT_SECONDS must be greater than zero.")

    newsnow_source_ids = _source_ids(values)
    newsnow_max_items = _integer(values, "NEWSNOW_MAX_ITEMS_PER_SOURCE", "20")
    newsnow_max_concurrency = _integer(values, "NEWSNOW_MAX_CONCURRENCY", "4")
    newsnow_timeout = _float(values, "NEWSNOW_TIMEOUT_SECONDS", "20")
    newsnow_user_agent = _text(
        values,
        "NEWSNOW_USER_AGENT",
        DEFAULT_NEWSNOW_USER_AGENT,
    )
    if not 1 <= newsnow_max_items <= 100:
        raise SettingsError("NEWSNOW_MAX_ITEMS_PER_SOURCE must be between 1 and 100.")
    if not 1 <= newsnow_max_concurrency <= 10:
        raise SettingsError("NEWSNOW_MAX_CONCURRENCY must be between 1 and 10.")
    if newsnow_timeout <= 0:
        raise SettingsError("NEWSNOW_TIMEOUT_SECONDS must be greater than zero.")
    if not newsnow_user_agent:
        raise SettingsError("NEWSNOW_USER_AGENT must not be empty.")

    log_level = _text(values, "HYSCRIPT_LOG_LEVEL", "INFO").upper()
    if log_level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}:
        raise SettingsError("HYSCRIPT_LOG_LEVEL is unsupported.")
    runs_dir = Path(_text(values, "HYSCRIPT_RUNS_DIR", "eval/traces/runs")).expanduser()
    if not runs_dir.is_absolute():
        runs_dir = PROJECT_ROOT / runs_dir

    return Settings(
        hy3=Hy3Config(
            base_url=_http_url(values, "HY3_BASE_URL"),
            api_key=_text(values, "HY3_API_KEY"),
            model=_text(values, "HY3_MODEL", "hy3") or "hy3",
            timeout_seconds=hy3_timeout,
            temperature=hy3_temperature,
            top_p=hy3_top_p,
        ),
        topic_recommendation=TopicRecommendationConfig(
            embedding_model=topic_embedding_model,
            similarity_threshold=topic_similarity_threshold,
            max_generation_concurrency=topic_max_generation_concurrency,
        ),
        tavily=TavilyConfig(
            api_key=_text(values, "TAVILY_API_KEY"),
            base_url=_http_url(
                values,
                "TAVILY_BASE_URL",
                "https://api.tavily.com",
            ),
            search_depth=cast(SearchDepth, search_depth),
            topic=cast(SearchTopic, topic),
            max_results=tavily_max_results,
            timeout_seconds=tavily_timeout,
        ),
        newsnow=NewsNowConfig(
            base_url=_http_url(
                values,
                "NEWSNOW_BASE_URL",
                "https://newsnow.busiyi.world",
            ),
            source_ids=newsnow_source_ids,
            max_items_per_source=newsnow_max_items,
            max_concurrency=newsnow_max_concurrency,
            timeout_seconds=newsnow_timeout,
            user_agent=newsnow_user_agent,
        ),
        runtime=RuntimeConfig(
            log_level=log_level,
            runs_dir=runs_dir.resolve(),
            run_live_tests=_boolean(values, "HYSCRIPT_RUN_LIVE_TESTS", "0"),
        ),
        search_provider=cast(SearchProviderName, search_provider),
        hotlist_provider=cast(HotlistProviderName, hotlist_provider),
        env_file=resolved_env_file,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings snapshot, loading it on first use."""

    return load_settings()


def reset_settings_cache() -> None:
    """Clear cached settings, mainly for isolated tests."""

    get_settings.cache_clear()


def reload_settings() -> Settings:
    """Reload the default dotenv file and current process environment."""

    reset_settings_cache()
    return get_settings()


class _LazySettings:
    """Proxy that keeps importing ``hyscript.config`` free of secret I/O."""

    __slots__ = ()

    @property
    def hy3(self) -> Hy3Config:
        return get_settings().hy3

    @property
    def topic_recommendation(self) -> TopicRecommendationConfig:
        return get_settings().topic_recommendation

    @property
    def tavily(self) -> TavilyConfig:
        return get_settings().tavily

    @property
    def newsnow(self) -> NewsNowConfig:
        return get_settings().newsnow

    @property
    def runtime(self) -> RuntimeConfig:
        return get_settings().runtime

    @property
    def search_provider(self) -> SearchProviderName:
        return get_settings().search_provider

    @property
    def hotlist_provider(self) -> HotlistProviderName:
        return get_settings().hotlist_provider

    @property
    def project_root(self) -> Path:
        return get_settings().project_root

    @property
    def env_file(self) -> Path | None:
        return get_settings().env_file

    def __repr__(self) -> str:
        return "<lazy hyscript settings>"


settings: Final[_LazySettings] = _LazySettings()
