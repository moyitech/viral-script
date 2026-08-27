"""Tavily implementation of the live-search provider boundary."""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Protocol

from tavily import AsyncTavilyClient, TavilyClient

from ..config import Settings, TavilyConfig, get_settings
from .base import SearchProviderError, SearchResponse, SearchResult


class TavilyClientLike(Protocol):
    """Subset of the SDK client used by the adapter and offline tests."""

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Execute one Tavily search request."""


class AsyncTavilyClientLike(Protocol):
    """Subset of the async SDK client used by the adapter and tests."""

    async def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Execute one Tavily search request asynchronously."""

    async def close(self) -> None:
        """Close owned HTTP resources."""


# Compatibility name kept for callers of the initial scaffold. Configuration
# now lives exclusively in hyscript.config.settings.
TavilySettings = TavilyConfig


class TavilySearchProvider:
    """Normalize Tavily SDK responses for agents and generation traces."""

    def __init__(
        self,
        settings: TavilyConfig,
        *,
        client: TavilyClientLike | None = None,
    ) -> None:
        self.settings = settings
        self._client = client or TavilyClient(
            api_key=settings.api_key,
            api_base_url=settings.sdk_base_url,
            client_source="hyscript",
        )

    @classmethod
    def from_settings(
        cls,
        app_settings: Settings | None = None,
    ) -> "TavilySearchProvider":
        """Create a provider from the central application settings."""

        return cls((app_settings or get_settings()).tavily)

    @classmethod
    def from_env(cls) -> "TavilySearchProvider":
        """Compatibility wrapper; use :meth:`from_settings` in new code."""

        return cls.from_settings()

    def search(self, query: str, *, limit: int = 20) -> SearchResponse:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Search query must not be empty.")
        if limit < 1:
            raise ValueError("Search limit must be greater than zero.")

        effective_limit = min(limit, self.settings.max_results)
        try:
            payload = self._client.search(
                normalized_query,
                search_depth=self.settings.search_depth,
                topic=self.settings.topic,
                max_results=effective_limit,
                include_answer=False,
                include_raw_content="text",
                include_images=False,
                include_usage=True,
                timeout=self.settings.timeout_seconds,
            )
        except Exception:
            # Do not include the SDK exception text: some client errors may
            # contain request details that should not reach user-facing logs.
            raise SearchProviderError("Tavily search request failed.") from None

        return self._response_from_payload(
            payload,
            query=normalized_query,
            limit=effective_limit,
        )

    @classmethod
    def _response_from_payload(
        cls,
        payload: dict[str, Any],
        *,
        query: str,
        limit: int,
    ) -> SearchResponse:
        payload = cls._unwrap_proxy_payload(payload)
        raw_results = payload.get("results", [])
        if not isinstance(raw_results, list):
            raise SearchProviderError("Tavily returned an invalid results payload.")

        results = tuple(
            cls._normalize_result(item, rank)
            for rank, item in enumerate(raw_results[:limit], start=1)
            if isinstance(item, dict)
        )
        return SearchResponse(
            provider="tavily",
            query=query,
            results=results,
            request_id=cls._optional_text(payload.get("request_id")),
            response_time=cls._optional_float(payload.get("response_time")),
            usage=payload.get("usage", {})
            if isinstance(payload.get("usage", {}), dict)
            else {},
        )

    @staticmethod
    def _unwrap_proxy_payload(payload: dict[str, Any]) -> dict[str, Any]:
        """Accept official responses and compatible Hub response envelopes."""

        status_code = payload.get("code")
        if isinstance(status_code, int) and status_code >= 400:
            raise SearchProviderError("Tavily proxy returned an unsuccessful response.")

        envelope = payload.get("data")
        if not isinstance(envelope, dict):
            return payload
        if envelope.get("ok") is False:
            raise SearchProviderError("Tavily proxy returned an unsuccessful response.")

        nested_payload = envelope.get("data")
        if isinstance(nested_payload, dict) and isinstance(
            nested_payload.get("results"),
            list,
        ):
            return nested_payload
        if isinstance(envelope.get("results"), list):
            return envelope
        return payload

    @classmethod
    def _normalize_result(cls, item: dict[str, Any], rank: int) -> SearchResult:
        title = cls._optional_text(item.get("title")) or "Untitled"
        url = cls._optional_text(item.get("url")) or ""
        snippet = cls._optional_text(item.get("content")) or ""
        raw_content = cls._optional_text(item.get("raw_content"))
        hash_source = raw_content or snippet
        content_hash = (
            sha256(hash_source.encode("utf-8")).hexdigest() if hash_source else None
        )
        return SearchResult(
            rank=rank,
            title=title,
            url=url,
            snippet=snippet,
            raw_content=raw_content,
            score=cls._optional_float(item.get("score")),
            published_at=cls._optional_text(
                item.get("published_date") or item.get("published_at")
            ),
            content_hash=content_hash,
        )

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None


class AsyncTavilySearchProvider:
    """Async Tavily adapter backed by the installed ``AsyncTavilyClient``."""

    def __init__(
        self,
        settings: TavilyConfig,
        *,
        client: AsyncTavilyClientLike | None = None,
    ) -> None:
        self.settings = settings
        self._owns_client = client is None
        self._client = client or AsyncTavilyClient(
            api_key=settings.api_key,
            api_base_url=settings.sdk_base_url,
            client_source="hyscript",
        )

    @classmethod
    def from_settings(
        cls,
        app_settings: Settings | None = None,
    ) -> "AsyncTavilySearchProvider":
        """Create a provider from the central application settings."""

        return cls((app_settings or get_settings()).tavily)

    async def search(self, query: str, *, limit: int = 20) -> SearchResponse:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Search query must not be empty.")
        if limit < 1:
            raise ValueError("Search limit must be greater than zero.")

        effective_limit = min(limit, self.settings.max_results)
        try:
            payload = await self._client.search(
                normalized_query,
                search_depth=self.settings.search_depth,
                topic=self.settings.topic,
                max_results=effective_limit,
                include_answer=False,
                include_raw_content="text",
                include_images=False,
                include_usage=True,
                timeout=self.settings.timeout_seconds,
            )
        except Exception:
            raise SearchProviderError("Tavily search request failed.") from None

        return TavilySearchProvider._response_from_payload(
            payload,
            query=normalized_query,
            limit=effective_limit,
        )

    async def aclose(self) -> None:
        """Close the internally-created Tavily client."""

        if self._owns_client:
            await self._client.close()

    async def __aenter__(self) -> "AsyncTavilySearchProvider":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


# Compatibility aliases for the initial scaffold. New code should use the
# provider-specific names so traces and configuration stay explicit.
WebSearchSettings = TavilySettings
WebSearchProvider = TavilySearchProvider
