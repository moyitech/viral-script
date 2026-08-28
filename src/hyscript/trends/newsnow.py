"""Asynchronous NewsNow hot-list provider."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import re
from typing import Any, Protocol, Sequence

import httpx

from ..config import NewsNowConfig, Settings, get_settings
from .base import (
    HotlistBatch,
    HotlistFetchFailure,
    HotlistItem,
    HotlistProviderError,
    HotlistSnapshot,
)


_SOURCE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class AsyncHttpResponseLike(Protocol):
    status_code: int

    def json(self) -> Any:
        """Decode the response body."""


class AsyncHttpClientLike(Protocol):
    async def get(self, url: str, **kwargs: Any) -> AsyncHttpResponseLike:
        """Execute one GET request."""

    async def aclose(self) -> None:
        """Close owned resources."""


class AsyncNewsNowHotlistProvider:
    """Fetch current NewsNow sources without blocking the event loop."""

    def __init__(
        self,
        settings: NewsNowConfig,
        *,
        client: AsyncHttpClientLike | None = None,
    ) -> None:
        self.settings = settings
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(follow_redirects=True)

    @classmethod
    def from_settings(
        cls,
        app_settings: Settings | None = None,
    ) -> "AsyncNewsNowHotlistProvider":
        """Create a provider from central application settings."""

        return cls((app_settings or get_settings()).newsnow)

    async def fetch(self, source_id: str) -> HotlistSnapshot:
        """Fetch and normalize one NewsNow source."""

        normalized_source_id = self._normalize_source_id(source_id)
        fetched_at = self._now()
        try:
            response = await self._client.get(
                self.settings.api_url,
                params={"id": normalized_source_id},
                headers={
                    "Accept": "application/json,text/plain,*/*",
                    "Referer": f"{self.settings.base_url.rstrip('/')}/",
                    "User-Agent": self.settings.user_agent,
                },
                timeout=self.settings.timeout_seconds,
            )
        except Exception:
            raise HotlistProviderError("NewsNow request failed.") from None

        if response.status_code != 200:
            raise HotlistProviderError("NewsNow returned an unsuccessful response.")
        try:
            payload = response.json()
        except Exception:
            raise HotlistProviderError("NewsNow returned a non-JSON response.") from None
        return self._snapshot_from_payload(
            payload,
            source_id=normalized_source_id,
            fetched_at=fetched_at,
            limit=self.settings.max_items_per_source,
        )

    async def fetch_many(
        self,
        source_ids: Sequence[str] | None = None,
    ) -> HotlistBatch:
        """Fetch configured sources while preserving partial failures."""

        selected_ids = self._normalize_source_ids(
            self.settings.source_ids if source_ids is None else source_ids
        )
        semaphore = asyncio.Semaphore(self.settings.max_concurrency)

        async def fetch_one(
            source_id: str,
        ) -> tuple[HotlistSnapshot | None, HotlistFetchFailure | None]:
            async with semaphore:
                try:
                    return await self.fetch(source_id), None
                except HotlistProviderError as exc:
                    return None, HotlistFetchFailure(source_id, str(exc))

        results = await asyncio.gather(*(fetch_one(source_id) for source_id in selected_ids))
        snapshots = tuple(result[0] for result in results if result[0] is not None)
        failures = tuple(result[1] for result in results if result[1] is not None)
        if not snapshots:
            raise HotlistProviderError("NewsNow did not return any usable hot lists.")
        return HotlistBatch(
            provider="newsnow",
            fetched_at=self._now(),
            snapshots=snapshots,
            failures=failures,
        )

    @classmethod
    def _snapshot_from_payload(
        cls,
        payload: Any,
        *,
        source_id: str,
        fetched_at: str,
        limit: int,
    ) -> HotlistSnapshot:
        if not isinstance(payload, dict) or payload.get("status") not in {
            "success",
            "cache",
        }:
            raise HotlistProviderError("NewsNow returned an unsuccessful payload.")
        raw_items = payload.get("items")
        if not isinstance(raw_items, list):
            raise HotlistProviderError("NewsNow returned an invalid items payload.")

        items: list[HotlistItem] = []
        for raw_item in raw_items:
            if len(items) >= limit:
                break
            if not isinstance(raw_item, dict):
                continue
            title = cls._optional_text(raw_item.get("title"))
            if title is None:
                continue
            item_id = cls._optional_text(raw_item.get("id")) or title
            extra = raw_item.get("extra")
            items.append(
                HotlistItem(
                    source_id=source_id,
                    rank=len(items) + 1,
                    item_id=item_id,
                    title=title,
                    url=cls._optional_text(raw_item.get("url")),
                    mobile_url=cls._optional_text(raw_item.get("mobileUrl")),
                    extra=extra if isinstance(extra, dict) else {},
                )
            )
        if not items:
            raise HotlistProviderError("NewsNow returned no usable hot-list items.")

        return HotlistSnapshot(
            provider="newsnow",
            source_id=source_id,
            fetched_at=fetched_at,
            updated_at=cls._timestamp(payload.get("updatedTime")),
            items=tuple(items),
            response_status=payload["status"],
        )

    @staticmethod
    def _normalize_source_id(source_id: str) -> str:
        normalized = source_id.strip().lower()
        if not _SOURCE_ID_PATTERN.fullmatch(normalized):
            raise ValueError("NewsNow source ID is invalid.")
        return normalized

    @classmethod
    def _normalize_source_ids(cls, source_ids: Sequence[str]) -> tuple[str, ...]:
        normalized = tuple(
            dict.fromkeys(cls._normalize_source_id(source_id) for source_id in source_ids)
        )
        if not normalized:
            raise ValueError("At least one NewsNow source ID is required.")
        return normalized

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None

    @staticmethod
    def _timestamp(value: Any) -> str | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        try:
            return datetime.fromtimestamp(value / 1000, UTC).isoformat()
        except (OverflowError, OSError, ValueError):
            return None

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    async def aclose(self) -> None:
        """Close the internally-created HTTP client."""

        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "AsyncNewsNowHotlistProvider":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
