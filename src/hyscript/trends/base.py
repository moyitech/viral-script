"""Provider-independent hot-list contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class HotlistItem:
    """One ranked entry from a public hot list."""

    source_id: str
    rank: int
    item_id: str
    title: str
    url: str | None = None
    mobile_url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HotlistSnapshot:
    """A normalized, timestamped snapshot of one NewsNow source."""

    provider: str
    source_id: str
    fetched_at: str
    updated_at: str | None
    items: tuple[HotlistItem, ...]
    response_status: str = "success"


@dataclass(frozen=True, slots=True)
class HotlistFetchFailure:
    """A source-level failure retained while other sources remain usable."""

    source_id: str
    message: str


@dataclass(frozen=True, slots=True)
class HotlistBatch:
    """Successful snapshots plus non-fatal source failures."""

    provider: str
    fetched_at: str
    snapshots: tuple[HotlistSnapshot, ...]
    failures: tuple[HotlistFetchFailure, ...] = ()


class HotlistProviderError(RuntimeError):
    """Raised when a hot-list provider cannot return usable data."""


class AsyncHotlistProvider(Protocol):
    """Async boundary consumed by the discovery workflow."""

    async def fetch(self, source_id: str) -> HotlistSnapshot:
        """Fetch and normalize one current hot list."""

    async def fetch_many(
        self,
        source_ids: Sequence[str] | None = None,
    ) -> HotlistBatch:
        """Fetch multiple hot lists with bounded concurrency."""
