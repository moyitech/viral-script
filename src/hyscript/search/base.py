"""Provider-independent search contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Normalized search result suitable for a generation trace."""

    rank: int
    title: str
    url: str
    snippet: str
    raw_content: str | None = None
    score: float | None = None
    published_at: str | None = None
    content_hash: str | None = None


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """Normalized provider response with metadata needed for audit and replay."""

    provider: str
    query: str
    results: tuple[SearchResult, ...]
    request_id: str | None = None
    response_time: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)


class SearchProviderError(RuntimeError):
    """Raised when a configured search provider cannot complete a request."""


class SearchProvider(Protocol):
    """Interface used by the research agent."""

    def search(self, query: str, *, limit: int = 20) -> SearchResponse:
        """Search the live Web and return normalized results and metadata."""


class AsyncSearchProvider(Protocol):
    """Asynchronous interface used by the research agent."""

    async def search(self, query: str, *, limit: int = 20) -> SearchResponse:
        """Search the live Web without blocking the event loop."""
