"""Independent asynchronous OpenAI-compatible embedding client."""

from __future__ import annotations

import math
from typing import Any, Protocol, Sequence

from openai import AsyncOpenAI

from ..config import EmbeddingConfig, Settings, get_settings
from .base import EmbeddingProviderError


class AsyncEmbeddingsLike(Protocol):
    async def create(self, **kwargs: Any) -> Any:
        """Create one embeddings response."""


class AsyncEmbeddingOpenAIClientLike(Protocol):
    embeddings: AsyncEmbeddingsLike

    async def close(self) -> None:
        """Close owned HTTP resources."""


class AsyncOpenAIEmbeddingClient:
    """Secret-safe embedding adapter with its own endpoint and API key."""

    def __init__(
        self,
        settings: EmbeddingConfig,
        *,
        client: AsyncEmbeddingOpenAIClientLike | None = None,
    ) -> None:
        self.settings = settings
        self._owns_client = client is None
        self._client = client or AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.openai_base_url,
            timeout=None,
            max_retries=0,
        )

    @classmethod
    def from_settings(
        cls,
        app_settings: Settings | None = None,
    ) -> "AsyncOpenAIEmbeddingClient":
        """Create a client from the central application settings."""

        return cls((app_settings or get_settings()).embedding)

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
    ) -> tuple[tuple[float, ...], ...]:
        """Embed text while preserving input order."""

        normalized_texts = tuple(text.strip() for text in texts)
        if not normalized_texts:
            raise ValueError("At least one text is required.")
        if any(not text for text in normalized_texts):
            raise ValueError("Embedding texts must not be empty.")
        requested_model = model.strip()
        if not requested_model:
            raise ValueError("Embedding model must not be empty.")

        try:
            response = await self._client.embeddings.create(
                model=requested_model,
                input=list(normalized_texts),
                encoding_format="float",
            )
        except Exception:
            raise EmbeddingProviderError("Embedding request failed.") from None

        try:
            payload = response.model_dump()
            raw_data = payload["data"]
            if (
                not isinstance(raw_data, list)
                or len(raw_data) != len(normalized_texts)
            ):
                raise TypeError

            indexed_vectors: dict[int, tuple[float, ...]] = {}
            dimension: int | None = None
            for item in raw_data:
                if not isinstance(item, dict):
                    raise TypeError
                index = item.get("index")
                raw_vector = item.get("embedding")
                if (
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or index in indexed_vectors
                    or not isinstance(raw_vector, list)
                    or not raw_vector
                ):
                    raise TypeError
                vector = tuple(float(value) for value in raw_vector)
                if any(not math.isfinite(value) for value in vector):
                    raise TypeError
                if dimension is None:
                    dimension = len(vector)
                elif len(vector) != dimension:
                    raise TypeError
                indexed_vectors[index] = vector

            if set(indexed_vectors) != set(range(len(normalized_texts))):
                raise TypeError
            return tuple(
                indexed_vectors[index] for index in range(len(normalized_texts))
            )
        except Exception:
            raise EmbeddingProviderError(
                "Embedding endpoint returned an invalid response payload."
            ) from None

    async def aclose(self) -> None:
        """Close the internally-created OpenAI client."""

        if self._owns_client:
            await self._client.close()

    async def __aenter__(self) -> "AsyncOpenAIEmbeddingClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


__all__ = ["AsyncOpenAIEmbeddingClient"]
