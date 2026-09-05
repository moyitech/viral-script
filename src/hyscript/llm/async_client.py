"""Asynchronous Hy3 client built on the OpenAI-compatible Python SDK."""

from __future__ import annotations

import asyncio
import math
from typing import Any, Protocol, Sequence

from openai import AsyncOpenAI

from ..config import Hy3Config, Settings, get_settings
from .base import (
    ChatMessage,
    ChatResponse,
    EmbeddingProviderError,
    LLMProviderError,
    _chat_response_from_payload,
)


class AsyncChatCompletionsLike(Protocol):
    """Subset of the OpenAI completions resource used by this adapter."""

    async def create(self, **kwargs: Any) -> Any:
        """Create one non-streaming chat completion."""


class AsyncChatResourceLike(Protocol):
    completions: AsyncChatCompletionsLike


class AsyncEmbeddingsLike(Protocol):
    """Subset of the OpenAI embeddings resource used by this adapter."""

    async def create(self, **kwargs: Any) -> Any:
        """Create one embeddings response."""


class AsyncOpenAIClientLike(Protocol):
    """Injectable OpenAI client boundary for offline tests."""

    chat: AsyncChatResourceLike
    embeddings: AsyncEmbeddingsLike

    async def close(self) -> None:
        """Close owned HTTP resources."""


class AsyncHy3Client:
    """Async, secret-safe Hy3 adapter using ``openai.AsyncOpenAI``."""

    def __init__(
        self,
        settings: Hy3Config,
        *,
        client: AsyncOpenAIClientLike | None = None,
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
    ) -> "AsyncHy3Client":
        """Create a client from the central application settings."""

        return cls((app_settings or get_settings()).hy3)

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        reasoning_effort: str = "no_think",
    ) -> str:
        """Return only assistant text."""

        response = await self.complete(
            messages,
            reasoning_effort=reasoning_effort,
        )
        return response.content

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        reasoning_effort: str = "no_think",
    ) -> ChatResponse:
        """Return normalized text and metadata without blocking the event loop."""

        if not messages:
            raise ValueError("At least one chat message is required.")
        if reasoning_effort not in {"no_think", "low", "high", "xhigh", "max"}:
            raise ValueError(
                "reasoning_effort must be no_think, low, high, xhigh, or max."
            )

        request: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "stream": False,
        }
        if reasoning_effort in {"xhigh", "max"}:
            # Candidate Judge endpoints use the top-level Chat Completions
            # parameter. Keep the established Hy3 chat-template shape
            # unchanged for existing effort levels.
            request["reasoning_effort"] = reasoning_effort
        else:
            request["extra_body"] = {
                "chat_template_kwargs": {
                    "reasoning_effort": reasoning_effort,
                }
            }

        rate_limit_attempt = 0
        while True:
            try:
                completion = await self._client.chat.completions.create(**request)
                break
            except Exception as exc:
                if not _is_rate_limit_error(exc):
                    # SDK exceptions may contain request details. Keep provider errors
                    # stable and safe for logs and user-facing traces.
                    raise LLMProviderError("Hy3 request failed.") from None
                rate_limit_attempt += 1
                await asyncio.sleep(_rate_limit_delay(exc, rate_limit_attempt))

        try:
            payload = completion.model_dump()
        except Exception:
            raise LLMProviderError("Hy3 returned an invalid response payload.") from None
        return _chat_response_from_payload(payload)

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
    ) -> tuple[tuple[float, ...], ...]:
        """Embed text through TokenHub while preserving input order."""

        normalized_texts = tuple(text.strip() for text in texts)
        if not normalized_texts:
            raise ValueError("At least one text is required.")
        if any(not text for text in normalized_texts):
            raise ValueError("Embedding texts must not be empty.")
        if not model.strip():
            raise ValueError("Embedding model must not be empty.")

        try:
            response = await self._client.embeddings.create(
                model=model.strip(),
                input=list(normalized_texts),
                encoding_format="float",
            )
        except Exception:
            raise EmbeddingProviderError("Embedding request failed.") from None

        try:
            payload = response.model_dump()
            raw_data = payload["data"]
            if not isinstance(raw_data, list) or len(raw_data) != len(normalized_texts):
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

            expected_indices = set(range(len(normalized_texts)))
            if set(indexed_vectors) != expected_indices:
                raise TypeError
            return tuple(indexed_vectors[index] for index in range(len(normalized_texts)))
        except Exception:
            raise EmbeddingProviderError(
                "Embedding endpoint returned an invalid response payload."
            ) from None

    async def aclose(self) -> None:
        """Close the internally-created OpenAI client."""

        if self._owns_client:
            await self._client.close()

    async def __aenter__(self) -> "AsyncHy3Client":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


def _is_rate_limit_error(exc: Exception) -> bool:
    """Recognize provider throttling without persisting exception details."""

    return getattr(exc, "status_code", None) == 429


def _rate_limit_delay(exc: Exception, attempt: int) -> float:
    """Use Retry-After when available, otherwise capped exponential backoff."""

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        raw_delay = headers.get("retry-after") or headers.get("Retry-After")
        try:
            delay = float(raw_delay)
        except (TypeError, ValueError):
            delay = 0.0
        if math.isfinite(delay) and delay > 0:
            return min(delay, 60.0)
    return min(0.5 * (2 ** min(attempt - 1, 6)), 30.0)
