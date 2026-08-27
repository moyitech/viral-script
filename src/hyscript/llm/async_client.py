"""Asynchronous Hy3 client built on the OpenAI-compatible Python SDK."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from openai import AsyncOpenAI

from ..config import Hy3Config, Settings, get_settings
from .base import (
    ChatMessage,
    ChatResponse,
    LLMProviderError,
    _chat_response_from_payload,
)


class AsyncChatCompletionsLike(Protocol):
    """Subset of the OpenAI completions resource used by this adapter."""

    async def create(self, **kwargs: Any) -> Any:
        """Create one non-streaming chat completion."""


class AsyncChatResourceLike(Protocol):
    completions: AsyncChatCompletionsLike


class AsyncOpenAIClientLike(Protocol):
    """Injectable OpenAI client boundary for offline tests."""

    chat: AsyncChatResourceLike

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
            timeout=settings.timeout_seconds,
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
        max_tokens: int | None = None,
    ) -> str:
        """Return only assistant text."""

        response = await self.complete(
            messages,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        return response.content

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        reasoning_effort: str = "no_think",
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Return normalized text and metadata without blocking the event loop."""

        if not messages:
            raise ValueError("At least one chat message is required.")
        if reasoning_effort not in {"no_think", "low", "high"}:
            raise ValueError("reasoning_effort must be no_think, low, or high.")

        resolved_max_tokens = max_tokens or self.settings.max_tokens
        if resolved_max_tokens < 1:
            raise ValueError("max_tokens must be greater than zero.")

        try:
            completion = await self._client.chat.completions.create(
                model=self.settings.model,
                messages=[
                    {"role": message.role, "content": message.content}
                    for message in messages
                ],
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
                max_tokens=resolved_max_tokens,
                stream=False,
                extra_body={
                    "chat_template_kwargs": {
                        "reasoning_effort": reasoning_effort,
                    }
                },
            )
        except Exception:
            # SDK exceptions may contain request details. Keep provider errors
            # stable and safe for logs and user-facing traces.
            raise LLMProviderError("Hy3 request failed.") from None

        try:
            payload = completion.model_dump()
        except Exception:
            raise LLMProviderError("Hy3 returned an invalid response payload.") from None
        return _chat_response_from_payload(payload)

    async def aclose(self) -> None:
        """Close the internally-created OpenAI client."""

        if self._owns_client:
            await self._client.close()

    async def __aenter__(self) -> "AsyncHy3Client":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
