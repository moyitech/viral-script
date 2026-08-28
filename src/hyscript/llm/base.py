"""Provider-independent LLM contracts and response normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A provider-independent chat message."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Normalized response and metadata used by traces and evaluation."""

    content: str
    model: str | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    reasoning_content: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


class LLMProviderError(RuntimeError):
    """Raised when the Hy3 endpoint fails or returns an invalid payload."""


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding endpoint fails or returns invalid vectors."""


class AsyncLLMClient(Protocol):
    """Asynchronous interface consumed by agents and Web/API entrypoints."""

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        reasoning_effort: str = "no_think",
        max_tokens: int | None = None,
    ) -> str:
        """Return the assistant text without blocking the event loop."""


class AsyncEmbeddingClient(Protocol):
    """Asynchronous embedding interface consumed by semantic clustering."""

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
    ) -> tuple[tuple[float, ...], ...]:
        """Return one dense vector per input text, preserving input order."""


def _chat_response_from_payload(payload: Any) -> ChatResponse:
    """Normalize an OpenAI-compatible chat-completions payload."""

    if not isinstance(payload, dict):
        raise LLMProviderError("Hy3 returned an invalid response payload.")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LLMProviderError("Hy3 response did not contain a completion choice.")

    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise LLMProviderError("Hy3 response did not contain a message.")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LLMProviderError("Hy3 response message did not contain text.")

    request_id = payload.get("id")
    reasoning_content = message.get("reasoning_content")
    usage = payload.get("usage", {})
    return ChatResponse(
        content=content.strip(),
        model=payload.get("model") if isinstance(payload.get("model"), str) else None,
        request_id=request_id
        if isinstance(request_id, str) and request_id
        else None,
        finish_reason=choice.get("finish_reason")
        if isinstance(choice.get("finish_reason"), str)
        else None,
        reasoning_content=reasoning_content.strip()
        if isinstance(reasoning_content, str) and reasoning_content.strip()
        else None,
        usage=usage if isinstance(usage, dict) else {},
    )
