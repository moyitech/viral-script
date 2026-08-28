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


@dataclass(frozen=True, slots=True)
class LLMCallUsage:
    """One provider-reported LLM usage record linked to a generation stage."""

    stage: str
    attempt: int
    model: str | None
    request_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    reasoning_tokens: int | None
    cached_input_tokens: int | None
    raw_usage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenUsageSummary:
    """Aggregate token totals across provider-reported LLM calls."""

    reported_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0


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

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        reasoning_effort: str = "no_think",
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Return assistant text plus provider metadata and token usage."""


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


def llm_call_usage(
    response: ChatResponse,
    *,
    stage: str,
    attempt: int,
) -> LLMCallUsage:
    """Normalize common OpenAI-compatible token fields without estimating usage."""

    usage = dict(response.usage)
    input_tokens = _usage_integer(usage, "prompt_tokens", "input_tokens")
    output_tokens = _usage_integer(usage, "completion_tokens", "output_tokens")
    total_tokens = _usage_integer(usage, "total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    prompt_details = usage.get("prompt_tokens_details")
    if not isinstance(prompt_details, dict):
        prompt_details = usage.get("input_tokens_details")
    completion_details = usage.get("completion_tokens_details")
    if not isinstance(completion_details, dict):
        completion_details = usage.get("output_tokens_details")
    cached_input_tokens = _mapping_integer(prompt_details, "cached_tokens")
    reasoning_tokens = _mapping_integer(completion_details, "reasoning_tokens")
    return LLMCallUsage(
        stage=stage,
        attempt=attempt,
        model=response.model,
        request_id=response.request_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_input_tokens=cached_input_tokens,
        raw_usage=usage,
    )


def summarize_token_usage(calls: Sequence[LLMCallUsage]) -> TokenUsageSummary:
    """Sum normalized fields while retaining how many calls reported tokens."""

    return TokenUsageSummary(
        reported_call_count=sum(
            1
            for call in calls
            if call.input_tokens is not None
            or call.output_tokens is not None
            or call.total_tokens is not None
        ),
        input_tokens=sum(call.input_tokens or 0 for call in calls),
        output_tokens=sum(call.output_tokens or 0 for call in calls),
        total_tokens=sum(call.total_tokens or 0 for call in calls),
        reasoning_tokens=sum(call.reasoning_tokens or 0 for call in calls),
        cached_input_tokens=sum(call.cached_input_tokens or 0 for call in calls),
    )


def _usage_integer(usage: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _mapping_integer(value: Any, key: str) -> int | None:
    if not isinstance(value, dict):
        return None
    item = value.get(key)
    if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
        return item
    return None
