"""Offline tests for the AsyncOpenAI-backed Hy3 client."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hyscript.config import Hy3Config
from hyscript.llm import (
    AsyncHy3Client,
    ChatMessage,
    ChatResponse,
    EmbeddingProviderError,
    LLMProviderError,
    llm_call_usage,
    summarize_token_usage,
)


class FakeCompletion:
    def __init__(self, payload: dict):
        self.payload = payload

    def model_dump(self) -> dict:
        return self.payload


class RecordingCompletions:
    def __init__(self, payload: dict | None = None, error: Exception | None = None):
        self.payload = payload or {
            "id": "chatcmpl-async-test",
            "model": "hy3",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "OK",
                        "reasoning_content": "brief reasoning",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }
        self.error = error
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return FakeCompletion(self.payload)


class RecordingEmbeddings:
    def __init__(self, payload: dict | None = None, error: Exception | None = None):
        self.payload = payload or {
            "object": "list",
            "model": "kinfra-text-embedding-4b",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]},
            ],
        }
        self.error = error
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return FakeCompletion(self.payload)


class FakeAsyncOpenAI:
    def __init__(
        self,
        completions: RecordingCompletions | None = None,
        embeddings: RecordingEmbeddings | None = None,
    ):
        self.completions = completions or RecordingCompletions()
        self.embeddings = embeddings or RecordingEmbeddings()
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class AsyncHy3ClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.settings = Hy3Config(
            base_url="https://example.com/v1/chat/completions",
            api_key="test-secret",
        )

    async def test_complete_uses_openai_shape_and_normalizes_metadata(self) -> None:
        completions = RecordingCompletions()
        client = AsyncHy3Client(
            self.settings,
            client=FakeAsyncOpenAI(completions),
        )

        response = await client.complete(
            [ChatMessage(role="user", content="只回复 OK")],
            reasoning_effort="no_think",
            max_tokens=16,
        )

        self.assertEqual(response.content, "OK")
        self.assertEqual(response.request_id, "chatcmpl-async-test")
        self.assertEqual(response.reasoning_content, "brief reasoning")
        self.assertEqual(response.usage, {"prompt_tokens": 3, "completion_tokens": 1})
        request = completions.calls[0]
        self.assertEqual(request["model"], "hy3")
        self.assertEqual(request["max_tokens"], 16)
        self.assertFalse(request["stream"])
        self.assertEqual(
            request["extra_body"],
            {"chat_template_kwargs": {"reasoning_effort": "no_think"}},
        )

    async def test_chat_returns_only_content(self) -> None:
        completions = RecordingCompletions()
        client = AsyncHy3Client(
            self.settings,
            client=FakeAsyncOpenAI(completions),
        )
        content = await client.chat([ChatMessage(role="user", content="hello")])
        self.assertEqual(content, "OK")
        self.assertNotIn("max_tokens", completions.calls[0])

    def test_normalizes_and_aggregates_provider_reported_token_usage(self) -> None:
        first = llm_call_usage(
            ChatResponse(
                content="first",
                model="hy3",
                request_id="request-1",
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 25,
                    "total_tokens": 125,
                    "prompt_tokens_details": {"cached_tokens": 20},
                    "completion_tokens_details": {"reasoning_tokens": 8},
                },
            ),
            stage="research.query_plan",
            attempt=1,
        )
        second = llm_call_usage(
            ChatResponse(
                content="second",
                usage={"input_tokens": 50, "output_tokens": 10},
            ),
            stage="script.generation",
            attempt=1,
        )

        summary = summarize_token_usage((first, second))

        self.assertEqual(first.cached_input_tokens, 20)
        self.assertEqual(first.reasoning_tokens, 8)
        self.assertEqual(second.total_tokens, 60)
        self.assertEqual(summary.reported_call_count, 2)
        self.assertEqual(summary.input_tokens, 150)
        self.assertEqual(summary.output_tokens, 35)
        self.assertEqual(summary.total_tokens, 185)
        self.assertEqual(summary.reasoning_tokens, 8)
        self.assertEqual(summary.cached_input_tokens, 20)

    async def test_embed_uses_float_format_and_restores_input_order(self) -> None:
        embeddings = RecordingEmbeddings(
            {
                "object": "list",
                "model": "kinfra-text-embedding-4b",
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ],
            }
        )
        client = AsyncHy3Client(
            self.settings,
            client=FakeAsyncOpenAI(embeddings=embeddings),
        )

        vectors = await client.embed(
            ["第一个热点", "第二个热点"],
            model="kinfra-text-embedding-4b",
        )

        self.assertEqual(vectors, ((1.0, 0.0), (0.0, 1.0)))
        self.assertEqual(
            embeddings.calls,
            [
                {
                    "model": "kinfra-text-embedding-4b",
                    "input": ["第一个热点", "第二个热点"],
                    "encoding_format": "float",
                }
            ],
        )

    async def test_embed_rejects_invalid_count_indices_and_dimensions(self) -> None:
        invalid_payloads = {
            "count": {"data": [{"index": 0, "embedding": [1.0, 0.0]}]},
            "indices": {
                "data": [
                    {"index": 0, "embedding": [1.0, 0.0]},
                    {"index": 2, "embedding": [0.0, 1.0]},
                ]
            },
            "dimensions": {
                "data": [
                    {"index": 0, "embedding": [1.0, 0.0]},
                    {"index": 1, "embedding": [0.0, 1.0, 0.0]},
                ]
            },
        }
        for name, payload in invalid_payloads.items():
            with self.subTest(name=name):
                client = AsyncHy3Client(
                    self.settings,
                    client=FakeAsyncOpenAI(
                        embeddings=RecordingEmbeddings(payload),
                    ),
                )
                with self.assertRaisesRegex(
                    EmbeddingProviderError,
                    "invalid response",
                ):
                    await client.embed(
                        ["第一个热点", "第二个热点"],
                        model="kinfra-text-embedding-4b",
                    )

    async def test_embedding_sdk_error_does_not_expose_secret(self) -> None:
        embeddings = RecordingEmbeddings(
            error=RuntimeError("request contained test-secret")
        )
        client = AsyncHy3Client(
            self.settings,
            client=FakeAsyncOpenAI(embeddings=embeddings),
        )

        with self.assertRaises(EmbeddingProviderError) as caught:
            await client.embed(["热点"], model="kinfra-text-embedding-4b")

        self.assertEqual(str(caught.exception), "Embedding request failed.")
        self.assertNotIn("test-secret", str(caught.exception))

    async def test_sdk_error_does_not_expose_secret(self) -> None:
        completions = RecordingCompletions(
            error=RuntimeError("request contained test-secret")
        )
        client = AsyncHy3Client(
            self.settings,
            client=FakeAsyncOpenAI(completions),
        )

        with self.assertRaises(LLMProviderError) as caught:
            await client.chat([ChatMessage(role="user", content="hello")])

        self.assertEqual(str(caught.exception), "Hy3 request failed.")
        self.assertNotIn("test-secret", str(caught.exception))

    async def test_owned_openai_client_uses_normalized_base_and_closes(self) -> None:
        sdk_client = FakeAsyncOpenAI()
        with patch(
            "hyscript.llm.async_client.AsyncOpenAI",
            return_value=sdk_client,
        ) as client_class:
            client = AsyncHy3Client(self.settings)
            await client.aclose()

        client_class.assert_called_once_with(
            api_key="test-secret",
            base_url="https://example.com/v1",
            timeout=180.0,
            max_retries=0,
        )
        self.assertTrue(sdk_client.closed)


if __name__ == "__main__":
    unittest.main()
