"""Offline tests for the AsyncOpenAI-backed Hy3 client."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hyscript.config import Hy3Config
from hyscript.llm import AsyncHy3Client, ChatMessage, LLMProviderError


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


class FakeAsyncOpenAI:
    def __init__(self, completions: RecordingCompletions | None = None):
        self.completions = completions or RecordingCompletions()
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class AsyncHy3ClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.settings = Hy3Config(
            base_url="https://example.com/v1/chat/completions",
            api_key="test-secret",
            max_tokens=128,
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
        request = completions.calls[0]
        self.assertEqual(request["model"], "hy3")
        self.assertEqual(request["max_tokens"], 16)
        self.assertFalse(request["stream"])
        self.assertEqual(
            request["extra_body"],
            {"chat_template_kwargs": {"reasoning_effort": "no_think"}},
        )

    async def test_chat_returns_only_content(self) -> None:
        client = AsyncHy3Client(self.settings, client=FakeAsyncOpenAI())
        content = await client.chat([ChatMessage(role="user", content="hello")])
        self.assertEqual(content, "OK")

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
            timeout=60.0,
            max_retries=0,
        )
        self.assertTrue(sdk_client.closed)


if __name__ == "__main__":
    unittest.main()
