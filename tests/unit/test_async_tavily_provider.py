"""Offline tests for the AsyncTavilyClient-backed search provider."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from hyscript.config import TavilyConfig
from hyscript.search import AsyncTavilySearchProvider, SearchProviderError


class FakeAsyncTavilyClient:
    def __init__(self, payload: dict | None = None, error: Exception | None = None):
        self.payload = payload or {"results": []}
        self.error = error
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    async def search(self, query: str, **kwargs):
        self.calls.append((query, kwargs))
        if self.error is not None:
            raise self.error
        return self.payload

    async def close(self) -> None:
        self.closed = True


class AsyncTavilySearchProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_normalizes_async_sdk_response(self) -> None:
        payload = {
            "request_id": "async-request-123",
            "response_time": 0.25,
            "usage": {"credits": 1},
            "results": [
                {
                    "title": " Async result ",
                    "url": "https://example.com/async",
                    "content": "Async evidence",
                    "score": 0.95,
                }
            ],
        }
        sdk_client = FakeAsyncTavilyClient(payload)
        provider = AsyncTavilySearchProvider(
            TavilyConfig(api_key="test-key"),
            client=sdk_client,
        )

        response = await provider.search("  async query  ", limit=1)

        self.assertEqual(response.query, "async query")
        self.assertEqual(response.request_id, "async-request-123")
        self.assertEqual(response.results[0].title, "Async result")
        query, kwargs = sdk_client.calls[0]
        self.assertEqual(query, "async query")
        self.assertEqual(kwargs["max_results"], 1)
        self.assertTrue(kwargs["include_usage"])

    async def test_sdk_error_does_not_expose_secret(self) -> None:
        sdk_client = FakeAsyncTavilyClient(
            error=RuntimeError("request contained test-key")
        )
        provider = AsyncTavilySearchProvider(
            TavilyConfig(api_key="test-key"),
            client=sdk_client,
        )

        with self.assertRaises(SearchProviderError) as caught:
            await provider.search("query")

        self.assertEqual(str(caught.exception), "Tavily search request failed.")
        self.assertNotIn("test-key", str(caught.exception))

    async def test_search_unwraps_compatible_hub_response(self) -> None:
        official_payload = {
            "request_id": "hub-request-123",
            "response_time": 0.35,
            "usage": {"credits": 1},
            "results": [
                {
                    "title": "Hub result",
                    "url": "https://example.com/hub",
                    "content": "Hub evidence",
                    "score": 0.9,
                }
            ],
        }
        wrapped_payload = {
            "code": 200,
            "message": "success",
            "data": {
                "data": official_payload,
                "credits": 1,
                "usage": "tavily",
                "ok": True,
            },
            "timestamp": 1,
            # AsyncTavilyClient adds this default at the wrong envelope level.
            "results": [],
        }
        provider = AsyncTavilySearchProvider(
            TavilyConfig(api_key="test-key"),
            client=FakeAsyncTavilyClient(wrapped_payload),
        )

        response = await provider.search("hub query", limit=1)

        self.assertEqual(response.request_id, "hub-request-123")
        self.assertEqual(len(response.results), 1)
        self.assertEqual(response.results[0].title, "Hub result")

    async def test_owned_sdk_client_uses_hub_base_and_closes(self) -> None:
        sdk_client = FakeAsyncTavilyClient()
        settings = TavilyConfig(
            api_key="test-key",
            base_url="https://tavily.sharyuke.com/api/proxy/search",
        )
        with patch(
            "hyscript.search.provider.AsyncTavilyClient",
            return_value=sdk_client,
        ) as client_class:
            provider = AsyncTavilySearchProvider(settings)
            await provider.aclose()

        client_class.assert_called_once_with(
            api_key="test-key",
            api_base_url="https://tavily.sharyuke.com/api/proxy",
            client_source="hyscript",
        )
        self.assertTrue(sdk_client.closed)


if __name__ == "__main__":
    unittest.main()
