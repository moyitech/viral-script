"""Offline tests for Tavily response normalization."""

from __future__ import annotations

from hashlib import sha256
import unittest
from unittest.mock import patch

from hyscript.config import TavilyConfig
from hyscript.search import (
    SearchProviderError,
    TavilySearchProvider,
)


class FakeTavilyClient:
    def __init__(self, payload: dict | None = None, error: Exception | None = None):
        self.payload = payload or {"results": []}
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    def search(self, query: str, **kwargs):
        self.calls.append((query, kwargs))
        if self.error is not None:
            raise self.error
        return self.payload


class TavilySearchProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = TavilyConfig(api_key="test-key")

    def test_search_normalizes_results_and_metadata(self) -> None:
        payload = {
            "request_id": "request-123",
            "response_time": 0.42,
            "usage": {"credits": 1},
            "results": [
                {
                    "title": " Example title ",
                    "url": "https://example.com/article",
                    "content": "Evidence snippet",
                    "score": 0.91,
                    "published_date": "2026-08-26",
                }
            ],
        }
        client = FakeTavilyClient(payload)
        provider = TavilySearchProvider(self.settings, client=client)

        response = provider.search("  example query  ", limit=20)

        self.assertEqual(response.provider, "tavily")
        self.assertEqual(response.query, "example query")
        self.assertEqual(response.request_id, "request-123")
        self.assertEqual(response.usage, {"credits": 1})
        self.assertEqual(len(response.results), 1)
        result = response.results[0]
        self.assertEqual(result.rank, 1)
        self.assertEqual(result.title, "Example title")
        self.assertEqual(result.score, 0.91)
        self.assertEqual(
            result.content_hash,
            sha256(b"Evidence snippet").hexdigest(),
        )

        query, kwargs = client.calls[0]
        self.assertEqual(query, "example query")
        self.assertEqual(kwargs["search_depth"], "basic")
        self.assertEqual(kwargs["max_results"], 20)
        self.assertTrue(kwargs["include_usage"])
        self.assertFalse(kwargs["include_raw_content"])

    def test_search_caps_limit_at_configured_maximum(self) -> None:
        settings = TavilyConfig(api_key="test-key", max_results=5)
        client = FakeTavilyClient()
        provider = TavilySearchProvider(settings, client=client)

        provider.search("query", limit=20)

        self.assertEqual(client.calls[0][1]["max_results"], 5)

    def test_full_search_url_is_normalized_before_sdk_construction(self) -> None:
        settings = TavilyConfig(
            api_key="test-key",
            base_url="https://tavily.sharyuke.com/api/proxy/search",
        )

        with patch("hyscript.search.provider.TavilyClient") as client_class:
            TavilySearchProvider(settings)

        client_class.assert_called_once_with(
            api_key="test-key",
            api_base_url="https://tavily.sharyuke.com/api/proxy",
            client_source="hyscript",
        )

    def test_search_rejects_empty_query(self) -> None:
        provider = TavilySearchProvider(self.settings, client=FakeTavilyClient())
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            provider.search("   ")

    def test_search_unwraps_compatible_hub_response(self) -> None:
        wrapped_payload = {
            "code": 200,
            "data": {
                "data": {
                    "request_id": "sync-hub-request",
                    "results": [
                        {
                            "title": "Sync Hub result",
                            "url": "https://example.com/sync-hub",
                            "content": "Evidence",
                        }
                    ],
                },
                "ok": True,
            },
            "results": [],
        }
        provider = TavilySearchProvider(
            self.settings,
            client=FakeTavilyClient(wrapped_payload),
        )

        response = provider.search("hub query", limit=1)

        self.assertEqual(response.request_id, "sync-hub-request")
        self.assertEqual(response.results[0].title, "Sync Hub result")

    def test_provider_error_does_not_expose_sdk_message(self) -> None:
        client = FakeTavilyClient(error=RuntimeError("request contained test-key"))
        provider = TavilySearchProvider(self.settings, client=client)

        with self.assertRaises(SearchProviderError) as caught:
            provider.search("query")

        self.assertEqual(str(caught.exception), "Tavily search request failed.")
        self.assertNotIn("test-key", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
