"""Offline tests for the asynchronous NewsNow hot-list provider."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from hyscript.config import NewsNowConfig
from hyscript.trends import AsyncNewsNowHotlistProvider, HotlistProviderError


class FakeResponse:
    def __init__(self, payload: object, *, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def json(self) -> object:
        return self.payload


class FakeAsyncHttpClient:
    def __init__(self, responses: dict[str, FakeResponse | Exception]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    async def get(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        source_id = kwargs["params"]["id"]
        response = self.responses[source_id]
        if isinstance(response, Exception):
            raise response
        return response

    async def aclose(self) -> None:
        self.closed = True


def successful_payload(
    source_id: str,
    *,
    count: int = 3,
    status: str = "success",
) -> dict:
    return {
        "status": status,
        "id": source_id,
        "updatedTime": 1_787_897_918_864,
        "items": [
            {
                "id": f"item-{index}",
                "title": f"热点 {index}",
                "url": f"https://example.com/{index}",
                "mobileUrl": f"https://m.example.com/{index}",
                "extra": {"heat": index},
            }
            for index in range(1, count + 1)
        ],
    }


class AsyncNewsNowHotlistProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_normalizes_items_and_browser_headers(self) -> None:
        client = FakeAsyncHttpClient(
            {
                "weibo": FakeResponse(
                    successful_payload("weibo", count=3, status="cache")
                )
            }
        )
        provider = AsyncNewsNowHotlistProvider(
            NewsNowConfig(max_items_per_source=2),
            client=client,
        )

        snapshot = await provider.fetch(" weibo ")

        self.assertEqual(snapshot.provider, "newsnow")
        self.assertEqual(snapshot.source_id, "weibo")
        self.assertEqual(snapshot.response_status, "cache")
        self.assertEqual(len(snapshot.items), 2)
        self.assertEqual(snapshot.items[0].rank, 1)
        self.assertEqual(snapshot.items[0].item_id, "item-1")
        self.assertEqual(snapshot.items[0].extra, {"heat": 1})
        url, kwargs = client.calls[0]
        self.assertEqual(url, "https://newsnow.busiyi.world/api/s")
        self.assertEqual(kwargs["params"], {"id": "weibo"})
        self.assertIn("Mozilla/5.0", kwargs["headers"]["User-Agent"])
        self.assertEqual(
            kwargs["headers"]["Referer"],
            "https://newsnow.busiyi.world/",
        )

    async def test_fetch_many_preserves_partial_failures(self) -> None:
        client = FakeAsyncHttpClient(
            {
                "weibo": FakeResponse(successful_payload("weibo")),
                "baidu": FakeResponse({}, status_code=503),
            }
        )
        provider = AsyncNewsNowHotlistProvider(
            NewsNowConfig(source_ids=("weibo", "baidu")),
            client=client,
        )

        batch = await provider.fetch_many()

        self.assertEqual([item.source_id for item in batch.snapshots], ["weibo"])
        self.assertEqual([item.source_id for item in batch.failures], ["baidu"])
        self.assertEqual(
            batch.failures[0].message,
            "NewsNow returned an unsuccessful response.",
        )

    async def test_fetch_many_rejects_an_all_source_failure(self) -> None:
        client = FakeAsyncHttpClient(
            {"weibo": RuntimeError("response contained a private detail")}
        )
        provider = AsyncNewsNowHotlistProvider(
            NewsNowConfig(source_ids=("weibo",)),
            client=client,
        )

        with self.assertRaises(HotlistProviderError) as caught:
            await provider.fetch_many()

        self.assertEqual(
            str(caught.exception),
            "NewsNow did not return any usable hot lists.",
        )
        self.assertNotIn("private detail", str(caught.exception))

    async def test_owned_http_client_is_closed(self) -> None:
        client = FakeAsyncHttpClient({})
        with patch(
            "hyscript.trends.newsnow.httpx.AsyncClient",
            return_value=client,
        ) as client_class:
            provider = AsyncNewsNowHotlistProvider(NewsNowConfig())
            await provider.aclose()

        client_class.assert_called_once_with(follow_redirects=True)
        self.assertTrue(client.closed)


if __name__ == "__main__":
    unittest.main()
