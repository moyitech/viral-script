"""Explicitly opted-in Tavily connectivity test that consumes API credits."""

from __future__ import annotations

import unittest

from hyscript.config import SettingsError, get_settings
from hyscript.search import AsyncTavilySearchProvider


def _live_test_enabled() -> bool:
    try:
        return get_settings().runtime.run_live_tests
    except SettingsError:
        return False


@unittest.skipUnless(
    _live_test_enabled(),
    "Set HYSCRIPT_RUN_LIVE_TESTS=1 and Hy3/Tavily settings to run live tests.",
)
class TavilyLiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_returns_at_least_one_result(self) -> None:
        async with AsyncTavilySearchProvider(get_settings().tavily) as provider:
            response = await provider.search("Tavily Search API", limit=1)

        self.assertEqual(response.provider, "tavily")
        self.assertGreaterEqual(len(response.results), 1)
        self.assertTrue(response.results[0].url.startswith(("http://", "https://")))


if __name__ == "__main__":
    unittest.main()
