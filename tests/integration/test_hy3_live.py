"""Explicitly opted-in Hy3 connectivity test that consumes API credits."""

from __future__ import annotations

import unittest

from hyscript.config import SettingsError, get_settings
from hyscript.llm import AsyncHy3Client, ChatMessage


def _live_test_enabled() -> bool:
    try:
        return get_settings().runtime.run_live_tests
    except SettingsError:
        return False


@unittest.skipUnless(
    _live_test_enabled(),
    "Set HYSCRIPT_RUN_LIVE_TESTS=1 and Hy3/Tavily settings to run live tests.",
)
class Hy3LiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_returns_text(self) -> None:
        async with AsyncHy3Client(get_settings().hy3) as client:
            response = await client.complete(
                [ChatMessage(role="user", content="只回复两个大写字母：OK")],
                reasoning_effort="no_think",
            )

        self.assertTrue(response.content)
        self.assertLessEqual(len(response.content), 64)


if __name__ == "__main__":
    unittest.main()
