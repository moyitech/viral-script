"""Minimal Tavily search example using central configuration."""

import asyncio

from hyscript.config import settings
from hyscript.search import AsyncTavilySearchProvider


async def main() -> None:
    async with AsyncTavilySearchProvider(settings.tavily) as provider:
        response = await provider.search(
            "生成式人工智能 内容创作 最新进展",
            limit=20,
        )
    print(
        f"provider={response.provider} request_id={response.request_id} "
        f"response_time={response.response_time}"
    )
    for result in response.results:
        print(f"{result.rank}. {result.title} ({result.score})\n   {result.url}")


if __name__ == "__main__":
    asyncio.run(main())
