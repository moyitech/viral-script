"""Minimal Tavily search example using central configuration."""

import asyncio

from hyscript.config import settings
from hyscript.search import AsyncTavilySearchProvider


def preview(text: str | None, *, limit: int) -> str:
    """Keep the demo readable while showing that full content is available."""

    if not text:
        return "<empty>"
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[:limit]}…"


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
        print(
            f"{result.rank}. {result.title} ({result.score})\n"
            f"   url: {result.url}\n"
            f"   content: {preview(result.snippet, limit=300)}\n"
            f"   raw_content: {preview(result.raw_content, limit=800)}"
        )


if __name__ == "__main__":
    asyncio.run(main())
