"""Fetch current NewsNow lists and ask Hy3 for 20 topic recommendations."""

import asyncio
from dataclasses import asdict
import json

from hyscript.agent import TopicAgent
from hyscript.config import settings
from hyscript.llm import AsyncHy3Client
from hyscript.trends import AsyncNewsNowHotlistProvider


async def main() -> None:
    async with (
        AsyncNewsNowHotlistProvider(settings.newsnow) as hotlists,
        AsyncHy3Client(settings.hy3) as llm,
    ):
        hotlist_batch = await hotlists.fetch_many()
        recommendations = await TopicAgent(
            llm,
            embeddings=llm,
            config=settings.topic_recommendation,
        ).recommend(
            hotlist_batch.snapshots,
            count=20,
        )

    output = {
        "hotlist_failures": [asdict(item) for item in hotlist_batch.failures],
        "result": asdict(recommendations),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
