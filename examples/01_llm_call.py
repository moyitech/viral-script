"""Minimal Hy3 chat-completions example using central configuration."""

import asyncio

from hyscript.config import settings
from hyscript.llm import AsyncHy3Client, ChatMessage


async def main() -> None:
    async with AsyncHy3Client(settings.hy3) as client:
        result = await client.chat(
            [ChatMessage(role="user", content="请生成三个知识型短视频选题。")]
        )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
