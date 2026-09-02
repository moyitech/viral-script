"""Creator-facing orchestration for recommendations and script generation."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from dataclasses import asdict, dataclass, replace
import logging
from pathlib import Path
from typing import Any, Callable

from hyscript.agent import (
    ResearchAgent,
    ScriptAgent,
    ScriptArtifact,
    ScriptTask,
    TopicAgent,
    TopicRecommendationBatch,
)
from hyscript.artifacts import RunTrace, build_generation_trace
from hyscript.config import Settings
from hyscript.llm import AsyncHy3Client
from hyscript.search import AsyncTavilySearchProvider
from hyscript.trends import AsyncNewsNowHotlistProvider


logger = logging.getLogger(__name__)

AsyncContextFactory = Callable[[Any], AbstractAsyncContextManager[Any]]


class CreatorGenerationError(RuntimeError):
    """Stable application error for a generation that cannot produce a draft."""


@dataclass(frozen=True, slots=True)
class GeneratedScriptRun:
    """One delivered script and its already-frozen generation trace."""

    task: ScriptTask
    script: ScriptArtifact
    trace: RunTrace
    trace_path: Path


class CreatorWorkflow:
    """Compose existing async providers and agents for creator-facing use."""

    def __init__(
        self,
        settings: Settings,
        *,
        hotlist_factory: AsyncContextFactory = AsyncNewsNowHotlistProvider,
        llm_factory: AsyncContextFactory = AsyncHy3Client,
        search_factory: AsyncContextFactory = AsyncTavilySearchProvider,
    ) -> None:
        self.settings = settings
        self._hotlist_factory = hotlist_factory
        self._llm_factory = llm_factory
        self._search_factory = search_factory

    async def recommend_topics(self, *, count: int = 20) -> TopicRecommendationBatch:
        """Fetch current public hot lists and create a bounded recommendation set."""

        logger.info("正在获取当前公开热榜")
        async with (
            self._hotlist_factory(self.settings.newsnow) as hotlists,
            self._llm_factory(self.settings.hy3) as llm,
        ):
            hotlist_batch = await hotlists.fetch_many()
            if hotlist_batch.failures:
                logger.warning(
                    "部分热榜获取失败：%d 个来源不可用",
                    len(hotlist_batch.failures),
                )
            logger.info(
                "热榜已就绪，正在生成 %d 条推荐选题",
                count,
            )
            result = await TopicAgent(
                llm,
                embeddings=llm,
                config=self.settings.topic_recommendation,
            ).recommend(hotlist_batch.snapshots, count=count)
        logger.info("推荐选题生成完成：%d 条", len(result.recommendations))
        return result

    async def generate_script(self, task: ScriptTask) -> GeneratedScriptRun:
        """Research, generate, and freeze one script before returning it."""

        script_generation_config = replace(
            self.settings.script_generation,
            final_rewrite_enabled=True,
        )
        async with (
            self._llm_factory(self.settings.hy3) as llm,
            self._search_factory(self.settings.tavily) as search,
        ):
            research = await ResearchAgent(
                llm,
                search,
                config=self.settings.research,
            ).collect_background(task)
            if research.status != "ready":
                detail = "；".join(research.errors) or "没有找到可用背景资料"
                raise CreatorGenerationError(f"调研未能完成：{detail}")
            script = await ScriptAgent(
                llm,
                config=script_generation_config,
            ).generate(task, research)

        trace = build_generation_trace(
            task,
            research,
            script,
            config={
                "research": asdict(self.settings.research),
                "script_generation": asdict(script_generation_config),
            },
        )
        trace_path = self.settings.runtime.runs_dir / f"{trace.run_id}.json"
        logger.info("[5/5] 正在保存不可变生成 trace")
        trace.write_json(trace_path)
        logger.info("生成完成，trace 已安全保存")
        return GeneratedScriptRun(
            task=task,
            script=script,
            trace=trace,
            trace_path=trace_path,
        )
