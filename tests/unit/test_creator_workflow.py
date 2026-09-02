"""Offline tests for the reusable creator-facing generation workflow."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from hyscript.agent import (
    PlannedQuery,
    QueryPlan,
    ResearchOutcome,
    ScriptArtifact,
    ScriptTask,
    TopicRecommendation,
    TopicRecommendationBatch,
)
from hyscript.config import load_settings
from hyscript.trends import HotlistBatch
from hyscript.workflows import CreatorGenerationError, CreatorWorkflow


def _settings(directory: str):
    loaded = load_settings(
        env_file=None,
        environ={
            "HY3_BASE_URL": "https://hy3.example/v1",
            "HY3_API_KEY": "test-key",
            "EMBEDDING_BASE_URL": "https://embedding.example/v1",
            "EMBEDDING_API_KEY": "test-embedding-key",
            "TAVILY_API_KEY": "test-search-key",
        },
    )
    runtime = replace(
        loaded.runtime,
        runs_dir=Path(directory) / "traces",
        evaluation_dir=Path(directory) / "evaluations",
    )
    return replace(loaded, runtime=runtime)


class _AsyncContext:
    def __init__(self, value) -> None:
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None


class _Hotlists:
    async def fetch_many(self) -> HotlistBatch:
        return HotlistBatch(provider="fake", fetched_at="2026-09-02T00:00:00Z", snapshots=())


def _ready_research() -> ResearchOutcome:
    return ResearchOutcome(
        status="ready",
        query_plan=QueryPlan(
            goal="补充背景",
            must_verify=("背景",),
            queries=(PlannedQuery(query="测试查询", purpose="补充背景"),),
        ),
        search_responses=(),
        evidence=(),
        claims=(),
        errors=(),
        query_plan_prompt_version="query-v1",
        evidence_prompt_version="background-v1",
        llm_request_count=1,
        search_request_count=1,
    )


def _script() -> ScriptArtifact:
    text = "这是一个用于验证桌面生成工作流的口播文案。"
    return ScriptArtifact(
        outline=("开场", "解释"),
        script_text=text,
        claim_usages=(),
        character_count=len(text),
        prompt_version="script-v1",
        generation_attempt_count=1,
    )


class CreatorWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_recommends_twenty_topics_through_existing_agent(self) -> None:
        seen_embedding_configs = []
        seen_embedding_clients = []
        embedding_client = object()
        expected = TopicRecommendationBatch(
            recommendations=(
                TopicRecommendation(
                    title="推荐选题",
                    angle="解释变化",
                    why_now="当前热榜正在讨论",
                    sources=(),
                ),
            ),
            prompt_version="topic-v1",
            input_hotspot_count=20,
            deduplicated_event_count=20,
            selected_event_count=20,
            generation_batch_count=4,
            embedding_request_count=1,
            llm_request_count=4,
            embedding_model="fake",
            similarity_threshold=0.72,
        )

        class FakeTopicAgent:
            def __init__(self, *args, **kwargs) -> None:
                seen_embedding_clients.append(kwargs["embeddings"])

            async def recommend(self, snapshots, *, count: int):
                self.test_count = count
                return expected

        with TemporaryDirectory() as directory, patch(
            "hyscript.workflows.creator.TopicAgent",
            FakeTopicAgent,
        ):
            workflow = CreatorWorkflow(
                _settings(directory),
                hotlist_factory=lambda config: _AsyncContext(_Hotlists()),
                llm_factory=lambda config: _AsyncContext(object()),
                embedding_factory=lambda config: (
                    seen_embedding_configs.append(config)
                    or _AsyncContext(embedding_client)
                ),
            )
            result = await workflow.recommend_topics(count=20)

        self.assertIs(result, expected)
        self.assertEqual(len(seen_embedding_configs), 1)
        self.assertEqual(seen_embedding_clients, [embedding_client])
        self.assertEqual(
            seen_embedding_configs[0].openai_base_url,
            "https://embedding.example/v1",
        )

    async def test_generation_freezes_trace_before_returning(self) -> None:
        research = _ready_research()
        script = _script()
        seen_script_configs = []

        class FakeResearchAgent:
            def __init__(self, *args, **kwargs) -> None:
                pass

            async def collect_background(self, task):
                return research

        class FakeScriptAgent:
            def __init__(self, *args, **kwargs) -> None:
                seen_script_configs.append(kwargs["config"])

            async def generate(self, task, outcome):
                self.asserted = outcome is research
                return script

        with TemporaryDirectory() as directory, patch(
            "hyscript.workflows.creator.ResearchAgent",
            FakeResearchAgent,
        ), patch(
            "hyscript.workflows.creator.ScriptAgent",
            FakeScriptAgent,
        ):
            workflow = CreatorWorkflow(
                _settings(directory),
                llm_factory=lambda config: _AsyncContext(object()),
                search_factory=lambda config: _AsyncContext(object()),
            )
            result = await workflow.generate_script(
                ScriptTask(topic="测试选题", target_length=100),
            )

            self.assertTrue(result.trace_path.is_file())
            self.assertTrue(seen_script_configs[0].final_rewrite_enabled)
            self.assertTrue(
                result.trace.config["script_generation"]["final_rewrite_enabled"]
            )
            frozen = result.trace_path.read_text(encoding="utf-8")
            self.assertIn(result.trace.run_id, frozen)
            self.assertIn(script.script_text, frozen)

    async def test_generation_stops_when_background_is_unavailable(self) -> None:
        research = replace(
            _ready_research(),
            status="insufficient_evidence",
            errors=("没有找到背景",),
        )

        class FakeResearchAgent:
            def __init__(self, *args, **kwargs) -> None:
                pass

            async def collect_background(self, task):
                return research

        with TemporaryDirectory() as directory, patch(
            "hyscript.workflows.creator.ResearchAgent",
            FakeResearchAgent,
        ):
            workflow = CreatorWorkflow(
                _settings(directory),
                llm_factory=lambda config: _AsyncContext(object()),
                search_factory=lambda config: _AsyncContext(object()),
            )
            with self.assertRaisesRegex(CreatorGenerationError, "没有找到背景"):
                await workflow.generate_script(
                    ScriptTask(topic="测试选题", target_length=100),
                )

            self.assertFalse(_settings(directory).runtime.runs_dir.exists())
