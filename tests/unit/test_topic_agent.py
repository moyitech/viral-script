"""Offline tests for NewsNow-backed topic recommendation generation."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

from hyscript.agent import TopicAgent, TopicGenerationError
from hyscript.config import TopicRecommendationConfig
from hyscript.llm import LLMProviderError
from hyscript.trends import HotlistItem, HotlistSnapshot


class FakeModel:
    def __init__(
        self,
        *,
        embedding_groups: list[list[str]] | None = None,
        fail_once_event: str | None = None,
        wrong_batch_count: bool = False,
    ) -> None:
        self.embedding_groups = embedding_groups or []
        self.fail_once_event = fail_once_event
        self.wrong_batch_count = wrong_batch_count
        self.embedding_calls: list[tuple[tuple[str, ...], str]] = []
        self.chat_calls: list[tuple[object, dict]] = []
        self.active_generation_calls = 0
        self.max_active_generation_calls = 0
        self._failed_events: set[str] = set()

    async def embed(self, texts, *, model):
        normalized_texts = tuple(texts)
        self.embedding_calls.append((normalized_texts, model))
        vectors = [
            [1.0 if index == dimension else 0.0 for dimension in range(len(texts))]
            for index in range(len(texts))
        ]
        for group in self.embedding_groups:
            anchor = int(group[0].removeprefix("H")) - 1
            for ref in group[1:]:
                index = int(ref.removeprefix("H")) - 1
                vectors[index] = list(vectors[anchor])
        return tuple(tuple(vector) for vector in vectors)

    async def chat(self, messages, **kwargs) -> str:
        self.chat_calls.append((messages, kwargs))
        input_payload = json.loads(messages[-1].content.rsplit("\n", 1)[1])
        self.active_generation_calls += 1
        self.max_active_generation_calls = max(
            self.max_active_generation_calls,
            self.active_generation_calls,
        )
        try:
            await asyncio.sleep(0.01)
            events = input_payload["events"]
            first_event_ref = events[0]["event_ref"]
            if (
                first_event_ref == self.fail_once_event
                and first_event_ref not in self._failed_events
            ):
                self._failed_events.add(first_event_ref)
                raise LLMProviderError("simulated provider failure")

            count = input_payload["recommendation_count"]
            if self.wrong_batch_count:
                count -= 1
            return json.dumps(
                {
                    "recommendations": [
                        {
                            "title": f"{event['event_ref']}热点影响值得关注吗？",
                            "angle": f"解释{event['event_ref']}对应热点的核心问题",
                            "why_now": "来自当前公开热榜信号",
                            "event_ref": event["event_ref"],
                            "hotspot_refs": [
                                item["ref"]
                                for item in event["hotspots"][:3]
                            ],
                        }
                        for event in events[:count]
                    ]
                },
                ensure_ascii=False,
            )
        finally:
            self.active_generation_calls -= 1


def snapshots() -> tuple[HotlistSnapshot, ...]:
    return tuple(
        HotlistSnapshot(
            provider="newsnow",
            source_id=source_id,
            fetched_at="2026-08-28T00:00:00+00:00",
            updated_at="2026-08-28T00:00:00+00:00",
            items=tuple(
                HotlistItem(
                    source_id=source_id,
                    rank=index,
                    item_id=f"{source_id}-{index}",
                    title=f"{source_id} 热点 {index}",
                    url=f"https://example.com/{source_id}/{index}",
                )
                for index in range(1, 13)
            ),
        )
        for source_id in ("weibo", "baidu")
    )


class TopicAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_embeds_once_then_generates_four_batches_concurrently(
        self,
    ) -> None:
        model = FakeModel()
        agent = TopicAgent(model, embeddings=model)

        result = await agent.recommend(snapshots())

        self.assertEqual(len(result.recommendations), 20)
        self.assertEqual(result.input_hotspot_count, 24)
        self.assertEqual(result.deduplicated_event_count, 24)
        self.assertEqual(result.selected_event_count, 24)
        self.assertEqual(result.generation_batch_count, 4)
        self.assertEqual(result.embedding_request_count, 1)
        self.assertEqual(result.llm_request_count, 4)
        self.assertEqual(result.embedding_model, "kinfra-text-embedding-4b")
        self.assertEqual(result.similarity_threshold, 0.72)
        self.assertEqual(result.recommendations[0].sources[0].source_id, "weibo")
        self.assertEqual(result.recommendations[1].sources[0].source_id, "baidu")
        self.assertEqual(result.prompt_version, "topic-recommendations-3.0.0")
        self.assertEqual(len(model.embedding_calls), 1)
        texts, embedding_model = model.embedding_calls[0]
        self.assertEqual(len(texts), 24)
        self.assertEqual(texts[0], "weibo 热点 1")
        self.assertEqual(texts[1], "baidu 热点 1")
        self.assertEqual(embedding_model, "kinfra-text-embedding-4b")
        self.assertEqual(len(model.chat_calls), 4)
        self.assertTrue(
            all(
                kwargs == {"reasoning_effort": "high"}
                for _, kwargs in model.chat_calls
            )
        )
        self.assertEqual(model.max_active_generation_calls, 4)

        for messages, _ in model.chat_calls:
            self.assertIn("不可信外部数据", messages[0].content)
            self.assertIn("高考志愿：父母的建议还是自己的梦想？", messages[0].content)
            self.assertIn("主谓宾和动宾搭配准确", messages[0].content)
            self.assertIn("至少 4 个标题使用明确疑问句", messages[1].content)
            self.assertIn("不能添加输入里没有的数字", messages[1].content)
            self.assertIn('"recommendation_count": 5', messages[1].content)

    async def test_embedding_cluster_preserves_all_source_references(self) -> None:
        model = FakeModel(embedding_groups=[["H001", "H002"]])

        result = await TopicAgent(model, embeddings=model).recommend(snapshots())

        self.assertEqual(result.deduplicated_event_count, 23)
        self.assertEqual(
            tuple(source.source_id for source in result.recommendations[0].sources),
            ("weibo", "baidu"),
        )

    def test_connected_components_apply_threshold_transitively(self) -> None:
        prepared = TopicAgent._collect_hotspots(snapshots())[:3]
        angle_40 = math.radians(40)
        angle_80 = math.radians(80)
        vectors = (
            (1.0, 0.0),
            (math.cos(angle_40), math.sin(angle_40)),
            (math.cos(angle_80), math.sin(angle_80)),
        )

        clusters = TopicAgent._cluster_by_cosine_similarity(
            prepared,
            vectors,
            threshold=0.72,
        )

        self.assertLess(sum(a * c for a, c in zip(vectors[0], vectors[2])), 0.72)
        self.assertEqual(
            tuple(item.ref for item in clusters[0]),
            ("H001", "H002", "H003"),
        )

    async def test_retries_only_a_failed_generation_batch_once(self) -> None:
        model = FakeModel(fail_once_event="E001")

        result = await TopicAgent(model, embeddings=model).recommend(snapshots())

        self.assertEqual(len(result.recommendations), 20)
        self.assertEqual(len(model.embedding_calls), 1)
        self.assertEqual(len(model.chat_calls), 5)
        self.assertEqual(result.embedding_request_count, 1)
        self.assertEqual(result.llm_request_count, 5)
        self.assertTrue(
            all(
                kwargs == {"reasoning_effort": "high"}
                for _, kwargs in model.chat_calls
            )
        )

    async def test_rejects_batch_with_wrong_count_after_one_retry(self) -> None:
        model = FakeModel(wrong_batch_count=True)
        agent = TopicAgent(model, embeddings=model)

        with self.assertRaisesRegex(
            TopicGenerationError,
            "failed after one retry",
        ):
            await agent.recommend(snapshots())

    async def test_requires_enough_raw_hotspots_before_embedding(self) -> None:
        one_snapshot = snapshots()[0]
        model = FakeModel()
        agent = TopicAgent(model, embeddings=model)

        with self.assertRaisesRegex(TopicGenerationError, "Not enough"):
            await agent.recommend((one_snapshot,))
        self.assertEqual(model.embedding_calls, [])
        self.assertEqual(model.chat_calls, [])

    async def test_requires_twenty_distinct_events_after_embedding(self) -> None:
        all_refs = [f"H{index:03d}" for index in range(1, 25)]
        model = FakeModel(embedding_groups=[all_refs])

        with self.assertRaisesRegex(TopicGenerationError, "distinct events"):
            await TopicAgent(model, embeddings=model).recommend(snapshots())
        self.assertEqual(len(model.embedding_calls), 1)
        self.assertEqual(model.chat_calls, [])

    async def test_rejects_embedding_count_mismatch(self) -> None:
        class ShortEmbeddingModel(FakeModel):
            async def embed(self, texts, *, model):
                vectors = await super().embed(texts, model=model)
                return vectors[:-1]

        model = ShortEmbeddingModel()
        with self.assertRaisesRegex(TopicGenerationError, "count"):
            await TopicAgent(model, embeddings=model).recommend(snapshots())

    def test_rejects_invalid_topic_configuration(self) -> None:
        invalid_configs = (
            TopicRecommendationConfig(embedding_model=""),
            TopicRecommendationConfig(similarity_threshold=0.0),
            TopicRecommendationConfig(max_generation_concurrency=0),
        )
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    TopicAgent(FakeModel(), embeddings=FakeModel(), config=config)


class TopicAgentEntrypointTests(unittest.TestCase):
    def test_source_file_can_be_executed_directly(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [
                sys.executable,
                str(project_root / "src/hyscript/agent/topic_agent.py"),
                "--help",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("NewsNow", result.stdout)
        self.assertIn("20", result.stdout)


if __name__ == "__main__":
    unittest.main()
