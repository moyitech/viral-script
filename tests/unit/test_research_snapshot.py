"""Offline tests for controlled replay of frozen research snapshots."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hyscript.agent import (
    Claim,
    Evidence,
    PlannedQuery,
    QueryPlan,
    ResearchOutcome,
    TitleChainPart,
)
from hyscript.artifacts import (
    ResearchSnapshotError,
    load_research_outcome,
    research_outcome_from_dict,
)
from hyscript.llm import LLMCallUsage
from hyscript.search import SearchResponse, SearchResult


def outcome() -> ResearchOutcome:
    query = PlannedQuery(query="权威材料", purpose="核实核心事实")
    return ResearchOutcome(
        status="ready",
        query_plan=QueryPlan(
            goal="核实事实和边界",
            must_verify=("事实", "边界"),
            queries=(query,),
            current_date="2026-08-31",
        ),
        search_responses=(
            SearchResponse(
                provider="fake-search",
                query=query.query,
                results=(
                    SearchResult(
                        rank=1,
                        title="权威来源",
                        url="https://authority.example/item",
                        snippet="材料摘要",
                        raw_content="材料全文说明事实和边界。",
                        score=0.9,
                        published_at="2026-08-30",
                        content_hash="a" * 64,
                    ),
                ),
                request_id="search-1",
                response_time=0.2,
                usage={"credits": 1},
            ),
        ),
        evidence=(
            Evidence(
                evidence_id="E001",
                result_ref="R001",
                title="权威来源",
                url="https://authority.example/item",
                excerpt="材料全文说明事实和边界。",
                source_query=query.query,
                published_at="2026-08-30",
                content_hash="a" * 64,
                score=0.9,
            ),
        ),
        claims=(
            Claim(
                claim_id="C001",
                text="材料支持核心事实并说明边界。",
                evidence_ids=("E001",),
                is_core=True,
                support_status="supported",
            ),
        ),
        errors=(),
        query_plan_prompt_version="query-v1",
        evidence_prompt_version="evidence-v1",
        llm_request_count=2,
        search_request_count=1,
        executed_queries=(query,),
        llm_usages=(
            LLMCallUsage(
                stage="research.query_plan",
                attempt=1,
                model="hy3",
                request_id="hy3-1",
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
                reasoning_tokens=5,
                cached_input_tokens=10,
                raw_usage={"total_tokens": 120},
            ),
        ),
        title_chain=(
            TitleChainPart(
                component="subject_scope",
                status="covered",
                claim_ids=("C001",),
                reason="核心论断覆盖标题主体和范围。",
            ),
        ),
    )


class ResearchSnapshotTests(unittest.TestCase):
    def test_restores_asdict_json_without_losing_contract_types(self) -> None:
        original = outcome()
        payload = json.loads(json.dumps(asdict(original), ensure_ascii=False))

        restored = research_outcome_from_dict(payload)

        self.assertEqual(restored, original)
        self.assertIsInstance(restored.query_plan.queries, tuple)
        self.assertIsInstance(restored.search_responses[0].results, tuple)
        self.assertIsInstance(restored.claims[0].evidence_ids, tuple)
        self.assertIsInstance(restored.llm_usages[0], LLMCallUsage)
        self.assertIsInstance(restored.title_chain[0], TitleChainPart)

    def test_loads_utf8_snapshot_file(self) -> None:
        original = outcome()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "research.json"
            path.write_text(
                json.dumps(asdict(original), ensure_ascii=False),
                encoding="utf-8",
            )

            restored = load_research_outcome(path)

        self.assertEqual(restored, original)

    def test_rejects_invalid_status_nested_shape_and_evidence_url(self) -> None:
        base = asdict(outcome())
        cases = []

        invalid_status = {**base, "status": "running"}
        cases.append(invalid_status)

        invalid_refs = json.loads(json.dumps(base))
        invalid_refs["claims"][0]["evidence_ids"] = "E001"
        cases.append(invalid_refs)

        invalid_url = json.loads(json.dumps(base))
        invalid_url["evidence"][0]["url"] = "not-a-url"
        cases.append(invalid_url)

        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ResearchSnapshotError):
                    research_outcome_from_dict(payload)


if __name__ == "__main__":
    unittest.main()
