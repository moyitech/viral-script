"""Tests for adapting generation contracts into immutable run traces."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hyscript.agent import (
    Claim,
    ClaimUsage,
    Evidence,
    PlannedQuery,
    QueryPlan,
    ResearchOutcome,
    ScriptArtifact,
    ScriptTask,
)
from hyscript.artifacts import build_generation_trace
from hyscript.evaluation.io import load_frozen_trace
from hyscript.llm import LLMCallUsage
from hyscript.search import SearchResponse, SearchResult


class GenerationTraceTests(unittest.TestCase):
    def test_preserves_generation_data_and_is_accepted_by_offline_loader(self) -> None:
        task = ScriptTask(topic="已选中的热点", target_length=50)
        search_response = SearchResponse(
            provider="fake-search",
            query="权威来源查询",
            request_id="search-request-1",
            response_time=0.25,
            usage={"credits": 1},
            results=(
                SearchResult(
                    rank=1,
                    title="权威材料",
                    url="https://authority.example/article",
                    snippet="材料摘要",
                    raw_content=(
                        "材料全文明确说明了适用范围。其他背景内容。"
                        "适用范围需要分别核对。"
                    ),
                    score=0.91,
                    content_hash="abc123",
                ),
            ),
        )
        research = ResearchOutcome(
            status="ready",
            query_plan=QueryPlan(
                goal="核实变化和边界",
                must_verify=("变化", "边界"),
                queries=(
                    PlannedQuery(query="权威来源查询", purpose="核实变化"),
                ),
                current_date="2026-08-29",
            ),
            search_responses=(search_response,),
            evidence=(
                Evidence(
                    evidence_id="E001",
                    result_ref="R001",
                    title="权威材料",
                    url="https://authority.example/article",
                    excerpt="材料全文明确说明了适用范围。",
                    source_query="权威来源查询",
                    content_hash="abc123",
                    score=0.91,
                ),
                Evidence(
                    evidence_id="E002",
                    result_ref="R001",
                    title="权威材料",
                    url="https://authority.example/article",
                    excerpt="适用范围需要分别核对。",
                    source_query="权威来源查询",
                    content_hash="abc123",
                    score=0.91,
                ),
            ),
            claims=(
                Claim(
                    claim_id="C001",
                    text="材料说明了明确适用范围。",
                    evidence_ids=("E001", "E002"),
                    is_core=True,
                ),
                Claim(
                    claim_id="C002",
                    text="不同报道对次要影响存在分歧。",
                    evidence_ids=("E001",),
                    is_core=False,
                    support_status="conflicting",
                ),
            ),
            errors=("Search request 2 failed.",),
            query_plan_prompt_version="research-query-plan-1.1.0",
            evidence_prompt_version="research-evidence-2.1.0",
            llm_request_count=2,
            search_request_count=2,
            executed_queries=(
                PlannedQuery(query="权威来源查询", purpose="核实变化"),
                PlannedQuery(query="补充查询失败", purpose="核实边界"),
            ),
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
                    raw_usage={"prompt_tokens": 100, "completion_tokens": 20},
                ),
                LLMCallUsage(
                    stage="research.evidence_selection",
                    attempt=1,
                    model="hy3",
                    request_id="hy3-2",
                    input_tokens=200,
                    output_tokens=40,
                    total_tokens=240,
                    reasoning_tokens=10,
                    cached_input_tokens=20,
                    raw_usage={"prompt_tokens": 200, "completion_tokens": 40},
                ),
            ),
        )
        script = ScriptArtifact(
            outline=("说明变化", "解释边界"),
            script_text="材料说明了明确适用范围，这项变化需要结合具体条件理解。",
            claim_usages=(
                ClaimUsage(
                    claim_id="C001",
                    script_quote="材料说明了明确适用范围",
                ),
            ),
            character_count=28,
            prompt_version="script-generation-1.0.0",
            generation_attempt_count=1,
            llm_usages=(
                LLMCallUsage(
                    stage="script.generation",
                    attempt=1,
                    model="hy3",
                    request_id="hy3-3",
                    input_tokens=50,
                    output_tokens=30,
                    total_tokens=80,
                    reasoning_tokens=8,
                    cached_input_tokens=5,
                    raw_usage={"prompt_tokens": 50, "completion_tokens": 30},
                ),
            ),
        )

        trace = build_generation_trace(
            task,
            research,
            script,
            run_id="run-test-001",
            created_at="2026-08-28T00:00:00+00:00",
            config={"research": {"max_search_requests": 5}},
        )

        self.assertEqual(trace.queries, ["权威来源查询", "补充查询失败"])
        self.assertEqual(trace.query_plan["current_date"], "2026-08-29")
        self.assertEqual(trace.search_results[0].raw_content, search_response.results[0].raw_content)
        self.assertEqual(trace.selected_evidence[0]["result_ref"], "R001")
        self.assertEqual(trace.claims[1]["support_status"], "conflicting")
        self.assertEqual(trace.errors[0]["stage"], "research")
        self.assertEqual(trace.latency["search_response_time_sum"], 0.25)
        self.assertEqual(
            trace.config["request_counts"],
            {
                "research_llm": 2,
                "search": 2,
                "script_llm": 1,
                "hy3_total": 3,
                "tavily_attempted": 2,
                "tavily_succeeded": 1,
                "tavily_failed": 1,
            },
        )
        self.assertEqual(
            trace.token_usage,
            {
                "hy3_reported_call_count": 3,
                "hy3_input_tokens": 350,
                "hy3_output_tokens": 90,
                "hy3_total_tokens": 440,
                "hy3_reasoning_tokens": 23,
                "hy3_cached_input_tokens": 35,
            },
        )
        self.assertEqual(
            trace.lineage["prompt_versions"],
            {
                "research_query_plan": "research-query-plan-1.1.0",
                "research_evidence": "research-evidence-2.1.0",
                "script_generation": "script-generation-1.0.0",
            },
        )
        self.assertEqual(
            trace.lineage["search_responses"][0]["search_result_indices"],
            [0],
        )
        self.assertEqual(
            trace.lineage["evidence_to_result_ref"],
            {"E001": "R001", "E002": "R001"},
        )
        self.assertEqual(
            trace.lineage["evidence_to_search_result_index"],
            {"E001": 0, "E002": 0},
        )
        self.assertEqual(len(trace.lineage["llm_calls"]), 3)
        self.assertEqual(trace.lineage["llm_calls"][0]["stage"], "research.query_plan")
        self.assertNotIn("evaluation", trace.config)
        self.assertNotIn("scores", trace.lineage)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            trace.write_json(path)
            frozen = load_frozen_trace(path)

        self.assertEqual(frozen.run_id, "run-test-001")
        self.assertEqual(frozen.script_text, script.script_text)
        self.assertEqual(len(frozen.selected_evidence), 2)
        self.assertEqual(
            frozen.selected_evidence[0]["excerpt"],
            "材料全文明确说明了适用范围。",
        )
        self.assertEqual(len(frozen.claims), 2)

    def test_generates_safe_default_run_metadata(self) -> None:
        trace = build_generation_trace(
            ScriptTask(topic="测试热点", target_length=50),
            ResearchOutcome(
                status="insufficient_evidence",
                query_plan=QueryPlan(
                    goal="核实",
                    must_verify=("事实",),
                    queries=(PlannedQuery(query="查询", purpose="核实"),),
                ),
                search_responses=(),
                evidence=(),
                claims=(),
                errors=("no evidence",),
                query_plan_prompt_version="query-v1",
                evidence_prompt_version="evidence-v1",
                llm_request_count=2,
                search_request_count=1,
            ),
            ScriptArtifact(
                outline=("停止",),
                script_text="",
                claim_usages=(),
                character_count=0,
                prompt_version="script-v1",
                generation_attempt_count=0,
            ),
        )

        self.assertRegex(trace.run_id, r"^run-\d{8}T\d{6}Z-[0-9a-f]{12}$")
        self.assertIn("+00:00", trace.created_at)


if __name__ == "__main__":
    unittest.main()
