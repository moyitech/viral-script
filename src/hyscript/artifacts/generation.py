"""Adapt generation-stage contracts into a frozen, score-free run trace."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

from hyscript.agent.contracts import (
    PlannedQuery,
    ResearchOutcome,
    ScriptArtifact,
    ScriptTask,
)
from hyscript.llm import summarize_token_usage

from .trace import RunTrace, TraceSearchResult


def build_generation_trace(
    task: ScriptTask,
    research: ResearchOutcome,
    script: ScriptArtifact,
    *,
    run_id: str | None = None,
    created_at: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> RunTrace:
    """Build a serializable trace without invoking or embedding evaluation."""

    effective_created_at = created_at or datetime.now(timezone.utc).isoformat()
    effective_run_id = run_id or _new_run_id()

    search_results: list[TraceSearchResult] = []
    search_lineage: list[dict[str, Any]] = []
    for response_index, response in enumerate(research.search_responses):
        first_result_index = len(search_results)
        search_results.extend(
            TraceSearchResult(
                rank=result.rank,
                title=result.title,
                url=result.url,
                snippet=result.snippet,
                raw_content=result.raw_content,
                score=result.score,
                published_at=result.published_at,
                content_hash=result.content_hash,
            )
            for result in response.results
        )
        search_lineage.append(
            {
                "response_index": response_index,
                "provider": response.provider,
                "query": response.query,
                "request_id": response.request_id,
                "response_time": response.response_time,
                "usage": dict(response.usage),
                "search_result_indices": list(
                    range(first_result_index, len(search_results))
                ),
            }
        )

    latency = {
        "search_response_time_sum": sum(
            response.response_time
            for response in research.search_responses
            if response.response_time is not None
        )
    }
    trace_config = dict(config or {})
    tavily_success_count = len(research.search_responses)
    tavily_failure_count = research.search_request_count - tavily_success_count
    llm_usages = (*research.llm_usages, *script.llm_usages)
    token_summary = summarize_token_usage(llm_usages)
    format_repair_calls = script.format_repair_attempt_count
    content_generation_calls = script.content_generation_attempt_count
    script_generation_calls = script.generation_attempt_count
    script_review_calls = script.grounding_review_attempt_count
    script_final_rewrite_calls = script.final_rewrite_attempt_count
    script_editor_calls = script.editor_attempt_count
    script_candidate_calls = (
        script_generation_calls - script_editor_calls
        if script.generation_mode == "editorial_candidates"
        else 0
    )
    if not content_generation_calls:
        content_generation_calls = (
            script_candidate_calls
            if script.generation_mode == "editorial_candidates"
            else script_generation_calls
        )
    script_calls = (
        script_generation_calls + script_review_calls + script_final_rewrite_calls
    )
    trace_config["request_counts"] = {
        "research_llm": research.llm_request_count,
        "search": research.search_request_count,
        "script_llm": script_calls,
        "script_generation_llm": script_generation_calls,
        "script_grounding_review_llm": script_review_calls,
        "script_final_rewrite_llm": script_final_rewrite_calls,
        "hy3_total": research.llm_request_count + script_calls,
        "tavily_attempted": research.search_request_count,
        "tavily_succeeded": tavily_success_count,
        "tavily_failed": tavily_failure_count,
    }
    if script.generation_mode == "editorial_candidates":
        trace_config["request_counts"].update(
            {
                "script_candidate_llm": script_candidate_calls,
                "script_editor_llm": script_editor_calls,
            }
        )
    elif script.generation_mode == "single_shot":
        trace_config["request_counts"].update(
            {
                "script_content_generation_llm": content_generation_calls,
                "script_format_repair_llm": format_repair_calls,
            }
        )
    executed_queries = research.executed_queries or _known_queries(research)
    retained_reference_ids = set(script.reference_ids)
    retained_evidence = (
        tuple(
            item
            for item in research.evidence
            if item.evidence_id in retained_reference_ids
        )
        if retained_reference_ids
        else research.evidence
    )
    first_search_index_by_url: dict[str, int] = {}
    for index, result in enumerate(search_results):
        first_search_index_by_url.setdefault(result.url, index)
    return RunTrace(
        run_id=effective_run_id,
        created_at=effective_created_at,
        task=asdict(task),
        config=trace_config,
        query_plan=asdict(research.query_plan),
        queries=[item.query for item in executed_queries],
        search_results=search_results,
        selected_evidence=[asdict(item) for item in retained_evidence],
        claims=[asdict(item) for item in research.claims],
        script_artifact=asdict(script),
        errors=[
            {"stage": "research", "message": message}
            for message in research.errors
        ],
        latency=latency,
        token_usage={
            "hy3_reported_call_count": token_summary.reported_call_count,
            "hy3_input_tokens": token_summary.input_tokens,
            "hy3_output_tokens": token_summary.output_tokens,
            "hy3_total_tokens": token_summary.total_tokens,
            "hy3_reasoning_tokens": token_summary.reasoning_tokens,
            "hy3_cached_input_tokens": token_summary.cached_input_tokens,
        },
        lineage={
            "prompt_versions": {
                "research_query_plan": research.query_plan_prompt_version,
                "research_evidence": research.evidence_prompt_version,
                "script_generation": script.prompt_version,
                "script_grounding_review": (
                    script.grounding_review_prompt_version
                ),
                "script_final_rewrite": script.final_rewrite_prompt_version,
                **(
                    {
                        "script_format_repair": (
                            script.format_repair_prompt_version
                        )
                    }
                    if script.generation_mode == "single_shot"
                    else {}
                ),
                **(
                    {
                        "script_candidate": (
                            script.generation_candidates[0].prompt_version
                        ),
                        "script_editor": script.editor_prompt_version,
                    }
                    if script.generation_candidates
                    else {}
                ),
            },
            "search_responses": search_lineage,
            "llm_calls": [asdict(item) for item in llm_usages],
            "evidence_to_result_ref": {
                item.evidence_id: item.result_ref for item in research.evidence
            },
            "evidence_to_search_result_index": {
                item.evidence_id: first_search_index_by_url.get(item.url)
                for item in research.evidence
            },
            "claim_to_evidence": {
                item.claim_id: list(item.evidence_ids) for item in research.claims
            },
            "claim_to_script_quote": {
                item.claim_id: item.script_quote for item in script.claim_usages
            },
            "script_reference_ids": list(script.reference_ids),
            "script_generation_mode": script.generation_mode,
            "script_selected_candidate_ids": list(script.selected_candidate_ids),
            "script_candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "strategy": item.strategy,
                    "prompt_version": item.prompt_version,
                    "reference_ids": list(item.reference_ids),
                    "character_count": item.character_count,
                }
                for item in script.generation_candidates
            ],
            "research_title_chain": [
                {
                    "component": item.component,
                    "status": item.status,
                    "claim_ids": list(item.claim_ids),
                    "reason": item.reason,
                }
                for item in research.title_chain
            ],
        },
    )


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{timestamp}-{uuid4().hex[:12]}"


def _known_queries(research: ResearchOutcome) -> tuple[PlannedQuery, ...]:
    """Backfill attempted-query lineage for manually constructed outcomes."""

    queries = list(research.query_plan.queries)
    seen = {item.query.casefold() for item in queries}
    for response in research.search_responses:
        if response.query.casefold() not in seen:
            queries.append(
                PlannedQuery(
                    query=response.query,
                    purpose="Follow-up query recovered from search response.",
                )
            )
            seen.add(response.query.casefold())
    return tuple(queries)
