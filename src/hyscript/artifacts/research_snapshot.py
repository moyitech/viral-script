"""Load a frozen research-stage snapshot for controlled script replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hyscript.agent.contracts import (
    Claim,
    Evidence,
    PlannedQuery,
    QueryPlan,
    ResearchOutcome,
    TitleChainPart,
)
from hyscript.llm import LLMCallUsage
from hyscript.search import SearchResponse, SearchResult


class ResearchSnapshotError(ValueError):
    """Raised when a serialized research outcome is incomplete or malformed."""


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ResearchSnapshotError(f"{context} must be an object.")
    return value


def _objects(value: Any, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ResearchSnapshotError(f"{context} must be a list of objects.")
    return value


def _strings(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ResearchSnapshotError(f"{context} must be a list of strings.")
    return tuple(value)


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResearchSnapshotError(f"{context} must be a non-negative integer.")
    return value


def _planned_query(payload: dict[str, Any], context: str) -> PlannedQuery:
    try:
        return PlannedQuery(query=payload["query"], purpose=payload["purpose"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ResearchSnapshotError(f"{context} is invalid.") from exc


def _query_plan(payload: dict[str, Any]) -> QueryPlan:
    queries = tuple(
        _planned_query(item, f"query_plan.queries[{index}]")
        for index, item in enumerate(_objects(payload.get("queries"), "query_plan.queries"))
    )
    current_date = payload.get("current_date")
    if current_date is not None and not isinstance(current_date, str):
        raise ResearchSnapshotError("query_plan.current_date must be a string or null.")
    try:
        return QueryPlan(
            goal=payload["goal"],
            must_verify=_strings(payload.get("must_verify"), "query_plan.must_verify"),
            queries=queries,
            current_date=current_date,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ResearchSnapshotError("query_plan is invalid.") from exc


def _search_result(payload: dict[str, Any], context: str) -> SearchResult:
    try:
        return SearchResult(
            rank=payload["rank"],
            title=payload["title"],
            url=payload["url"],
            snippet=payload["snippet"],
            raw_content=payload.get("raw_content"),
            score=payload.get("score"),
            published_at=payload.get("published_at"),
            content_hash=payload.get("content_hash"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ResearchSnapshotError(f"{context} is invalid.") from exc


def _search_response(payload: dict[str, Any], context: str) -> SearchResponse:
    results = tuple(
        _search_result(item, f"{context}.results[{index}]")
        for index, item in enumerate(_objects(payload.get("results"), f"{context}.results"))
    )
    usage = payload.get("usage", {})
    if not isinstance(usage, dict):
        raise ResearchSnapshotError(f"{context}.usage must be an object.")
    try:
        return SearchResponse(
            provider=payload["provider"],
            query=payload["query"],
            results=results,
            request_id=payload.get("request_id"),
            response_time=payload.get("response_time"),
            usage=usage,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ResearchSnapshotError(f"{context} is invalid.") from exc


def _evidence(payload: dict[str, Any], context: str) -> Evidence:
    try:
        return Evidence(
            evidence_id=payload["evidence_id"],
            result_ref=payload["result_ref"],
            title=payload["title"],
            url=payload["url"],
            excerpt=payload["excerpt"],
            source_query=payload["source_query"],
            published_at=payload.get("published_at"),
            content_hash=payload.get("content_hash"),
            score=payload.get("score"),
            source_type=payload.get("source_type", "unclassified"),
            source_scope=payload.get("source_scope", ""),
            time_basis=payload.get("time_basis", ""),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ResearchSnapshotError(f"{context} is invalid.") from exc


def _claim(payload: dict[str, Any], context: str) -> Claim:
    try:
        return Claim(
            claim_id=payload["claim_id"],
            text=payload["text"],
            evidence_ids=_strings(payload.get("evidence_ids"), f"{context}.evidence_ids"),
            is_core=payload["is_core"],
            support_status=payload.get("support_status", "supported"),
            claim_kind=payload.get("claim_kind", "descriptive_context"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ResearchSnapshotError(f"{context} is invalid.") from exc


def _title_chain_part(payload: dict[str, Any], context: str) -> TitleChainPart:
    try:
        return TitleChainPart(
            component=payload["component"],
            status=payload["status"],
            claim_ids=_strings(payload.get("claim_ids"), f"{context}.claim_ids"),
            reason=payload["reason"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ResearchSnapshotError(f"{context} is invalid.") from exc


def _optional_token_count(value: Any, context: str) -> int | None:
    if value is None:
        return None
    return _integer(value, context)


def _llm_usage(payload: dict[str, Any], context: str) -> LLMCallUsage:
    raw_usage = payload.get("raw_usage", {})
    if not isinstance(raw_usage, dict):
        raise ResearchSnapshotError(f"{context}.raw_usage must be an object.")
    try:
        return LLMCallUsage(
            stage=payload["stage"],
            attempt=_integer(payload.get("attempt"), f"{context}.attempt"),
            model=payload.get("model"),
            request_id=payload.get("request_id"),
            input_tokens=_optional_token_count(
                payload.get("input_tokens"), f"{context}.input_tokens"
            ),
            output_tokens=_optional_token_count(
                payload.get("output_tokens"), f"{context}.output_tokens"
            ),
            total_tokens=_optional_token_count(
                payload.get("total_tokens"), f"{context}.total_tokens"
            ),
            reasoning_tokens=_optional_token_count(
                payload.get("reasoning_tokens"), f"{context}.reasoning_tokens"
            ),
            cached_input_tokens=_optional_token_count(
                payload.get("cached_input_tokens"),
                f"{context}.cached_input_tokens",
            ),
            raw_usage=raw_usage,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ResearchSnapshotError(f"{context} is invalid.") from exc


def research_outcome_from_dict(payload: Any) -> ResearchOutcome:
    """Restore the exact provider-independent contracts from ``asdict`` JSON."""

    root = _object(payload, "research snapshot")
    status = root.get("status")
    if status not in {"ready", "insufficient_evidence"}:
        raise ResearchSnapshotError("research snapshot status is invalid.")
    try:
        return ResearchOutcome(
            status=status,
            query_plan=_query_plan(_object(root.get("query_plan"), "query_plan")),
            search_responses=tuple(
                _search_response(item, f"search_responses[{index}]")
                for index, item in enumerate(
                    _objects(root.get("search_responses"), "search_responses")
                )
            ),
            evidence=tuple(
                _evidence(item, f"evidence[{index}]")
                for index, item in enumerate(_objects(root.get("evidence"), "evidence"))
            ),
            claims=tuple(
                _claim(item, f"claims[{index}]")
                for index, item in enumerate(_objects(root.get("claims"), "claims"))
            ),
            errors=_strings(root.get("errors"), "errors"),
            query_plan_prompt_version=root["query_plan_prompt_version"],
            evidence_prompt_version=root["evidence_prompt_version"],
            llm_request_count=_integer(
                root.get("llm_request_count"), "llm_request_count"
            ),
            search_request_count=_integer(
                root.get("search_request_count"), "search_request_count"
            ),
            executed_queries=tuple(
                _planned_query(item, f"executed_queries[{index}]")
                for index, item in enumerate(
                    _objects(root.get("executed_queries", []), "executed_queries")
                )
            ),
            llm_usages=tuple(
                _llm_usage(item, f"llm_usages[{index}]")
                for index, item in enumerate(
                    _objects(root.get("llm_usages", []), "llm_usages")
                )
            ),
            title_chain=tuple(
                _title_chain_part(item, f"title_chain[{index}]")
                for index, item in enumerate(
                    _objects(root.get("title_chain", []), "title_chain")
                )
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ResearchSnapshotError):
            raise
        raise ResearchSnapshotError("research snapshot is invalid.") from exc


def load_research_outcome(path: Path) -> ResearchOutcome:
    """Read one UTF-8 JSON research snapshot and restore its contracts."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResearchSnapshotError(f"Could not load research snapshot: {path}") from exc
    return research_outcome_from_dict(payload)


__all__ = [
    "ResearchSnapshotError",
    "load_research_outcome",
    "research_outcome_from_dict",
]
