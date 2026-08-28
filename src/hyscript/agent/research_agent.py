"""Plan, execute, and revise live-search queries while recording lineage."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
import logging
from typing import Any, Literal, Sequence
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from hyscript.config import ResearchConfig
from hyscript.llm import (
    AsyncLLMClient,
    ChatMessage,
    LLMCallUsage,
    LLMProviderError,
    llm_call_usage,
)
from hyscript.llm.prompts import (
    RESEARCH_EVIDENCE_PROMPT_VERSION,
    RESEARCH_EVIDENCE_SYSTEM_PROMPT,
    RESEARCH_QUERY_PLAN_PROMPT_VERSION,
    RESEARCH_QUERY_PLAN_SYSTEM_PROMPT,
)
from hyscript.search import (
    AsyncSearchProvider,
    SearchProviderError,
    SearchResponse,
    SearchResult,
)

from ._structured import (
    StructuredOutputError,
    json_object,
    required_text,
    text_list,
)
from .contracts import (
    Claim,
    ClaimSupportStatus,
    Evidence,
    PlannedQuery,
    QueryPlan,
    ResearchOutcome,
    ScriptTask,
)

_AssessmentStatus = Literal["ready", "needs_more", "insufficient_evidence"]
logger = logging.getLogger(__name__)
_RESEARCH_TIMEZONE = ZoneInfo("Asia/Shanghai")


def _log_text(value: str) -> str:
    """Collapse untrusted text before placing it on one console log line."""

    return " ".join(value.split())[:200]


@dataclass(frozen=True, slots=True)
class _Candidate:
    ref: str
    query: str
    result: SearchResult
    content: str


@dataclass(frozen=True, slots=True)
class _EvidenceSelection:
    selection_ref: str
    result_ref: str
    excerpt: str


@dataclass(frozen=True, slots=True)
class _ClaimSelection:
    text: str
    evidence_refs: tuple[str, ...]
    is_core: bool
    support_status: ClaimSupportStatus


@dataclass(frozen=True, slots=True)
class _Assessment:
    status: _AssessmentStatus
    evidence: tuple[_EvidenceSelection, ...]
    claims: tuple[_ClaimSelection, ...]
    follow_up_queries: tuple[PlannedQuery, ...]


class ResearchGenerationError(RuntimeError):
    """Raised when query planning or evidence structuring repeatedly fails."""


class ResearchAgent:
    """Generate live queries, search concurrently, and select traceable evidence."""

    _MAX_LLM_ATTEMPTS = 2
    _MAX_CANDIDATES = 20
    _MAX_EVIDENCE_ITEMS = 8
    _MAX_CLAIMS = 8
    _MIN_EVIDENCE_ITEMS = 2
    _MIN_SOURCE_DOMAINS = 2
    _MAX_EXCERPT_LENGTH = 1200

    def __init__(
        self,
        llm: AsyncLLMClient,
        search: AsyncSearchProvider,
        *,
        config: ResearchConfig = ResearchConfig(),
    ) -> None:
        self._validate_config(config)
        self._llm = llm
        self._search = search
        self._config = config

    async def research(
        self,
        task: ScriptTask,
        *,
        current_date: date | None = None,
    ) -> ResearchOutcome:
        """Research one selected topic and return evidence or an explicit stop."""

        effective_date = current_date or datetime.now(_RESEARCH_TIMEZONE).date()
        research_date = effective_date.isoformat()
        logger.info("[1/5] 正在生成检索计划：%s", _log_text(task.topic))
        plan, plan_requests, plan_usages = await self._plan_queries(
            task,
            current_date=research_date,
        )
        usages = list(plan_usages)
        queries = list(plan.queries)
        logger.info(
            "检索计划完成，共 %d 个初始查询：%s",
            len(queries),
            "；".join(_log_text(item.query) for item in queries),
        )
        logger.info("[2/5] 正在并发执行 %d 个初始搜索", len(queries))
        responses, errors = await self._search_queries(queries)
        search_request_count = len(queries)
        llm_request_count = plan_requests
        logger.info(
            "初始搜索完成：成功 %d，失败 %d",
            len(responses),
            len(errors),
        )

        candidates = self._collect_candidates(responses)
        if not candidates:
            logger.warning("[3/5] 没有可用搜索正文，调研停止")
            return self._outcome(
                status="insufficient_evidence",
                plan=plan,
                responses=responses,
                evidence=(),
                claims=(),
                errors=(*errors, "Search returned no usable evidence content."),
                llm_request_count=llm_request_count,
                search_request_count=search_request_count,
                executed_queries=queries,
                llm_usages=usages,
            )

        remaining_budget = self._config.max_search_requests - search_request_count
        logger.info(
            "[3/5] 正在从 %d 条候选结果中筛选证据和论断",
            len(candidates),
        )
        assessment, assessment_requests, assessment_usages = await self._assess_evidence(
            task,
            plan,
            candidates,
            remaining_search_budget=remaining_budget,
        )
        llm_request_count += assessment_requests
        usages.extend(assessment_usages)

        if assessment.status == "needs_more" and remaining_budget > 0:
            existing_queries = {item.query.casefold() for item in queries}
            follow_up_queries = tuple(
                item
                for item in assessment.follow_up_queries
                if item.query.casefold() not in existing_queries
            )[:remaining_budget]
            if follow_up_queries:
                logger.info(
                    "证据编辑要求补搜 %d 次：%s",
                    len(follow_up_queries),
                    "；".join(_log_text(item.query) for item in follow_up_queries),
                )
                follow_up_responses, follow_up_errors = await self._search_queries(
                    follow_up_queries
                )
                queries.extend(follow_up_queries)
                responses = (*responses, *follow_up_responses)
                errors = (*errors, *follow_up_errors)
                search_request_count += len(follow_up_queries)
                candidates = self._collect_candidates(responses)
                logger.info(
                    "补充搜索完成：成功 %d，失败 %d；正在重新筛选证据",
                    len(follow_up_responses),
                    len(follow_up_errors),
                )
                assessment, extra_requests, extra_usages = await self._assess_evidence(
                    task,
                    plan,
                    candidates,
                    remaining_search_budget=0,
                )
                llm_request_count += extra_requests
                usages.extend(extra_usages)

        evidence, claims = self._materialize(assessment, candidates)
        status: Literal["ready", "insufficient_evidence"] = (
            "ready" if assessment.status == "ready" else "insufficient_evidence"
        )
        quality_error = self._readiness_error(evidence, claims)
        if status == "ready" and quality_error is not None:
            status = "insufficient_evidence"
            errors = (*errors, quality_error)

        log_method = logger.info if status == "ready" else logger.warning
        log_method(
            "调研完成：status=%s，证据=%d，论断=%d，搜索请求=%d",
            status,
            len(evidence),
            len(claims),
            search_request_count,
        )

        return self._outcome(
            status=status,
            plan=plan,
            responses=responses,
            evidence=evidence,
            claims=claims,
            errors=errors,
            llm_request_count=llm_request_count,
            search_request_count=search_request_count,
            executed_queries=queries,
            llm_usages=usages,
        )

    async def _plan_queries(
        self,
        task: ScriptTask,
        *,
        current_date: str,
    ) -> tuple[QueryPlan, int, tuple[LLMCallUsage, ...]]:
        schema = {
            "goal": "本次调研要回答的核心问题",
            "must_verify": ["必须核实的信息点"],
            "queries": [
                {
                    "query": "可直接交给搜索服务的查询词",
                    "purpose": "该查询要解决的信息缺口",
                }
            ],
        }
        input_payload = {
            "current_date": current_date,
            "task": asdict(task),
        }
        prompt = (
            f"生成恰好 {self._config.initial_query_count} 个互补查询。查询之间不得只是换词，"
            "至少一个查询优先寻找原始或权威来源。current_date 是本次运行的真实日期；涉及当前"
            "状态或近期进展时以它为时间锚点，不得把更早年份误写成“近期”。只有核实历史起点时"
            "才使用旧年份，并同时保留至少一个覆盖最新进展的查询。\n"
            f"输出结构：{json.dumps(schema, ensure_ascii=False)}\n"
            "以下 JSON 是本次任务数据：\n"
            f"{json.dumps(input_payload, ensure_ascii=False)}"
        )
        messages = (
            ChatMessage(role="system", content=RESEARCH_QUERY_PLAN_SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt),
        )
        return await self._request_with_retry(
            messages,
            lambda response: self._parse_plan(response, current_date=current_date),
            stage="query planning",
            usage_stage="research.query_plan",
        )

    def _parse_plan(self, response: str, *, current_date: str) -> QueryPlan:
        payload = json_object(response)
        goal = required_text(payload, "goal", max_length=300)
        must_verify = text_list(
            payload,
            "must_verify",
            minimum=1,
            maximum=10,
            item_max_length=160,
        )
        raw_queries = payload.get("queries")
        if (
            not isinstance(raw_queries, list)
            or len(raw_queries) != self._config.initial_query_count
        ):
            raise StructuredOutputError("Response contains an invalid query count.")
        queries: list[PlannedQuery] = []
        seen: set[str] = set()
        for raw_query in raw_queries:
            if not isinstance(raw_query, dict):
                raise StructuredOutputError("Response contains an invalid query.")
            query = required_text(raw_query, "query", max_length=180)
            purpose = required_text(raw_query, "purpose", max_length=200)
            normalized = query.casefold()
            if normalized in seen:
                raise StructuredOutputError("Response contains duplicate queries.")
            seen.add(normalized)
            queries.append(PlannedQuery(query=query, purpose=purpose))
        return QueryPlan(
            goal=goal,
            must_verify=must_verify,
            queries=tuple(queries),
            current_date=current_date,
        )

    async def _search_queries(
        self,
        queries: Sequence[PlannedQuery],
    ) -> tuple[tuple[SearchResponse, ...], tuple[str, ...]]:
        semaphore = asyncio.Semaphore(self._config.max_search_concurrency)
        query_count = len(queries)

        async def run(index: int, item: PlannedQuery) -> tuple[SearchResponse | None, str | None]:
            async with semaphore:
                logger.info(
                    "搜索 %d/%d 开始：%s",
                    index,
                    query_count,
                    _log_text(item.query),
                )
                try:
                    response = await self._search.search(
                        item.query,
                        limit=self._config.results_per_query,
                    )
                except SearchProviderError:
                    logger.warning(
                        "搜索 %d/%d 失败：%s",
                        index,
                        query_count,
                        _log_text(item.query),
                    )
                    return None, f"Search request {index} failed."
                logger.info(
                    "搜索 %d/%d 完成：返回 %d 条结果",
                    index,
                    query_count,
                    len(response.results),
                )
                return response, None

        results = await asyncio.gather(
            *(run(index, item) for index, item in enumerate(queries, start=1))
        )
        return (
            tuple(response for response, _ in results if response is not None),
            tuple(error for _, error in results if error is not None),
        )

    def _collect_candidates(
        self,
        responses: Sequence[SearchResponse],
    ) -> tuple[_Candidate, ...]:
        candidates: list[_Candidate] = []
        seen_urls: set[str] = set()
        max_result_count = max((len(response.results) for response in responses), default=0)
        for result_index in range(max_result_count):
            for response in responses:
                if result_index >= len(response.results):
                    continue
                result = response.results[result_index]
                normalized_url = result.url.strip()
                parsed = urlparse(normalized_url)
                if (
                    not normalized_url
                    or normalized_url in seen_urls
                    or parsed.scheme not in {"http", "https"}
                    or not parsed.netloc
                ):
                    continue
                content = (result.raw_content or result.snippet).strip()
                if not content:
                    continue
                seen_urls.add(normalized_url)
                candidates.append(
                    _Candidate(
                        ref=f"R{len(candidates) + 1:03d}",
                        query=response.query,
                        result=result,
                        content=content[: self._config.max_content_chars_per_result],
                    )
                )
                if len(candidates) >= self._MAX_CANDIDATES:
                    return tuple(candidates)
        return tuple(candidates)

    async def _assess_evidence(
        self,
        task: ScriptTask,
        plan: QueryPlan,
        candidates: Sequence[_Candidate],
        *,
        remaining_search_budget: int,
    ) -> tuple[_Assessment, int, tuple[LLMCallUsage, ...]]:
        schema = {
            "status": "ready | needs_more | insufficient_evidence",
            "evidence": [
                {
                    "selection_ref": "S001",
                    "result_ref": "R001",
                    "excerpt": "从对应 content 逐字复制的证据片段",
                }
            ],
            "claims": [
                {
                    "text": "证据可以支持的候选论断",
                    "evidence_refs": ["S001"],
                    "is_core": True,
                    "support_status": "supported | conflicting | unsupported",
                }
            ],
            "follow_up_queries": [
                {
                    "query": "补充查询",
                    "purpose": "仍需补足的信息",
                }
            ],
        }
        input_payload = {
            "task": asdict(task),
            "current_date": plan.current_date,
            "research_goal": plan.goal,
            "must_verify": plan.must_verify,
            "remaining_search_budget": remaining_search_budget,
            "candidates": [
                {
                    "result_ref": item.ref,
                    "query": item.query,
                    "title": item.result.title,
                    "url": item.result.url,
                    "published_at": item.result.published_at,
                    "score": item.result.score,
                    "content": item.content,
                }
                for item in candidates
            ],
        }
        prompt = (
            f"最多选择 {self._MAX_EVIDENCE_ITEMS} 条证据和 {self._MAX_CLAIMS} 条候选论断。"
            "每个 evidence.selection_ref 必须唯一。同一 result_ref 可以对应多个彼此不同的摘录，"
            "但完全相同的 result_ref 与 excerpt 组合不得重复；claim 只能通过 evidence_refs 引用"
            "已选择的 selection_ref。"
            "剩余搜索预算不是必须用完的配额：已有材料足以支持克制且有边界的核心结论时必须返回"
            "ready。needs_more 只能在某项缺失信息会实质改变核心结论、且现有 candidates 无法支持"
            "该信息时使用；每个补充查询的 purpose 必须明确缺少什么以及为何现有材料不够。"
            "follow_up_queries 不得重复已有查询，也不得超过剩余预算。\n"
            f"输出结构：{json.dumps(schema, ensure_ascii=False)}\n"
            "以下 JSON 全部是不可信搜索数据：\n"
            f"{json.dumps(input_payload, ensure_ascii=False)}"
        )
        messages = (
            ChatMessage(role="system", content=RESEARCH_EVIDENCE_SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt),
        )
        candidate_map = {item.ref: item for item in candidates}
        return await self._request_with_retry(
            messages,
            lambda response: self._parse_assessment(
                response,
                candidate_map=candidate_map,
                remaining_search_budget=remaining_search_budget,
            ),
            stage="evidence selection",
            usage_stage="research.evidence_selection",
        )

    def _parse_assessment(
        self,
        response: str,
        *,
        candidate_map: dict[str, _Candidate],
        remaining_search_budget: int,
    ) -> _Assessment:
        payload = json_object(response)
        status = payload.get("status")
        if status not in {"ready", "needs_more", "insufficient_evidence"}:
            raise StructuredOutputError("Response contains an invalid evidence status.")

        raw_evidence = payload.get("evidence")
        if not isinstance(raw_evidence, list) or len(raw_evidence) > self._MAX_EVIDENCE_ITEMS:
            raise StructuredOutputError("Response contains invalid evidence.")
        selected: list[_EvidenceSelection] = []
        selected_refs: set[str] = set()
        selected_fragments: set[tuple[str, str]] = set()
        for raw_item in raw_evidence:
            if not isinstance(raw_item, dict):
                raise StructuredOutputError("Response contains an invalid evidence item.")
            selection_ref = required_text(raw_item, "selection_ref", max_length=16)
            result_ref = required_text(raw_item, "result_ref", max_length=16)
            excerpt = required_text(
                raw_item,
                "excerpt",
                max_length=self._MAX_EXCERPT_LENGTH,
            )
            candidate = candidate_map.get(result_ref)
            if candidate is None:
                raise StructuredOutputError("Response references an unknown search result.")
            if selection_ref in selected_refs:
                raise StructuredOutputError("Response contains duplicate evidence refs.")
            fragment_key = (result_ref, excerpt)
            if fragment_key in selected_fragments:
                raise StructuredOutputError("Response selects an identical evidence fragment.")
            if excerpt not in candidate.content:
                raise StructuredOutputError("Evidence excerpt is not present in source content.")
            selected_refs.add(selection_ref)
            selected_fragments.add(fragment_key)
            selected.append(
                _EvidenceSelection(
                    selection_ref=selection_ref,
                    result_ref=result_ref,
                    excerpt=excerpt,
                )
            )

        raw_claims = payload.get("claims")
        if not isinstance(raw_claims, list) or len(raw_claims) > self._MAX_CLAIMS:
            raise StructuredOutputError("Response contains invalid claims.")
        claims: list[_ClaimSelection] = []
        seen_claims: set[str] = set()
        for raw_claim in raw_claims:
            if not isinstance(raw_claim, dict):
                raise StructuredOutputError("Response contains an invalid claim.")
            text = required_text(raw_claim, "text", max_length=300)
            normalized_text = text.casefold()
            if normalized_text in seen_claims:
                raise StructuredOutputError("Response contains duplicate claims.")
            seen_claims.add(normalized_text)
            raw_refs = raw_claim.get("evidence_refs")
            if not isinstance(raw_refs, list) or not raw_refs:
                raise StructuredOutputError("Response contains a claim without refs.")
            evidence_refs = tuple(
                dict.fromkeys(
                    ref.strip()
                    for ref in raw_refs
                    if isinstance(ref, str) and ref.strip()
                )
            )
            if not evidence_refs or any(ref not in selected_refs for ref in evidence_refs):
                raise StructuredOutputError("Claim references unselected evidence.")
            is_core = raw_claim.get("is_core")
            if not isinstance(is_core, bool):
                raise StructuredOutputError("Claim is_core must be boolean.")
            support_status = raw_claim.get("support_status")
            if support_status not in {"supported", "conflicting", "unsupported"}:
                raise StructuredOutputError("Claim support_status is invalid.")
            claims.append(
                _ClaimSelection(
                    text=text,
                    evidence_refs=evidence_refs,
                    is_core=is_core,
                    support_status=support_status,
                )
            )

        raw_follow_ups = payload.get("follow_up_queries", [])
        if not isinstance(raw_follow_ups, list):
            raise StructuredOutputError("Response contains invalid follow-up queries.")
        if len(raw_follow_ups) > remaining_search_budget:
            raise StructuredOutputError("Response exceeds the remaining search budget.")
        follow_ups: list[PlannedQuery] = []
        seen_follow_ups: set[str] = set()
        for raw_query in raw_follow_ups:
            if not isinstance(raw_query, dict):
                raise StructuredOutputError("Response contains an invalid follow-up query.")
            query = required_text(raw_query, "query", max_length=180)
            purpose = required_text(raw_query, "purpose", max_length=200)
            normalized_query = query.casefold()
            if normalized_query in seen_follow_ups:
                raise StructuredOutputError("Response contains duplicate follow-up queries.")
            seen_follow_ups.add(normalized_query)
            follow_ups.append(PlannedQuery(query=query, purpose=purpose))

        if status == "needs_more" and (remaining_search_budget < 1 or not follow_ups):
            raise StructuredOutputError("needs_more requires a follow-up query and budget.")
        if status == "ready" and (
            not selected
            or not claims
            or not any(item.is_core and item.support_status == "supported" for item in claims)
        ):
            raise StructuredOutputError("ready requires supported core claims.")
        return _Assessment(
            status=status,
            evidence=tuple(selected),
            claims=tuple(claims),
            follow_up_queries=tuple(follow_ups),
        )

    @staticmethod
    def _materialize(
        assessment: _Assessment,
        candidates: Sequence[_Candidate],
    ) -> tuple[tuple[Evidence, ...], tuple[Claim, ...]]:
        candidate_map = {item.ref: item for item in candidates}
        ref_to_evidence_id = {
            item.selection_ref: f"E{index:03d}"
            for index, item in enumerate(assessment.evidence, start=1)
        }
        evidence = tuple(
            Evidence(
                evidence_id=ref_to_evidence_id[item.selection_ref],
                result_ref=item.result_ref,
                title=candidate_map[item.result_ref].result.title,
                url=candidate_map[item.result_ref].result.url,
                excerpt=item.excerpt,
                source_query=candidate_map[item.result_ref].query,
                published_at=candidate_map[item.result_ref].result.published_at,
                content_hash=candidate_map[item.result_ref].result.content_hash,
                score=candidate_map[item.result_ref].result.score,
            )
            for item in assessment.evidence
        )
        claims = tuple(
            Claim(
                claim_id=f"C{index:03d}",
                text=item.text,
                evidence_ids=tuple(
                    ref_to_evidence_id[ref]
                    for ref in item.evidence_refs
                ),
                is_core=item.is_core,
                support_status=item.support_status,
            )
            for index, item in enumerate(assessment.claims, start=1)
        )
        return evidence, claims

    def _readiness_error(
        self,
        evidence: Sequence[Evidence],
        claims: Sequence[Claim],
    ) -> str | None:
        if len(evidence) < self._MIN_EVIDENCE_ITEMS:
            return "Ready research requires at least two evidence items."
        domains = {
            urlparse(item.url).netloc.casefold().removeprefix("www.")
            for item in evidence
        }
        if len(domains) < self._MIN_SOURCE_DOMAINS:
            return "Ready research requires evidence from at least two source domains."
        core_claims = [item for item in claims if item.is_core]
        if not core_claims or any(
            item.support_status != "supported" for item in core_claims
        ):
            return "Every core claim must be supported before script generation."
        return None

    async def _request_with_retry(
        self,
        messages: Sequence[ChatMessage],
        parser: Any,
        *,
        stage: str,
        usage_stage: str,
    ) -> tuple[Any, int, tuple[LLMCallUsage, ...]]:
        current_messages = tuple(messages)
        usages: list[LLMCallUsage] = []
        for attempt in range(1, self._MAX_LLM_ATTEMPTS + 1):
            response_content: str | None = None
            try:
                response = await self._llm.complete(
                    current_messages,
                    reasoning_effort="high",
                )
                response_content = response.content
                usage = llm_call_usage(
                    response,
                    stage=usage_stage,
                    attempt=attempt,
                )
                usages.append(usage)
                self._log_token_usage(usage)
                return parser(response_content), attempt, tuple(usages)
            except LLMProviderError:
                logger.warning(
                    "%s 的第 %d 次 Hy3 请求失败",
                    stage,
                    attempt,
                )
            except StructuredOutputError as exc:
                logger.warning(
                    "%s 的第 %d 次输出未通过校验：%s",
                    stage,
                    attempt,
                    exc,
                )
                if response_content is not None:
                    current_messages = (
                        *messages,
                        ChatMessage(role="assistant", content=response_content),
                        ChatMessage(
                            role="user",
                            content=(
                                f"上一次输出未通过结构校验：{exc} "
                                "请重新输出完整 JSON，不要解释。"
                            ),
                        ),
                    )
            if attempt >= self._MAX_LLM_ATTEMPTS:
                raise ResearchGenerationError(
                    f"Research {stage} failed after one retry."
                ) from None
        raise AssertionError("unreachable")

    @staticmethod
    def _log_token_usage(usage: LLMCallUsage) -> None:
        logger.info(
            "Hy3 usage：stage=%s，input=%s，output=%s，total=%s",
            usage.stage,
            usage.input_tokens if usage.input_tokens is not None else "unknown",
            usage.output_tokens if usage.output_tokens is not None else "unknown",
            usage.total_tokens if usage.total_tokens is not None else "unknown",
        )

    @staticmethod
    def _validate_config(config: ResearchConfig) -> None:
        if not 1 <= config.initial_query_count <= 5:
            raise ValueError("initial_query_count must be between 1 and 5.")
        if not config.initial_query_count <= config.max_search_requests <= 10:
            raise ValueError("max_search_requests is outside the supported range.")
        if not 1 <= config.results_per_query <= 20:
            raise ValueError("results_per_query must be between 1 and 20.")
        if not 1 <= config.max_search_concurrency <= 10:
            raise ValueError("max_search_concurrency must be between 1 and 10.")
        if not 500 <= config.max_content_chars_per_result <= 20000:
            raise ValueError("max_content_chars_per_result is outside the supported range.")

    @staticmethod
    def _outcome(
        *,
        status: Literal["ready", "insufficient_evidence"],
        plan: QueryPlan,
        responses: Sequence[SearchResponse],
        evidence: Sequence[Evidence],
        claims: Sequence[Claim],
        errors: Sequence[str],
        llm_request_count: int,
        search_request_count: int,
        executed_queries: Sequence[PlannedQuery],
        llm_usages: Sequence[LLMCallUsage],
    ) -> ResearchOutcome:
        return ResearchOutcome(
            status=status,
            query_plan=plan,
            search_responses=tuple(responses),
            evidence=tuple(evidence),
            claims=tuple(claims),
            errors=tuple(errors),
            query_plan_prompt_version=RESEARCH_QUERY_PLAN_PROMPT_VERSION,
            evidence_prompt_version=RESEARCH_EVIDENCE_PROMPT_VERSION,
            llm_request_count=llm_request_count,
            search_request_count=search_request_count,
            executed_queries=tuple(executed_queries),
            llm_usages=tuple(llm_usages),
        )
