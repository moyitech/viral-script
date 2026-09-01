"""Generate and validate an evidence-grounded oral script."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import json
import logging
import math
import re
from typing import Sequence

from hyscript.config import ScriptGenerationConfig
from hyscript.llm import (
    AsyncLLMClient,
    ChatMessage,
    LLMCallUsage,
    LLMProviderError,
    llm_call_usage,
)
from hyscript.llm.prompts import (
    BACKGROUND_SCRIPT_CANDIDATE_PROMPT_VERSION,
    BACKGROUND_SCRIPT_CANDIDATE_SYSTEM_PROMPT,
    BACKGROUND_SCRIPT_EDITOR_PROMPT_VERSION,
    BACKGROUND_SCRIPT_EDITOR_SYSTEM_PROMPT,
    BACKGROUND_SCRIPT_GENERATION_PROMPT_VERSION,
    BACKGROUND_SCRIPT_GENERATION_SYSTEM_PROMPT,
    BACKGROUND_SCRIPT_PIPELINE_VERSION,
    RESEARCH_EVIDENCE_PROMPT_VERSION,
    SCRIPT_GENERATION_PROMPT_VERSION,
    SCRIPT_GENERATION_SYSTEM_PROMPT,
    SCRIPT_GROUNDING_REVIEW_PROMPT_VERSION,
    SCRIPT_GROUNDING_REVIEW_SYSTEM_PROMPT,
)

from ._structured import (
    StructuredOutputError,
    json_object,
    required_text,
    text_list,
)
from .contracts import (
    CORE_SOURCE_TYPES_BY_CLAIM_KIND,
    Claim,
    ClaimUsage,
    Evidence,
    ResearchOutcome,
    ScriptArtifact,
    ScriptCandidate,
    ScriptTask,
)


_URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_CITATION_PATTERN = re.compile(r"(?:\[\d+\]|【\d+】)")
_MARKDOWN_PATTERN = re.compile(
    r"(?:```|^\s{0,3}(?:#{1,6}\s+|>\s+|[-+*]\s+|\d+[.)]\s+)|"
    r"\[[^\]\n]+\]\([^\)\n]+\)|\*\*[^*\n]+\*\*|__[^_\n]+__)",
    re.MULTILINE,
)
_META_PATTERN = re.compile(
    r"(?:</?think(?:\s[^>]*)?>|"
    r"^\s*(?:标题|正文|口播稿|参考资料|引用来源|写作说明|修改说明|"
    r"开场钩子|结构分析|写作思路|自我评价)\s*[:：]|"
    r"(?:这|本)(?:篇|段)?(?:文案|口播).{0,18}(?:符合|满足|达到).{0,12}"
    r"(?:要求|标准|高分|满分))",
    re.IGNORECASE | re.MULTILINE,
)
_CLAUSE_BOUNDARY_PATTERN = re.compile(r"[。！？；!?;]+[”’」』》】）\])]*")
logger = logging.getLogger(__name__)


class ScriptGenerationError(RuntimeError):
    """Raised when research is unusable or script generation cannot be repaired.

    Partial usage is retained so batch manifests can account for valid provider
    responses even when no final artifact was produced.
    """

    def __init__(
        self,
        message: str,
        *,
        generation_attempt_count: int = 0,
        grounding_review_attempt_count: int = 0,
        llm_usages: Sequence[LLMCallUsage] = (),
    ) -> None:
        super().__init__(message)
        self.generation_attempt_count = generation_attempt_count
        self.grounding_review_attempt_count = grounding_review_attempt_count
        self.llm_usages = tuple(llm_usages)


class ScriptAgent:
    """Turn ready research into a clean oral script with claim lineage."""

    _MAX_OUTLINE_ITEMS = 8
    _MAX_SCRIPT_LENGTH = 10000
    _MAX_QUOTE_LENGTH = 500
    _MAX_PROVIDER_RETRIES = 2
    _MAX_GROUNDING_REVIEW_ATTEMPTS = 2
    _MAX_REVIEW_ISSUES = 8
    _EDITORIAL_STRATEGIES = (
        (
            "C01",
            "conflict_interest",
            "冲突/利益优先：第一句话就让观众看见具体损失、选择冲突或切身利益；中段逐步兑现开头，结尾给出改变判断的落点。",
        ),
        (
            "C02",
            "scene_conversation",
            "场景/对话优先：从普通人能代入的动作、瞬间或自然问句进入；核心解释始终像面对观众说话，不能退回新闻解说。",
        ),
        (
            "C03",
            "counterintuitive_turn",
            "反常识/转折优先：先给出鲜明但克制的反差判断，再用至少两次转折揭示原因、边界和普通人影响。",
        ),
    )
    _CLAUSE_AUDIT_KINDS = frozenset({"D", "I", "A"})
    _REVIEW_ISSUE_CODES = frozenset(
        {
            "unsupported_claim",
            "scope_expansion",
            "causal_leap",
            "source_attribution",
            "current_rule_gap",
            "unsupported_advice",
            "repetition",
            "oral_style",
            "insufficient_evidence",
            "other",
        }
    )
    def __init__(
        self,
        llm: AsyncLLMClient,
        *,
        config: ScriptGenerationConfig = ScriptGenerationConfig(),
        request_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self._validate_config(config)
        self._llm = llm
        self._config = config
        self._request_semaphore = request_semaphore

    async def generate(
        self,
        task: ScriptTask,
        research: ResearchOutcome,
    ) -> ScriptArtifact:
        """Generate one validated script from background or legacy evidence data."""

        if research.status != "ready":
            raise ScriptGenerationError(
                "Script generation requires research with status 'ready'."
            )
        if not research.claims:
            if self._config.generation_mode == "editorial_candidates":
                return await self._generate_from_background_editorial(task, research)
            return await self._generate_from_background(task, research)
        evidence_ids = tuple(item.evidence_id for item in research.evidence)
        claim_ids = tuple(item.claim_id for item in research.claims)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ScriptGenerationError("Research contains duplicate evidence IDs.")
        if len(set(claim_ids)) != len(claim_ids):
            raise ScriptGenerationError("Research contains duplicate claim IDs.")
        supported_claims = tuple(
            claim for claim in research.claims if claim.support_status == "supported"
        )
        if not supported_claims or not any(claim.is_core for claim in supported_claims):
            raise ScriptGenerationError(
                "Script generation requires at least one supported core claim."
            )

        known_evidence_ids = set(evidence_ids)
        if any(
            not claim.evidence_ids
            or any(item not in known_evidence_ids for item in claim.evidence_ids)
            for claim in supported_claims
        ):
            raise ScriptGenerationError(
                "Supported claims must reference evidence present in the research outcome."
            )

        evidence_by_id = {item.evidence_id: item for item in research.evidence}
        if any(
            not self._claim_has_eligible_source(claim, evidence_by_id)
            for claim in supported_claims
            if claim.is_core
        ):
            raise ScriptGenerationError(
                "Core claim source quality does not satisfy its claim_kind."
            )
        self._validate_research_title_chain(research, supported_claims)
        generation_claims = tuple(
            claim
            for claim in supported_claims
            if claim.is_core
            or self._claim_has_eligible_source(claim, evidence_by_id)
        )

        messages = self._messages(task, research, generation_claims)
        minimum_length, maximum_length = self._length_bounds(task)
        current_messages = messages
        usages: list[LLMCallUsage] = []
        last_failure = "unknown failure"
        request_count = 0
        content_attempt_count = 0
        consecutive_provider_failures = 0
        max_request_count = (
            self._config.max_generation_attempts + self._MAX_PROVIDER_RETRIES
        )
        logger.info(
            "[4/5] 正在生成口播文案：目标 %d 个非空白字符",
            task.target_length,
        )
        while (
            content_attempt_count < self._config.max_generation_attempts
            and request_count < max_request_count
        ):
            request_count += 1
            logger.info(
                "文案生成请求 %d/%d（已使用内容修复额度 %d/%d）",
                request_count,
                max_request_count,
                content_attempt_count,
                self._config.max_generation_attempts,
            )
            try:
                response = await self._complete(current_messages)
            except LLMProviderError as exc:
                consecutive_provider_failures += 1
                last_failure = self._failure_reason("provider_error", exc)
                logger.warning("第 %d 次文案 Hy3 请求失败", request_count)
                if consecutive_provider_failures > self._MAX_PROVIDER_RETRIES:
                    break
                continue

            consecutive_provider_failures = 0
            content_attempt_count += 1
            response_content = response.content
            usage = llm_call_usage(
                response,
                stage="script.generation",
                attempt=request_count,
            )
            usages.append(usage)
            self._log_token_usage(usage)
            try:
                artifact = self._parse_artifact(
                    response_content,
                    task=task,
                    supported_claims=generation_claims,
                    attempt=request_count,
                    llm_usages=usages,
                )
            except StructuredOutputError as exc:
                last_failure = self._failure_reason("structured_output_error", exc)
                logger.warning(
                    "第 %d 次文案输出未通过校验：%s",
                    request_count,
                    exc,
                )
                current_messages = (
                    *messages,
                    ChatMessage(role="assistant", content=response_content),
                    ChatMessage(
                        role="user",
                        content=(
                            f"上一次输出未通过校验：{exc} "
                            f"正文目标仍为 {task.target_length} 个非空白字符；"
                            f"允许区间是 {minimum_length} 至 {maximum_length}。"
                            "请修复后重新输出完整 JSON，不要解释。"
                        ),
                    ),
                )
                continue

            logger.info(
                "文案生成完成：%d 个非空白字符，使用 %d 个 claim",
                artifact.character_count,
                len(artifact.claim_usages),
            )
            if not self._config.grounding_review_enabled:
                return artifact
            return await self._review_artifact(
                task,
                research,
                generation_claims,
                artifact,
                usages,
            )

        raise ScriptGenerationError(
            f"Script generation failed after {request_count} request(s); "
            f"used {content_attempt_count}/{self._config.max_generation_attempts} "
            f"structured/content attempt(s). Last failure: {last_failure}",
            generation_attempt_count=request_count,
            llm_usages=usages,
        ) from None

    async def _generate_from_background_editorial(
        self,
        task: ScriptTask,
        research: ResearchOutcome,
    ) -> ScriptArtifact:
        """Generate three drafts concurrently, then synthesize one final script."""

        reference_ids = tuple(item.evidence_id for item in research.evidence)
        if not reference_ids:
            raise ScriptGenerationError(
                "Background generation requires at least one reference."
            )
        if len(set(reference_ids)) != len(reference_ids):
            raise ScriptGenerationError("Background contains duplicate reference IDs.")
        known_reference_ids = frozenset(reference_ids)

        logger.info(
            "正在并行生成三篇口播候选：目标 %d 个非空白字符",
            task.target_length,
        )
        raw_results = await asyncio.gather(
            *(
                self._generate_editorial_candidate(
                    task,
                    research,
                    candidate_id=candidate_id,
                    strategy=strategy,
                    strategy_instruction=strategy_instruction,
                    known_reference_ids=known_reference_ids,
                )
                for candidate_id, strategy, strategy_instruction in (
                    self._EDITORIAL_STRATEGIES
                )
            ),
            return_exceptions=True,
        )

        candidates: list[ScriptCandidate] = []
        usages: list[LLMCallUsage] = []
        candidate_request_count = 0
        failures: list[ScriptGenerationError] = []
        for result in raw_results:
            if isinstance(result, ScriptGenerationError):
                failures.append(result)
                usages.extend(result.llm_usages)
                candidate_request_count += result.generation_attempt_count
                continue
            if isinstance(result, BaseException):
                failures.append(ScriptGenerationError("Candidate generation failed."))
                continue
            candidate, candidate_usages, request_count = result
            candidates.append(candidate)
            usages.extend(candidate_usages)
            candidate_request_count += request_count
        if failures:
            raise ScriptGenerationError(
                "Editorial generation requires all three candidates; at least one "
                "candidate failed after bounded provider retries.",
                generation_attempt_count=candidate_request_count,
                llm_usages=usages,
            ) from None

        return await self._finalize_editorial_candidates(
            task,
            research,
            candidates,
            prior_usages=usages,
            prior_request_count=candidate_request_count,
        )

    async def edit_background_candidates(
        self,
        task: ScriptTask,
        research: ResearchOutcome,
        candidates: Sequence[ScriptCandidate],
    ) -> ScriptArtifact:
        """Re-run only the chief-editor stage over already frozen candidates."""

        if research.status != "ready" or research.claims:
            raise ScriptGenerationError(
                "Frozen-candidate editing requires ready background research."
            )
        known_reference_ids = {item.evidence_id for item in research.evidence}
        expected_candidate_ids = {item[0] for item in self._EDITORIAL_STRATEGIES}
        if (
            len(candidates) != len(expected_candidate_ids)
            or {item.candidate_id for item in candidates} != expected_candidate_ids
        ):
            raise ScriptGenerationError(
                "Frozen-candidate editing requires C01, C02, and C03 exactly once."
            )
        if any(
            not item.reference_ids
            or any(reference_id not in known_reference_ids for reference_id in item.reference_ids)
            for item in candidates
        ):
            raise ScriptGenerationError(
                "Frozen candidate references are absent from the research background."
            )
        return await self._finalize_editorial_candidates(
            task,
            research,
            candidates,
            prior_usages=(),
            prior_request_count=0,
        )

    async def _finalize_editorial_candidates(
        self,
        task: ScriptTask,
        research: ResearchOutcome,
        candidates: Sequence[ScriptCandidate],
        *,
        prior_usages: Sequence[LLMCallUsage],
        prior_request_count: int,
    ) -> ScriptArtifact:
        """Run the versioned chief-editor prompt over three validated candidates."""

        editor_messages = self._editor_messages(task, research, candidates)
        current_messages = editor_messages
        usages = list(prior_usages)
        candidate_request_count = prior_request_count
        known_reference_ids = frozenset(
            item.evidence_id for item in research.evidence
        )
        editor_request_count = 0
        consecutive_provider_failures = 0
        length_repair_attempted = False
        minimum_length, maximum_length = self._length_bounds(task)
        while True:
            editor_request_count += 1
            try:
                response = await self._complete(current_messages)
            except LLMProviderError:
                consecutive_provider_failures += 1
                if consecutive_provider_failures > self._MAX_PROVIDER_RETRIES:
                    raise ScriptGenerationError(
                        "Chief-editor generation failed after bounded provider retries.",
                        generation_attempt_count=(
                            candidate_request_count + editor_request_count
                        ),
                        llm_usages=usages,
                    ) from None
                continue

            consecutive_provider_failures = 0
            usage = llm_call_usage(
                response,
                stage="script.editor",
                attempt=editor_request_count,
            )
            usages.append(usage)
            self._log_token_usage(usage)
            try:
                (
                    outline,
                    script_text,
                    final_reference_ids,
                    selected_candidate_ids,
                ) = self._parse_editor_response(
                    response.content,
                    task=task,
                    known_reference_ids=known_reference_ids,
                    known_candidate_ids=frozenset(
                        candidate.candidate_id for candidate in candidates
                    ),
                )
            except StructuredOutputError as exc:
                current_messages = (
                    *editor_messages,
                    ChatMessage(role="assistant", content=response.content),
                    ChatMessage(
                        role="user",
                        content=(
                            f"上一次JSON或正文格式未通过校验：{exc} "
                            "请修复完整JSON。不得解释，不得改变字段；reference_ids和"
                            "selected_candidate_ids只能使用输入中存在的ID。"
                        ),
                    ),
                )
                continue

            character_count = self._count_characters(script_text)
            length_within_tolerance = (
                minimum_length <= character_count <= maximum_length
            )
            if not length_within_tolerance and not length_repair_attempted:
                length_repair_attempted = True
                if character_count < minimum_length:
                    length_instruction = (
                        f"至少还需增加{minimum_length - character_count}个非空白字符，"
                        f"建议接近目标值再增加约{task.target_length - character_count}个。"
                        "只能展开候选里已有但最终稿尚未讲清的机制、取舍或普通人影响，"
                        "不得同义重复。"
                    )
                else:
                    length_instruction = (
                        f"至少需要删减{character_count - maximum_length}个非空白字符，"
                        f"建议接近目标值共删减约{character_count - task.target_length}个。"
                        "优先删除重复判断、过密例子和不影响结论的修饰，不能删掉安全边界。"
                    )
                current_messages = (
                    *editor_messages,
                    ChatMessage(role="assistant", content=response.content),
                    ChatMessage(
                        role="user",
                        content=(
                            f"这次JSON和正文格式已经合格，但正文实际为{character_count}个"
                            f"非空白字符，目标允许区间为{minimum_length}至{maximum_length}。"
                            f"{length_instruction}"
                            "这是唯一一次字数修复机会：保持核心判断、关键信息、自然口语、"
                            "引用ID和候选选择不变，把正文调整到允许区间并重新输出完整JSON。"
                            "不要解释。"
                        ),
                    ),
                )
                continue

            logger.info(
                "主编定稿完成：%d 个非空白字符，候选请求=%d，主编请求=%d",
                character_count,
                candidate_request_count,
                editor_request_count,
            )
            return ScriptArtifact(
                outline=outline,
                script_text=script_text,
                claim_usages=(),
                character_count=character_count,
                prompt_version=BACKGROUND_SCRIPT_PIPELINE_VERSION,
                generation_attempt_count=(
                    candidate_request_count + editor_request_count
                ),
                llm_usages=tuple(usages),
                reference_ids=final_reference_ids,
                generation_mode="editorial_candidates",
                generation_candidates=tuple(candidates),
                selected_candidate_ids=selected_candidate_ids,
                editor_prompt_version=BACKGROUND_SCRIPT_EDITOR_PROMPT_VERSION,
                editor_attempt_count=editor_request_count,
                length_within_tolerance=length_within_tolerance,
                length_repair_attempted=length_repair_attempted,
            )

    async def _generate_editorial_candidate(
        self,
        task: ScriptTask,
        research: ResearchOutcome,
        *,
        candidate_id: str,
        strategy: str,
        strategy_instruction: str,
        known_reference_ids: frozenset[str],
    ) -> tuple[ScriptCandidate, tuple[LLMCallUsage, ...], int]:
        messages = self._candidate_messages(
            task,
            research,
            candidate_id=candidate_id,
            strategy_instruction=strategy_instruction,
        )
        current_messages = messages
        request_count = 0
        consecutive_provider_failures = 0
        usages: list[LLMCallUsage] = []
        while True:
            request_count += 1
            try:
                response = await self._complete(current_messages)
            except LLMProviderError:
                consecutive_provider_failures += 1
                if consecutive_provider_failures > self._MAX_PROVIDER_RETRIES:
                    raise ScriptGenerationError(
                        f"Candidate {candidate_id} failed after bounded provider retries.",
                        generation_attempt_count=request_count,
                        llm_usages=usages,
                    ) from None
                continue

            consecutive_provider_failures = 0
            usage = llm_call_usage(
                response,
                stage=f"script.candidate.{strategy}",
                attempt=request_count,
            )
            usages.append(usage)
            self._log_token_usage(usage)
            try:
                candidate = self._parse_candidate_response(
                    response.content,
                    task=task,
                    candidate_id=candidate_id,
                    strategy=strategy,
                    known_reference_ids=known_reference_ids,
                )
            except StructuredOutputError as exc:
                current_messages = (
                    *messages,
                    ChatMessage(role="assistant", content=response.content),
                    ChatMessage(
                        role="user",
                        content=(
                            f"上一次JSON或正文格式未通过校验：{exc} "
                            "请修复完整JSON，不要解释。reference_ids只能使用输入中存在的ID。"
                            "候选稿字数不触发重写；不要因为字数自行增加或删除信息。"
                        ),
                    ),
                )
                continue
            return candidate, tuple(usages), request_count

    def _candidate_messages(
        self,
        task: ScriptTask,
        research: ResearchOutcome,
        *,
        candidate_id: str,
        strategy_instruction: str,
    ) -> tuple[ChatMessage, ...]:
        schema = {
            "outline": ["正文结构要点，不会出现在正文中"],
            "script_text": "可直接朗读的纯正文",
            "reference_ids": ["E001", "E002"],
        }
        payload = self._background_payload(task, research)
        prompt = (
            f"候选ID：{candidate_id}。本稿必须采用以下策略：{strategy_instruction}"
            f"正文目标为{task.target_length}个非空白字符，允许偏差"
            f"{self._format_percentage(self._config.length_tolerance_ratio)}。"
            f"篇幅结构：{self._length_guidance(task.target_length)}"
            "先在心中默读并调整气口，但不要输出检查过程。reference_ids至少选择一个真正"
            "帮助本稿的来源，且只作为正文外元数据。\n"
            f"输出结构：{json.dumps(schema, ensure_ascii=False)}\n"
            "以下JSON是任务和搜索背景：\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        return (
            ChatMessage(
                role="system",
                content=BACKGROUND_SCRIPT_CANDIDATE_SYSTEM_PROMPT,
            ),
            ChatMessage(role="user", content=prompt),
        )

    def _editor_messages(
        self,
        task: ScriptTask,
        research: ResearchOutcome,
        candidates: Sequence[ScriptCandidate],
    ) -> tuple[ChatMessage, ...]:
        schema = {
            "outline": ["最终正文结构要点，不会出现在正文中"],
            "script_text": "最终可直接朗读的纯正文",
            "reference_ids": ["E001", "E002"],
            "selected_candidate_ids": ["C01", "C02"],
        }
        payload = {
            **self._background_payload(task, research),
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "strategy": candidate.strategy,
                    "outline": list(candidate.outline),
                    "script_text": candidate.script_text,
                    "reference_ids": list(candidate.reference_ids),
                    "character_count": candidate.character_count,
                }
                for candidate in candidates
            ],
        }
        prompt = (
            f"最终正文目标为{task.target_length}个非空白字符，允许偏差"
            f"{self._format_percentage(self._config.length_tolerance_ratio)}。"
            f"篇幅结构：{self._length_guidance(task.target_length)}"
            "请在三篇候选中择优并重新定稿，不要平均拼接。先完整默读一遍最终稿，确认中段"
            "仍像真人说话。reference_ids和selected_candidate_ids都必须至少包含一个输入中的ID。\n"
            f"输出结构：{json.dumps(schema, ensure_ascii=False)}\n"
            "以下JSON是任务、搜索背景和三篇候选：\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        return (
            ChatMessage(role="system", content=BACKGROUND_SCRIPT_EDITOR_SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt),
        )

    def _background_payload(
        self,
        task: ScriptTask,
        research: ResearchOutcome,
    ) -> dict[str, object]:
        return {
            "task": asdict(task),
            "research_goal": research.query_plan.goal,
            "background_references": [
                {
                    "reference_id": item.evidence_id,
                    "title": item.title,
                    "url": item.url,
                    "background_excerpt": item.excerpt,
                    "source_query": item.source_query,
                    "published_at": item.published_at,
                }
                for item in research.evidence
            ],
        }

    def _parse_candidate_response(
        self,
        response: str,
        *,
        task: ScriptTask,
        candidate_id: str,
        strategy: str,
        known_reference_ids: frozenset[str],
    ) -> ScriptCandidate:
        payload = json_object(response)
        if set(payload) != {"outline", "script_text", "reference_ids"}:
            raise StructuredOutputError(
                "Candidate response must contain exactly outline, script_text, "
                "and reference_ids."
            )
        outline = text_list(
            payload,
            "outline",
            minimum=1,
            maximum=self._MAX_OUTLINE_ITEMS,
            item_max_length=200,
        )
        script_text = required_text(
            payload,
            "script_text",
            max_length=self._MAX_SCRIPT_LENGTH,
        )
        self._validate_body(script_text, task)
        reference_ids = self._parse_reference_ids(
            payload.get("reference_ids"),
            known_reference_ids=known_reference_ids,
        )
        return ScriptCandidate(
            candidate_id=candidate_id,
            strategy=strategy,
            outline=outline,
            script_text=script_text,
            reference_ids=reference_ids,
            character_count=self._count_characters(script_text),
            prompt_version=BACKGROUND_SCRIPT_CANDIDATE_PROMPT_VERSION,
        )

    def _parse_editor_response(
        self,
        response: str,
        *,
        task: ScriptTask,
        known_reference_ids: frozenset[str],
        known_candidate_ids: frozenset[str],
    ) -> tuple[tuple[str, ...], str, tuple[str, ...], tuple[str, ...]]:
        payload = json_object(response)
        if set(payload) != {
            "outline",
            "script_text",
            "reference_ids",
            "selected_candidate_ids",
        }:
            raise StructuredOutputError(
                "Editor response must contain exactly outline, script_text, "
                "reference_ids, and selected_candidate_ids."
            )
        outline = text_list(
            payload,
            "outline",
            minimum=1,
            maximum=self._MAX_OUTLINE_ITEMS,
            item_max_length=200,
        )
        script_text = required_text(
            payload,
            "script_text",
            max_length=self._MAX_SCRIPT_LENGTH,
        )
        self._validate_body(script_text, task)
        reference_ids = self._parse_reference_ids(
            payload.get("reference_ids"),
            known_reference_ids=known_reference_ids,
        )
        selected_candidate_ids = self._parse_identifier_list(
            payload.get("selected_candidate_ids"),
            known_ids=known_candidate_ids,
            label="selected_candidate_ids",
        )
        return outline, script_text, reference_ids, selected_candidate_ids

    @staticmethod
    def _parse_identifier_list(
        raw_values: object,
        *,
        known_ids: frozenset[str],
        label: str,
    ) -> tuple[str, ...]:
        if (
            not isinstance(raw_values, list)
            or not raw_values
            or len(raw_values) > len(known_ids)
        ):
            raise StructuredOutputError(f"Response contains invalid {label}.")
        normalized: list[str] = []
        for value in raw_values:
            if not isinstance(value, str) or not value.strip():
                raise StructuredOutputError(f"Response contains an invalid {label} ID.")
            identifier = value.strip()
            if identifier not in known_ids:
                raise StructuredOutputError(f"Response contains an unknown {label} ID.")
            if identifier in normalized:
                raise StructuredOutputError(f"Response contains duplicate {label} IDs.")
            normalized.append(identifier)
        return tuple(normalized)

    @classmethod
    def _parse_reference_ids(
        cls,
        raw_values: object,
        *,
        known_reference_ids: frozenset[str],
    ) -> tuple[str, ...]:
        return cls._parse_identifier_list(
            raw_values,
            known_ids=known_reference_ids,
            label="reference_ids",
        )

    async def _generate_from_background(
        self,
        task: ScriptTask,
        research: ResearchOutcome,
    ) -> ScriptArtifact:
        """Write for audience quality while retaining separate source metadata."""

        reference_ids = tuple(item.evidence_id for item in research.evidence)
        if not reference_ids:
            raise ScriptGenerationError(
                "Background generation requires at least one reference."
            )
        if len(set(reference_ids)) != len(reference_ids):
            raise ScriptGenerationError("Background contains duplicate reference IDs.")

        messages = self._background_messages(task, research)
        minimum_length, maximum_length = self._length_bounds(task)
        current_messages = messages
        usages: list[LLMCallUsage] = []
        last_failure = "unknown failure"
        request_count = 0
        content_attempt_count = 0
        consecutive_provider_failures = 0
        max_request_count = (
            self._config.max_generation_attempts + self._MAX_PROVIDER_RETRIES
        )
        logger.info(
            "正在根据背景生成口播文案：目标 %d 个非空白字符",
            task.target_length,
        )
        while (
            content_attempt_count < self._config.max_generation_attempts
            and request_count < max_request_count
        ):
            request_count += 1
            try:
                response = await self._complete(current_messages)
            except LLMProviderError as exc:
                consecutive_provider_failures += 1
                last_failure = self._failure_reason("provider_error", exc)
                if consecutive_provider_failures > self._MAX_PROVIDER_RETRIES:
                    break
                continue

            consecutive_provider_failures = 0
            content_attempt_count += 1
            usage = llm_call_usage(
                response,
                stage="script.generation",
                attempt=request_count,
            )
            usages.append(usage)
            self._log_token_usage(usage)
            try:
                return self._parse_background_artifact(
                    response.content,
                    task=task,
                    known_reference_ids=frozenset(reference_ids),
                    attempt=request_count,
                    llm_usages=usages,
                )
            except StructuredOutputError as exc:
                last_failure = self._failure_reason("structured_output_error", exc)
                current_messages = (
                    *messages,
                    ChatMessage(role="assistant", content=response.content),
                    ChatMessage(
                        role="user",
                        content=(
                            f"上一次输出未通过校验：{exc} "
                            f"正文目标仍为 {task.target_length} 个非空白字符；"
                            f"允许区间是 {minimum_length} 至 {maximum_length}。"
                            "请修复完整 JSON。reference_ids 只保留输入中存在的来源 ID，"
                            "不要把引用写入正文，不要解释。"
                        ),
                    ),
                )

        raise ScriptGenerationError(
            f"Script generation failed after {request_count} request(s); "
            f"used {content_attempt_count}/{self._config.max_generation_attempts} "
            f"structured/content attempt(s). Last failure: {last_failure}",
            generation_attempt_count=request_count,
            llm_usages=usages,
        ) from None

    def _background_messages(
        self,
        task: ScriptTask,
        research: ResearchOutcome,
    ) -> tuple[ChatMessage, ...]:
        schema = {
            "outline": ["正文结构要点，不会出现在正文中"],
            "script_text": "可直接朗读的纯正文",
            "reference_ids": ["E001", "E002"],
        }
        references = [
            {
                "reference_id": item.evidence_id,
                "title": item.title,
                "url": item.url,
                "background_excerpt": item.excerpt,
                "source_query": item.source_query,
                "published_at": item.published_at,
            }
            for item in research.evidence
        ]
        payload = {
            "task": asdict(task),
            "research_goal": research.query_plan.goal,
            "background_references": references,
        }
        prompt = (
            f"正文目标为 {task.target_length} 个非空白字符，允许偏差 "
            f"{self._format_percentage(self._config.length_tolerance_ratio)}。"
            f"篇幅结构：{self._length_guidance(task.target_length)}"
            "背景资料帮助你理解和选材，不要求逐句证明，也不要在正文显示引用。"
            "reference_ids 至少选择一个真正帮助成稿的来源；这些 ID 只作为独立元数据供后续评分。\n"
            f"输出结构：{json.dumps(schema, ensure_ascii=False)}\n"
            "以下 JSON 是任务和搜索背景：\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        return (
            ChatMessage(
                role="system",
                content=BACKGROUND_SCRIPT_GENERATION_SYSTEM_PROMPT,
            ),
            ChatMessage(role="user", content=prompt),
        )

    def _parse_background_artifact(
        self,
        response: str,
        *,
        task: ScriptTask,
        known_reference_ids: frozenset[str],
        attempt: int,
        llm_usages: Sequence[LLMCallUsage],
    ) -> ScriptArtifact:
        payload = json_object(response)
        if set(payload) != {"outline", "script_text", "reference_ids"}:
            raise StructuredOutputError(
                "Background script response must contain exactly outline, "
                "script_text, and reference_ids."
            )
        outline = text_list(
            payload,
            "outline",
            minimum=1,
            maximum=self._MAX_OUTLINE_ITEMS,
            item_max_length=200,
        )
        script_text = required_text(
            payload,
            "script_text",
            max_length=self._MAX_SCRIPT_LENGTH,
        )
        character_count = self._count_characters(script_text)
        minimum_length, maximum_length = self._length_bounds(task)
        if not minimum_length <= character_count <= maximum_length:
            raise StructuredOutputError(
                "script_text is outside the configured target-length tolerance: "
                f"actual={character_count}, allowed={minimum_length}..{maximum_length}."
            )
        self._validate_body(script_text, task)
        raw_reference_ids = payload.get("reference_ids")
        if (
            not isinstance(raw_reference_ids, list)
            or not raw_reference_ids
            or len(raw_reference_ids) > len(known_reference_ids)
        ):
            raise StructuredOutputError("Response contains invalid reference_ids.")
        normalized_reference_ids: list[str] = []
        for value in raw_reference_ids:
            if not isinstance(value, str) or not value.strip():
                raise StructuredOutputError("Response contains an invalid reference ID.")
            reference_id = value.strip()
            if reference_id not in known_reference_ids:
                raise StructuredOutputError(
                    "Response references a source absent from the background."
                )
            if reference_id in normalized_reference_ids:
                raise StructuredOutputError("Response contains duplicate reference IDs.")
            normalized_reference_ids.append(reference_id)
        return ScriptArtifact(
            outline=outline,
            script_text=script_text,
            claim_usages=(),
            character_count=character_count,
            prompt_version=BACKGROUND_SCRIPT_GENERATION_PROMPT_VERSION,
            generation_attempt_count=attempt,
            llm_usages=tuple(llm_usages),
            reference_ids=tuple(normalized_reference_ids),
        )

    async def _review_artifact(
        self,
        task: ScriptTask,
        research: ResearchOutcome,
        supported_claims: Sequence[Claim],
        draft: ScriptArtifact,
        usages: list[LLMCallUsage],
    ) -> ScriptArtifact:
        """Run a bounded evidence review, retaining explicit review outcome."""

        logger.info("[5/5] 正在校对文案事实边界和重复表达")
        review_messages = self._review_messages(
            task,
            research,
            supported_claims,
            draft,
        )
        current_messages = review_messages
        failure_reason: str | None = None
        request_count = 0
        content_attempt_count = 0
        consecutive_provider_failures = 0
        max_request_count = (
            self._MAX_GROUNDING_REVIEW_ATTEMPTS + self._MAX_PROVIDER_RETRIES
        )
        while (
            content_attempt_count < self._MAX_GROUNDING_REVIEW_ATTEMPTS
            and request_count < max_request_count
        ):
            request_count += 1
            response_content: str | None = None
            try:
                response = await self._complete(current_messages)
            except LLMProviderError as exc:
                consecutive_provider_failures += 1
                failure_reason = self._failure_reason("provider_error", exc)
                logger.warning(
                    "事实边界校对第 %d 次请求失败：%s",
                    request_count,
                    exc,
                )
                if consecutive_provider_failures > self._MAX_PROVIDER_RETRIES:
                    break
                current_messages = self._repair_messages(
                    review_messages,
                    response_content=None,
                    failure_reason=failure_reason,
                )
                continue

            consecutive_provider_failures = 0
            content_attempt_count += 1
            response_content = response.content
            usage = llm_call_usage(
                response,
                stage="script.grounding_review",
                attempt=request_count,
            )
            usages.append(usage)
            self._log_token_usage(usage)
            try:
                reviewed, review_issues = self._parse_review_response(
                    response_content,
                    task=task,
                    supported_claims=supported_claims,
                    generation_attempt_count=draft.generation_attempt_count,
                    llm_usages=usages,
                )
            except StructuredOutputError as exc:
                failure_reason = self._failure_reason("structured_output_error", exc)
                logger.warning(
                    "事实边界校对第 %d 次未通过本地校验：%s",
                    request_count,
                    exc,
                )
            else:
                if reviewed is None:
                    logger.warning(
                        "事实边界校对拒绝草稿，需返回研究或重新生成：%s",
                        "；".join(review_issues),
                    )
                    return replace(
                        draft,
                        llm_usages=tuple(usages),
                        grounding_review_attempt_count=request_count,
                        grounding_review_status="rejected",
                        grounding_review_prompt_version=(
                            SCRIPT_GROUNDING_REVIEW_PROMPT_VERSION
                        ),
                        grounding_review_draft_text=draft.script_text,
                        grounding_review_draft_character_count=draft.character_count,
                        grounding_review_issues=review_issues,
                        grounding_review_failure_reason="review_rejected",
                    )
                logger.info(
                    "事实边界校对完成：%d 个非空白字符",
                    reviewed.character_count,
                )
                return replace(
                    reviewed,
                    grounding_review_attempt_count=request_count,
                    grounding_review_status="accepted",
                    grounding_review_prompt_version=(
                        SCRIPT_GROUNDING_REVIEW_PROMPT_VERSION
                    ),
                    grounding_review_draft_text=draft.script_text,
                    grounding_review_draft_character_count=draft.character_count,
                    grounding_review_issues=(),
                    grounding_review_failure_reason=None,
                )

            if (
                content_attempt_count < self._MAX_GROUNDING_REVIEW_ATTEMPTS
                and request_count < max_request_count
            ):
                current_messages = self._repair_messages(
                    review_messages,
                    response_content=response_content,
                    failure_reason=failure_reason or "unknown failure",
                )
                continue
            break
        return replace(
            draft,
            llm_usages=tuple(usages),
            grounding_review_attempt_count=request_count,
            grounding_review_status="fallback",
            grounding_review_prompt_version=(
                SCRIPT_GROUNDING_REVIEW_PROMPT_VERSION
            ),
            grounding_review_draft_text=draft.script_text,
            grounding_review_draft_character_count=draft.character_count,
            grounding_review_issues=(),
            grounding_review_failure_reason=failure_reason,
        )

    @staticmethod
    def _repair_messages(
        messages: Sequence[ChatMessage],
        *,
        response_content: str | None,
        failure_reason: str,
    ) -> tuple[ChatMessage, ...]:
        previous_output = (
            (ChatMessage(role="assistant", content=response_content),)
            if response_content is not None
            else ()
        )
        return (
            *messages,
            *previous_output,
            ChatMessage(
                role="user",
                content=(
                    f"上一次校对失败：{failure_reason}。"
                    "请基于原始草稿和证据修复后重新输出完整 JSON，不要解释。"
                ),
            ),
        )

    @staticmethod
    def _failure_reason(kind: str, exc: Exception) -> str:
        detail = " ".join(str(exc).split()) or "no detail"
        return f"{kind}: {detail}"[:300]

    def _messages(
        self,
        task: ScriptTask,
        research: ResearchOutcome,
        supported_claims: Sequence[Claim],
    ) -> tuple[ChatMessage, ...]:
        schema = {
            "outline": ["正文结构要点，不是要输出到正文里的小标题"],
            "script_text": "可直接朗读的纯正文",
            "claim_usages": [
                {
                    "claim_id": "C001",
                    "script_quote": "script_text 中逐字出现的短句",
                }
            ],
        }
        referenced_evidence_ids = {
            evidence_id
            for claim in supported_claims
            for evidence_id in claim.evidence_ids
        }
        payload = {
            "task": asdict(task),
            "research_goal": research.query_plan.goal,
            "title_chain": [asdict(item) for item in research.title_chain],
            "supported_claims": [asdict(claim) for claim in supported_claims],
            "evidence": [
                asdict(item)
                for item in research.evidence
                if item.evidence_id in referenced_evidence_ids
            ],
        }
        length_guidance = self._length_guidance(task.target_length)
        prompt = (
            f"正文目标为 {task.target_length} 个非空白字符，允许偏差 "
            f"{self._format_percentage(self._config.length_tolerance_ratio)}。"
            f"篇幅结构：{length_guidance}"
            "每个 is_core=true 的 claim 必须且只能在 claim_usages 中出现一次；"
            "这不要求照抄 claim 文本，可以用更自然的口语准确转述。非核心 claim 只有在能推进"
            "主线且篇幅允许时才使用，不要为了覆盖材料而堆入正文；没有写入正文就不要添加 usage。"
            "title_chain 是待复核审计，不是额外证据；逐项用其中 claim_ids 对应的 claim 和 excerpt"
            "核实，reason 不能补足缺失链路。若 covered 判断与原文不符，不得沿用该判断。"
            "script_quote 必须是正文中的逐字短句。\n"
            f"输出结构：{json.dumps(schema, ensure_ascii=False)}\n"
            "以下 JSON 是已经过程序校验的任务、候选论断和证据数据：\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        return (
            ChatMessage(role="system", content=SCRIPT_GENERATION_SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt),
        )

    def _review_messages(
        self,
        task: ScriptTask,
        research: ResearchOutcome,
        supported_claims: Sequence[Claim],
        draft: ScriptArtifact,
    ) -> tuple[ChatMessage, ...]:
        schema = {
            "decision": "accepted | rejected",
            "issues": [
                "问题码: 只有 rejected 时填写的具体且可操作原因"
            ],
            "outline": ["修订后的正文结构要点"],
            "script_text": "校对后可直接朗读的纯正文",
            "claim_usages": [
                {
                    "claim_id": "C001",
                    "script_quote": "修订后 script_text 中逐字出现的短句",
                }
            ],
            "clause_audit": [
                {
                    "script_quote": "修订后的完整句子或分句，包含结尾标点。",
                    "kind": "D | I | A",
                    "claim_ids": ["C001"],
                }
            ],
        }
        referenced_evidence_ids = {
            evidence_id
            for claim in supported_claims
            for evidence_id in claim.evidence_ids
        }
        payload = {
            "task": asdict(task),
            "research_goal": research.query_plan.goal,
            "title_chain": [asdict(item) for item in research.title_chain],
            "supported_claims": [asdict(claim) for claim in supported_claims],
            "evidence": [
                asdict(item)
                for item in research.evidence
                if item.evidence_id in referenced_evidence_ids
            ],
            "draft": {
                "outline": list(draft.outline),
                "script_text": draft.script_text,
                "claim_usages": [asdict(item) for item in draft.claim_usages],
            },
        }
        prompt = (
            f"正文仍须保持 {task.target_length} 个非空白字符，允许偏差 "
            f"{self._format_percentage(self._config.length_tolerance_ratio)}。"
            "先在内部逐句核对事实边界和重复表达，再只输出完整 JSON。能够修复且通过时令 "
            "decision=accepted、issues=[]，并输出修订稿；若缺少关键证据、无法在字数内无重复地"
            "保留核心论断，令 decision=rejected，并至少填写一项 issues。每项 issue 格式必须是"
            "‘问题码: 具体原因’，问题码只能取 unsupported_claim、scope_expansion、causal_leap、"
            "source_attribution、current_rule_gap、unsupported_advice、repetition、oral_style、"
            "insufficient_evidence、other。rejected 时正文相关字段可原样返回草稿。不得加入任何"
            "新事实。title_chain 的 reason 不是证据；必须核对每个 component 的 claim_ids 是否由"
            "对应 excerpt 直接覆盖，任何语义相邻替代都应 rejected。accepted 时 clause_audit 必须"
            "按‘。！？；!?;’切分后的原顺序"
            "逐项覆盖修订稿；D 的 claim_ids 必须与 claim_usages 完全一致。rejected 不需要"
            "为无法通过的草稿制造完整审计，可返回 clause_audit=[]。\n"
            f"输出结构：{json.dumps(schema, ensure_ascii=False)}\n"
            "以下 JSON 是待校对数据：\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        return (
            ChatMessage(
                role="system",
                content=SCRIPT_GROUNDING_REVIEW_SYSTEM_PROMPT,
            ),
            ChatMessage(role="user", content=prompt),
        )

    def _parse_review_response(
        self,
        response: str,
        *,
        task: ScriptTask,
        supported_claims: Sequence[Claim],
        generation_attempt_count: int,
        llm_usages: Sequence[LLMCallUsage],
    ) -> tuple[ScriptArtifact | None, tuple[str, ...]]:
        payload = json_object(response)
        base_fields = {
            "decision",
            "issues",
            "outline",
            "script_text",
            "claim_usages",
        }
        allowed_fields = base_fields | {"clause_audit"}
        if not base_fields.issubset(payload) or set(payload) - allowed_fields:
            raise StructuredOutputError(
                "Grounding review must contain exactly the required fields."
            )
        decision = required_text(payload, "decision", max_length=16)
        if decision not in {"accepted", "rejected"}:
            raise StructuredOutputError("Grounding review decision is invalid.")
        issues = text_list(
            payload,
            "issues",
            minimum=0,
            maximum=self._MAX_REVIEW_ISSUES,
            item_max_length=300,
        )
        for issue in issues:
            code, separator, detail = issue.partition(":")
            if (
                not separator
                or code.strip() not in self._REVIEW_ISSUE_CODES
                or not detail.strip()
            ):
                raise StructuredOutputError(
                    "Grounding review issue must use an allowed code and detail."
                )
        if decision == "accepted" and issues:
            raise StructuredOutputError(
                "An accepted grounding review must not contain issues."
            )
        if decision == "rejected":
            if not issues:
                raise StructuredOutputError(
                    "A rejected grounding review requires at least one issue."
                )
            raw_audit = payload.get("clause_audit", [])
            if not isinstance(raw_audit, list):
                raise StructuredOutputError(
                    "Rejected grounding review clause_audit must be a list."
                )
            return None, issues

        if "clause_audit" not in payload:
            raise StructuredOutputError(
                "An accepted grounding review requires clause_audit."
            )

        artifact_payload = {
            "outline": payload["outline"],
            "script_text": payload["script_text"],
            "claim_usages": payload["claim_usages"],
        }
        reviewed = self._parse_artifact(
            json.dumps(artifact_payload, ensure_ascii=False),
            task=task,
            supported_claims=supported_claims,
            attempt=generation_attempt_count,
            llm_usages=llm_usages,
        )
        self._validate_clause_audit(
            payload["clause_audit"],
            script_text=reviewed.script_text,
            supported_claims=supported_claims,
            claim_usages=reviewed.claim_usages,
        )
        return reviewed, ()

    @classmethod
    def _validate_clause_audit(
        cls,
        raw_audit: object,
        *,
        script_text: str,
        supported_claims: Sequence[Claim],
        claim_usages: Sequence[ClaimUsage],
    ) -> None:
        expected_clauses = cls._audit_clauses(script_text)
        if not isinstance(raw_audit, list) or len(raw_audit) != len(expected_clauses):
            raise StructuredOutputError(
                "Accepted grounding review clause_audit must cover every clause exactly once."
            )

        supported_ids = {claim.claim_id for claim in supported_claims}
        audited_claim_ids: set[str] = set()
        audited_clauses_by_claim: dict[str, list[str]] = {}
        for index, (raw_item, expected_quote) in enumerate(
            zip(raw_audit, expected_clauses, strict=True),
            start=1,
        ):
            if not isinstance(raw_item, dict) or set(raw_item) != {
                "script_quote",
                "kind",
                "claim_ids",
            }:
                raise StructuredOutputError(
                    f"Grounding review clause_audit item {index} is invalid."
                )
            script_quote = required_text(
                raw_item,
                "script_quote",
                max_length=cls._MAX_SCRIPT_LENGTH,
            )
            if script_quote != expected_quote:
                raise StructuredOutputError(
                    "Grounding review clause_audit is missing, duplicated, or out of order."
                )
            kind = required_text(raw_item, "kind", max_length=1)
            if kind not in cls._CLAUSE_AUDIT_KINDS:
                raise StructuredOutputError(
                    "Grounding review clause_audit contains an invalid kind."
                )
            raw_claim_ids = raw_item.get("claim_ids")
            if not isinstance(raw_claim_ids, list):
                raise StructuredOutputError(
                    "Grounding review clause_audit claim_ids must be a list."
                )
            claim_ids: list[str] = []
            for raw_claim_id in raw_claim_ids:
                if (
                    not isinstance(raw_claim_id, str)
                    or not raw_claim_id.strip()
                    or len(raw_claim_id.strip()) > 64
                ):
                    raise StructuredOutputError(
                        "Grounding review clause_audit contains an invalid claim ID."
                    )
                claim_id = raw_claim_id.strip()
                if claim_id in claim_ids:
                    raise StructuredOutputError(
                        "Grounding review clause_audit contains duplicate claim IDs."
                    )
                if claim_id not in supported_ids:
                    raise StructuredOutputError(
                        "Grounding review clause_audit references an unknown or unsupported claim."
                    )
                claim_ids.append(claim_id)

            if kind == "D" and not claim_ids:
                raise StructuredOutputError(
                    "Every D clause must reference at least one supported claim."
                )
            if kind != "D" and claim_ids:
                raise StructuredOutputError(
                    "I and A clauses must not reference claims."
                )
            for claim_id in claim_ids:
                if claim_id in audited_claim_ids:
                    raise StructuredOutputError(
                        "A claim may appear in only one audited D clause."
                    )
                audited_claim_ids.add(claim_id)
                audited_clauses_by_claim.setdefault(claim_id, []).append(script_quote)

        usage_ids = {usage.claim_id for usage in claim_usages}
        if audited_claim_ids != usage_ids:
            raise StructuredOutputError(
                "Grounding review D claim IDs must match claim_usages exactly."
            )
        for usage in claim_usages:
            matching_clauses = audited_clauses_by_claim[usage.claim_id]
            if not any(
                usage.script_quote in clause or clause in usage.script_quote
                for clause in matching_clauses
            ):
                raise StructuredOutputError(
                    "Grounding review claim usage is detached from its audited D clause."
                )

    @staticmethod
    def _audit_clauses(script_text: str) -> tuple[str, ...]:
        clauses: list[str] = []
        start = 0
        for boundary in _CLAUSE_BOUNDARY_PATTERN.finditer(script_text):
            clause = script_text[start : boundary.end()].strip()
            if clause:
                clauses.append(clause)
            start = boundary.end()
        trailing = script_text[start:].strip()
        if trailing:
            clauses.append(trailing)
        return tuple(clauses)

    @staticmethod
    def _format_percentage(value: float) -> str:
        return f"{value * 100:g}%"

    @staticmethod
    def _claim_has_eligible_source(
        claim: Claim,
        evidence_by_id: dict[str, Evidence],
    ) -> bool:
        """Require one referenced source strong enough for the claim kind."""

        allowed_types = CORE_SOURCE_TYPES_BY_CLAIM_KIND[claim.claim_kind]
        return any(
            evidence_by_id[evidence_id].source_type in allowed_types
            for evidence_id in claim.evidence_ids
        )

    @staticmethod
    def _validate_research_title_chain(
        research: ResearchOutcome,
        supported_claims: Sequence[Claim],
    ) -> None:
        """Require current research outputs to preserve a complete ready chain."""

        expected_components = (
            "subject_scope",
            "stated_context",
            "question_predicate",
        )
        if (
            research.evidence_prompt_version == RESEARCH_EVIDENCE_PROMPT_VERSION
            and not research.title_chain
        ):
            raise ScriptGenerationError(
                "Current research output is missing the persisted title chain."
            )
        if not research.title_chain:
            return
        if tuple(part.component for part in research.title_chain) != expected_components:
            raise ScriptGenerationError(
                "Research title chain must contain each required component exactly once."
            )
        supported_core_ids = {
            claim.claim_id for claim in supported_claims if claim.is_core
        }
        for part in research.title_chain:
            if part.status != "covered":
                raise ScriptGenerationError(
                    "Ready research title chain contains a missing component."
                )
            if not part.claim_ids or any(
                claim_id not in supported_core_ids for claim_id in part.claim_ids
            ):
                raise ScriptGenerationError(
                    "Research title chain references a non-core or unsupported claim."
                )

    @staticmethod
    def _length_guidance(target_length: int) -> str:
        if target_length <= 320:
            return (
                "短稿只保留一个核心判断、最多两个关键依据和一个具体提醒；开头后立即回答，"
                "删除品牌或机构清单，核心论断之外最多使用一条非核心事实。"
            )
        if target_length <= 550:
            return (
                "中篇按“问题判断—关键机制或取舍—普通人影响”展开，每部分只保留最强证据，"
                "核心论断之外最多使用两条非核心事实，结尾不要复述开头。"
            )
        non_core_limit = 3 if target_length <= 900 else 4
        return (
            "长稿按三个层次推进因果或利弊，每层提供不同信息；主动解释反例与不确定性，但同一"
            f"边界只说一次，核心论断之外最多使用{non_core_limit}条非核心事实，不用重复结论填字数。"
        )

    def _parse_artifact(
        self,
        response: str,
        *,
        task: ScriptTask,
        supported_claims: Sequence[Claim],
        attempt: int,
        llm_usages: Sequence[LLMCallUsage],
    ) -> ScriptArtifact:
        payload = json_object(response)
        unknown_fields = set(payload) - {"outline", "script_text", "claim_usages"}
        if unknown_fields:
            raise StructuredOutputError("Response contains unexpected top-level fields.")
        outline = text_list(
            payload,
            "outline",
            minimum=1,
            maximum=self._MAX_OUTLINE_ITEMS,
            item_max_length=200,
        )
        script_text = required_text(
            payload,
            "script_text",
            max_length=self._MAX_SCRIPT_LENGTH,
        )
        character_count = self._count_characters(script_text)
        minimum_length, maximum_length = self._length_bounds(task)
        if not minimum_length <= character_count <= maximum_length:
            raise StructuredOutputError(
                "script_text is outside the configured target-length tolerance: "
                f"actual={character_count}, allowed={minimum_length}..{maximum_length}."
            )
        self._validate_body(script_text, task)
        claim_usages = self._parse_claim_usages(
            payload,
            script_text=script_text,
            supported_claims=supported_claims,
        )
        return ScriptArtifact(
            outline=outline,
            script_text=script_text,
            claim_usages=claim_usages,
            character_count=character_count,
            prompt_version=SCRIPT_GENERATION_PROMPT_VERSION,
            generation_attempt_count=attempt,
            llm_usages=tuple(llm_usages),
        )

    @staticmethod
    def _log_token_usage(usage: LLMCallUsage) -> None:
        logger.info(
            "Hy3 usage：stage=%s，input=%s，output=%s，total=%s",
            usage.stage,
            usage.input_tokens if usage.input_tokens is not None else "unknown",
            usage.output_tokens if usage.output_tokens is not None else "unknown",
            usage.total_tokens if usage.total_tokens is not None else "unknown",
        )

    async def _complete(
        self,
        messages: Sequence[ChatMessage],
    ):
        """Run one Hy3 request under the optional batch-wide request limit."""

        if self._request_semaphore is None:
            return await self._llm.complete(
                messages,
                reasoning_effort="high",
            )
        async with self._request_semaphore:
            return await self._llm.complete(
                messages,
                reasoning_effort="high",
            )

    @classmethod
    def _parse_claim_usages(
        cls,
        payload: dict[str, object],
        *,
        script_text: str,
        supported_claims: Sequence[Claim],
    ) -> tuple[ClaimUsage, ...]:
        raw_usages = payload.get("claim_usages")
        if not isinstance(raw_usages, list) or len(raw_usages) > len(supported_claims):
            raise StructuredOutputError("Response contains invalid claim_usages.")

        supported_map = {claim.claim_id: claim for claim in supported_claims}
        usages: list[ClaimUsage] = []
        seen_ids: set[str] = set()
        for raw_usage in raw_usages:
            if not isinstance(raw_usage, dict):
                raise StructuredOutputError("Response contains an invalid claim usage.")
            claim_id = required_text(raw_usage, "claim_id", max_length=64)
            script_quote = required_text(
                raw_usage,
                "script_quote",
                max_length=cls._MAX_QUOTE_LENGTH,
            )
            if set(raw_usage) - {"claim_id", "script_quote"}:
                raise StructuredOutputError("Claim usage contains unexpected fields.")
            if claim_id not in supported_map:
                raise StructuredOutputError(
                    "Claim usage references an unknown or unsupported claim."
                )
            if claim_id in seen_ids:
                raise StructuredOutputError("Response contains duplicate claim usages.")
            if script_quote not in script_text:
                raise StructuredOutputError(
                    "Claim usage quote is not present verbatim in script_text."
                )
            seen_ids.add(claim_id)
            usages.append(ClaimUsage(claim_id=claim_id, script_quote=script_quote))

        required_core_ids = {
            claim.claim_id for claim in supported_claims if claim.is_core
        }
        if not required_core_ids.issubset(seen_ids):
            raise StructuredOutputError("Every supported core claim requires a usage.")
        return tuple(usages)

    @staticmethod
    def _validate_body(script_text: str, task: ScriptTask) -> None:
        if _URL_PATTERN.search(script_text):
            raise StructuredOutputError("script_text contains a URL.")
        if _CITATION_PATTERN.search(script_text):
            raise StructuredOutputError("script_text contains an inline citation label.")
        if _MARKDOWN_PATTERN.search(script_text):
            raise StructuredOutputError("script_text contains Markdown formatting.")
        if _META_PATTERN.search(script_text):
            raise StructuredOutputError("script_text contains writing instructions or meta text.")
        if any(phrase in script_text for phrase in task.forbidden_phrases):
            raise StructuredOutputError("script_text contains a forbidden phrase.")

    @staticmethod
    def _count_characters(text: str) -> int:
        return sum(1 for character in text if not character.isspace())

    def _length_bounds(self, task: ScriptTask) -> tuple[int, int]:
        tolerance = task.target_length * self._config.length_tolerance_ratio
        return (
            math.ceil(task.target_length - tolerance),
            math.floor(task.target_length + tolerance),
        )

    @staticmethod
    def _validate_config(config: ScriptGenerationConfig) -> None:
        if not 0 <= config.length_tolerance_ratio <= 0.5:
            raise ValueError("length_tolerance_ratio must be between 0 and 0.5.")
        if not 1 <= config.max_generation_attempts <= 3:
            raise ValueError("max_generation_attempts must be between 1 and 3.")
        if not isinstance(config.grounding_review_enabled, bool):
            raise ValueError("grounding_review_enabled must be a boolean.")
        if config.generation_mode not in {"single", "editorial_candidates"}:
            raise ValueError(
                "generation_mode must be single or editorial_candidates."
            )
