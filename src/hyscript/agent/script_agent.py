"""Generate and validate an evidence-grounded oral script."""

from __future__ import annotations

from dataclasses import asdict
import json
import logging
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
    SCRIPT_GENERATION_PROMPT_VERSION,
    SCRIPT_GENERATION_SYSTEM_PROMPT,
)

from ._structured import (
    StructuredOutputError,
    json_object,
    required_text,
    text_list,
)
from .contracts import Claim, ClaimUsage, ResearchOutcome, ScriptArtifact, ScriptTask


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
logger = logging.getLogger(__name__)


class ScriptGenerationError(RuntimeError):
    """Raised when research is unusable or script generation cannot be repaired."""


class ScriptAgent:
    """Turn ready research into a clean oral script with claim lineage."""

    _MAX_OUTLINE_ITEMS = 8
    _MAX_SCRIPT_LENGTH = 10000
    _MAX_QUOTE_LENGTH = 500

    def __init__(
        self,
        llm: AsyncLLMClient,
        *,
        config: ScriptGenerationConfig = ScriptGenerationConfig(),
    ) -> None:
        self._validate_config(config)
        self._llm = llm
        self._config = config

    async def generate(
        self,
        task: ScriptTask,
        research: ResearchOutcome,
    ) -> ScriptArtifact:
        """Generate one validated script, refusing to write without ready evidence."""

        if research.status != "ready":
            raise ScriptGenerationError(
                "Script generation requires research with status 'ready'."
            )
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

        messages = self._messages(task, research, supported_claims)
        current_messages = messages
        usages: list[LLMCallUsage] = []
        logger.info(
            "[4/5] 正在生成口播文案：目标 %d 个非空白字符",
            task.target_length,
        )
        for attempt in range(1, self._config.max_generation_attempts + 1):
            response_content: str | None = None
            logger.info(
                "文案生成尝试 %d/%d",
                attempt,
                self._config.max_generation_attempts,
            )
            try:
                response = await self._llm.complete(
                    current_messages,
                    reasoning_effort="high",
                )
                response_content = response.content
                usage = llm_call_usage(
                    response,
                    stage="script.generation",
                    attempt=attempt,
                )
                usages.append(usage)
                self._log_token_usage(usage)
                artifact = self._parse_artifact(
                    response_content,
                    task=task,
                    supported_claims=supported_claims,
                    attempt=attempt,
                    llm_usages=usages,
                )
                logger.info(
                    "文案生成完成：%d 个非空白字符，使用 %d 个 claim",
                    artifact.character_count,
                    len(artifact.claim_usages),
                )
                return artifact
            except LLMProviderError:
                logger.warning("第 %d 次文案 Hy3 请求失败", attempt)
            except StructuredOutputError as exc:
                logger.warning("第 %d 次文案输出未通过校验：%s", attempt, exc)
                if response_content is not None:
                    current_messages = (
                        *messages,
                        ChatMessage(role="assistant", content=response_content),
                        ChatMessage(
                            role="user",
                            content=(
                                f"上一次输出未通过校验：{exc} "
                                "请修复后重新输出完整 JSON，不要解释。"
                            ),
                        ),
                    )
            if attempt >= self._config.max_generation_attempts:
                raise ScriptGenerationError(
                    "Script generation failed after "
                    f"{self._config.max_generation_attempts} attempt(s)."
                ) from None
        raise AssertionError("unreachable")

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
            "supported_claims": [asdict(claim) for claim in supported_claims],
            "evidence": [
                asdict(item)
                for item in research.evidence
                if item.evidence_id in referenced_evidence_ids
            ],
        }
        prompt = (
            f"正文目标为 {task.target_length} 个非空白字符，允许偏差 "
            f"{self._format_percentage(self._config.length_tolerance_ratio)}。"
            "每个 is_core=true 的 claim 必须且只能在 claim_usages 中出现一次；"
            "非核心 claim 如果没有写入正文则不要添加 usage。script_quote 必须是正文中的逐字短句。\n"
            f"输出结构：{json.dumps(schema, ensure_ascii=False)}\n"
            "以下 JSON 是已经过程序校验的任务、候选论断和证据数据：\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        return (
            ChatMessage(role="system", content=SCRIPT_GENERATION_SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt),
        )

    @staticmethod
    def _format_percentage(value: float) -> str:
        return f"{value * 100:g}%"

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
        if (
            abs(character_count - task.target_length)
            > task.target_length * self._config.length_tolerance_ratio
        ):
            raise StructuredOutputError(
                "script_text is outside the configured target-length tolerance."
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

    @staticmethod
    def _validate_config(config: ScriptGenerationConfig) -> None:
        if not 0 <= config.length_tolerance_ratio <= 0.5:
            raise ValueError("length_tolerance_ratio must be between 0 and 0.5.")
        if not 1 <= config.max_generation_attempts <= 3:
            raise ValueError("max_generation_attempts must be between 1 and 3.")
