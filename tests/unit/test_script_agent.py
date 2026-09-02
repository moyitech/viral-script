"""Offline tests for evidence-grounded oral-script generation."""

from __future__ import annotations

from dataclasses import replace
import asyncio
import json
import unittest

from hyscript.agent import (
    Claim,
    Evidence,
    PlannedQuery,
    QueryPlan,
    ResearchOutcome,
    ScriptAgent,
    ScriptGenerationError,
    ScriptTask,
    TitleChainPart,
)
from hyscript.config import ScriptGenerationConfig
from hyscript.llm import ChatResponse, LLMProviderError
from hyscript.script_style import OUTLINE_LABEL_PATTERN


VALID_SCRIPT = (
    "公开材料已经确认核心变化。它影响的不是所有人，而是符合特定条件的办理者。"
    "普通人先核对适用范围，再按照正式说明准备材料，能减少误读和无效操作。"
)
VALID_QUOTE = "公开材料已经确认核心变化"
REVIEWED_SCRIPT = (
    "公开材料清楚说明核心变化。"
    "面对这类变化，先别急着下结论，也不要自然套到每个人身上。"
    "普通人可以先核对正式说明中的适用条件，再决定是否按其准备材料。"
)
REVIEWED_QUOTE = "公开材料清楚说明核心变化"
REVIEWED_CONTEXT_QUOTE = (
    "面对这类变化，先别急着下结论，也不要自然套到每个人身上"
)
REVIEWED_CLAUSE_AUDIT = [
    {
        "script_quote": "公开材料清楚说明核心变化。",
        "kind": "D",
        "claim_ids": ["C001"],
    },
    {
        "script_quote": "面对这类变化，先别急着下结论，也不要自然套到每个人身上。",
        "kind": "A",
        "claim_ids": [],
    },
    {
        "script_quote": "普通人可以先核对正式说明中的适用条件，再决定是否按其准备材料。",
        "kind": "A",
        "claim_ids": [],
    },
]


def count_characters(text: str) -> int:
    return sum(1 for character in text if not character.isspace())


def ready_research(*, status: str = "ready") -> ResearchOutcome:
    return ResearchOutcome(
        status=status,
        query_plan=QueryPlan(
            goal="核实变化、边界和现实影响",
            must_verify=("变化是否确认", "适用范围"),
            queries=(PlannedQuery(query="权威说明", purpose="核实变化"),),
        ),
        search_responses=(),
        evidence=(
            Evidence(
                evidence_id="E001",
                result_ref="R001",
                title="来源一",
                url="https://authority.example/one",
                excerpt="正式材料确认了核心变化。",
                source_query="权威说明",
                source_type="official_primary",
            ),
            Evidence(
                evidence_id="E002",
                result_ref="R002",
                title="来源二",
                url="https://news.example/two",
                excerpt="变化只适用于符合特定条件的办理者。",
                source_query="适用范围",
                source_type="independent_secondary",
            ),
        ),
        claims=(
            Claim(
                claim_id="C001",
                text="公开材料已经确认核心变化。",
                evidence_ids=("E001",),
                is_core=True,
            ),
            Claim(
                claim_id="C002",
                text="变化只适用于符合特定条件的办理者。",
                evidence_ids=("E002",),
                is_core=False,
            ),
            Claim(
                claim_id="C003",
                text="不同材料对影响范围存在冲突。",
                evidence_ids=("E001", "E002"),
                is_core=False,
                support_status="conflicting",
            ),
        ),
        errors=(),
        query_plan_prompt_version="research-query-plan-1.1.0",
        evidence_prompt_version="research-evidence-2.1.0",
        llm_request_count=2,
        search_request_count=3,
        title_chain=(
            TitleChainPart(
                component="subject_scope",
                status="covered",
                claim_ids=("C001",),
                reason="核心论断覆盖标题主体。",
            ),
            TitleChainPart(
                component="stated_context",
                status="covered",
                claim_ids=("C001",),
                reason="核心论断覆盖题设情境。",
            ),
            TitleChainPart(
                component="question_predicate",
                status="covered",
                claim_ids=("C001",),
                reason="核心论断直接回答所问谓词。",
            ),
        ),
    )


def output_payload(
    script_text: str = VALID_SCRIPT,
    *,
    claim_usages: list[dict[str, str]] | None = None,
) -> str:
    return json.dumps(
        {
            "outline": ["提出变化", "解释适用边界", "给出行动建议"],
            "script_text": script_text,
            "claim_usages": claim_usages
            if claim_usages is not None
            else [{"claim_id": "C001", "script_quote": VALID_QUOTE}],
        },
        ensure_ascii=False,
    )


def background_output_payload(script_text: str = VALID_SCRIPT) -> str:
    return json.dumps(
        {
            "outline": ["建立冲突", "解释影响", "留下判断"],
            "script_text": script_text,
            "reference_ids": ["E002", "E001"],
        },
        ensure_ascii=False,
    )


def final_rewrite_payload(
    script_text: str = VALID_SCRIPT,
    *,
    claim_usages: list[dict[str, str]] | None = None,
) -> str:
    payload: dict[str, object] = {"script_text": script_text}
    if claim_usages is not None:
        payload["claim_usages"] = claim_usages
    return json.dumps(payload, ensure_ascii=False)


def editorial_candidate_payload(
    script_text: str = VALID_SCRIPT,
    *,
    reference_ids: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "outline": ["建立冲突", "口语解释", "留下余味"],
            "script_text": script_text,
            "reference_ids": reference_ids or ["E001"],
        },
        ensure_ascii=False,
    )


def editorial_final_payload(
    script_text: str = VALID_SCRIPT,
    *,
    reference_ids: list[str] | None = None,
    candidate_ids: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "outline": ["冲突开场", "两次推进", "自然收束"],
            "script_text": script_text,
            "reference_ids": reference_ids or ["E001", "E002"],
            "selected_candidate_ids": candidate_ids or ["C01", "C02"],
        },
        ensure_ascii=False,
    )


def review_output_payload(
    script_text: str = REVIEWED_SCRIPT,
    *,
    decision: str = "accepted",
    issues: list[str] | None = None,
    claim_usages: list[dict[str, str]] | None = None,
    clause_audit: list[dict[str, object]] | None = None,
) -> str:
    return json.dumps(
        {
            "decision": decision,
            "issues": issues or [],
            "outline": ["修订主线", "核对边界"],
            "script_text": script_text,
            "claim_usages": claim_usages
            if claim_usages is not None
            else [{"claim_id": "C001", "script_quote": REVIEWED_QUOTE}],
            "clause_audit": clause_audit
            if clause_audit is not None
            else ([] if decision == "rejected" else REVIEWED_CLAUSE_AUDIT),
        },
        ensure_ascii=False,
    )


class FakeLLM:
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[object, dict[str, object]]] = []

    async def complete(self, messages, **kwargs) -> ChatResponse:
        self.calls.append((messages, kwargs))
        if not self.responses:
            raise AssertionError("Unexpected LLM request")
        call_number = len(self.calls)
        next_response = self.responses.pop(0)
        if isinstance(next_response, Exception):
            raise next_response
        return ChatResponse(
            content=next_response,
            model="hy3-test",
            request_id=f"script-request-{call_number}",
            usage={
                "prompt_tokens": 80,
                "completion_tokens": 30,
                "total_tokens": 110,
                "prompt_tokens_details": {"cached_tokens": 8},
                "completion_tokens_details": {"reasoning_tokens": 6},
            },
        )


class ScriptAgentTests(unittest.IsolatedAsyncioTestCase):
    def test_script_task_preserves_arbitrary_valid_integer_lengths(self) -> None:
        for target_length in (50, 321, 5000):
            with self.subTest(target_length=target_length):
                task = ScriptTask(topic="任意合法长度", target_length=target_length)

                self.assertEqual(task.target_length, target_length)

    def test_script_task_rejects_non_integer_lengths(self) -> None:
        for target_length in (321.5, "321", True):
            with self.subTest(target_length=target_length):
                with self.assertRaisesRegex(ValueError, "must be an integer"):
                    ScriptTask(topic="非法长度", target_length=target_length)

    async def test_background_generation_keeps_references_outside_body(self) -> None:
        research = replace(ready_research(), claims=(), title_chain=())
        llm = FakeLLM([background_output_payload()])
        task = ScriptTask(
            topic="背景增强的热点",
            target_length=count_characters(VALID_SCRIPT),
        )

        artifact = await ScriptAgent(
            llm,
            config=ScriptGenerationConfig(grounding_review_enabled=True),
        ).generate(task, research)

        self.assertEqual(artifact.script_text, VALID_SCRIPT)
        self.assertEqual(artifact.claim_usages, ())
        self.assertEqual(artifact.reference_ids, ("E002", "E001"))
        self.assertEqual(artifact.grounding_review_status, "disabled")
        self.assertEqual(artifact.grounding_review_attempt_count, 0)
        self.assertEqual(
            artifact.prompt_version,
            "script-generation-background-1.1.0",
        )
        self.assertEqual(len(llm.calls), 1)
        system_message, user_message = llm.calls[0][0]
        self.assertIn("不是要求逐句绑定的论证链", system_message.content)
        self.assertIn("正文之外的引用元数据", system_message.content)
        self.assertIn("结尾记忆点", system_message.content)
        self.assertIn("承接句或问句", system_message.content)
        self.assertIn('"background_references"', user_message.content)
        self.assertNotIn("E001", artifact.script_text)

    async def test_editorial_generation_runs_three_candidates_then_editor(self) -> None:
        research = replace(ready_research(), claims=(), title_chain=())
        llm = FakeLLM(
            [
                editorial_candidate_payload(),
                editorial_candidate_payload(),
                editorial_candidate_payload(),
                editorial_final_payload(),
            ]
        )
        task = ScriptTask(
            topic="三候选并行主编",
            target_length=count_characters(VALID_SCRIPT),
        )

        artifact = await ScriptAgent(
            llm,
            config=ScriptGenerationConfig(
                generation_mode="editorial_candidates"
            ),
        ).generate(task, research)

        self.assertEqual(artifact.generation_mode, "editorial_candidates")
        self.assertEqual(len(artifact.generation_candidates), 3)
        self.assertEqual(
            tuple(item.candidate_id for item in artifact.generation_candidates),
            ("C01", "C02", "C03"),
        )
        self.assertEqual(artifact.selected_candidate_ids, ("C01", "C02"))
        self.assertEqual(artifact.generation_attempt_count, 4)
        self.assertEqual(artifact.editor_attempt_count, 1)
        self.assertFalse(artifact.length_repair_attempted)
        self.assertTrue(artifact.length_within_tolerance)
        self.assertEqual(
            tuple(item.stage for item in artifact.llm_usages),
            (
                "script.candidate.conflict_interest",
                "script.candidate.scene_conversation",
                "script.candidate.counterintuitive_turn",
                "script.editor",
            ),
        )
        self.assertTrue(
            all(kwargs == {"reasoning_effort": "high"} for _, kwargs in llm.calls)
        )
        self.assertNotIn("max_tokens", llm.calls[-1][1])

    async def test_editorial_candidates_are_actually_concurrent(self) -> None:
        research = replace(ready_research(), claims=(), title_chain=())

        class ConcurrentFakeLLM:
            def __init__(self) -> None:
                self.call_count = 0
                self.active = 0
                self.max_active = 0
                self.three_started = asyncio.Event()

            async def complete(self, messages, **kwargs) -> ChatResponse:
                self.call_count += 1
                call_number = self.call_count
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                if call_number <= 3:
                    if call_number == 3:
                        self.three_started.set()
                    await self.three_started.wait()
                    content = editorial_candidate_payload()
                else:
                    content = editorial_final_payload()
                await asyncio.sleep(0)
                self.active -= 1
                return ChatResponse(content=content, model="fake")

        llm = ConcurrentFakeLLM()
        artifact = await ScriptAgent(
            llm,
            config=ScriptGenerationConfig(
                generation_mode="editorial_candidates"
            ),
            request_semaphore=asyncio.Semaphore(3),
        ).generate(
            ScriptTask(
                topic="并发候选",
                target_length=count_characters(VALID_SCRIPT),
            ),
            research,
        )

        self.assertEqual(llm.max_active, 3)
        self.assertEqual(artifact.generation_attempt_count, 4)

    async def test_editorial_json_repairs_are_unbounded_by_length_budget(self) -> None:
        research = replace(ready_research(), claims=(), title_chain=())
        llm = FakeLLM(
            [
                "不是JSON",
                editorial_candidate_payload(),
                editorial_candidate_payload(),
                editorial_candidate_payload(),
                "仍然不是JSON",
                editorial_final_payload(),
            ]
        )
        artifact = await ScriptAgent(
            llm,
            config=ScriptGenerationConfig(
                generation_mode="editorial_candidates"
            ),
        ).generate(
            ScriptTask(
                topic="JSON持续修复",
                target_length=count_characters(VALID_SCRIPT),
            ),
            research,
        )

        self.assertEqual(artifact.generation_attempt_count, 6)
        self.assertEqual(artifact.editor_attempt_count, 2)
        self.assertFalse(artifact.length_repair_attempted)
        self.assertTrue(artifact.length_within_tolerance)

    async def test_editorial_length_repairs_once_then_accepts_mismatch(self) -> None:
        research = replace(ready_research(), claims=(), title_chain=())
        too_short = "这句话很短。"
        llm = FakeLLM(
            [
                editorial_candidate_payload(),
                editorial_candidate_payload(),
                editorial_candidate_payload(),
                editorial_final_payload(too_short),
                editorial_final_payload(too_short),
            ]
        )
        artifact = await ScriptAgent(
            llm,
            config=ScriptGenerationConfig(
                generation_mode="editorial_candidates"
            ),
        ).generate(
            ScriptTask(
                topic="字数由评估器扣分",
                target_length=count_characters(VALID_SCRIPT),
            ),
            research,
        )

        self.assertEqual(artifact.script_text, too_short)
        self.assertEqual(artifact.editor_attempt_count, 2)
        self.assertEqual(artifact.generation_attempt_count, 5)
        self.assertTrue(artifact.length_repair_attempted)
        self.assertFalse(artifact.length_within_tolerance)
        self.assertIn("唯一一次字数修复机会", llm.calls[-1][0][-1].content)

    async def test_generates_valid_script_with_supported_claim_lineage(self) -> None:
        task = ScriptTask(
            topic="一个已经完成检索的热点",
            target_length=count_characters(VALID_SCRIPT),
        )
        llm = FakeLLM([output_payload()])

        with self.assertLogs("hyscript.agent.script_agent", level="INFO") as logs:
            artifact = await ScriptAgent(llm).generate(task, ready_research())

        self.assertEqual(artifact.script_text, VALID_SCRIPT)
        self.assertEqual(artifact.character_count, count_characters(VALID_SCRIPT))
        self.assertEqual(artifact.claim_usages[0].claim_id, "C001")
        self.assertEqual(artifact.prompt_version, "script-generation-1.7.7")
        self.assertEqual(artifact.generation_attempt_count, 1)
        self.assertEqual(len(artifact.llm_usages), 1)
        self.assertEqual(artifact.llm_usages[0].input_tokens, 80)
        self.assertEqual(artifact.llm_usages[0].output_tokens, 30)
        self.assertEqual(artifact.llm_usages[0].reasoning_tokens, 6)
        self.assertEqual(llm.calls[0][1], {"reasoning_effort": "high"})
        system_message, user_message = llm.calls[0][0]
        self.assertIn("不可信外部数据", system_message.content)
        self.assertIn("claim 文本只是待转述摘要", system_message.content)
        self.assertIn("相邻句不得换词复述同一判断", system_message.content)
        self.assertIn("把单一地点写成“等地”", system_message.content)
        self.assertIn('"claim_id": "C001"', user_message.content)
        self.assertNotIn('"claim_id": "C002"', user_message.content)
        self.assertNotIn('"claim_id": "C003"', user_message.content)
        self.assertIn("短稿只保留一个核心判断", user_message.content)
        self.assertIn("不要求照抄 claim 文本", user_message.content)
        self.assertIn('"title_chain"', user_message.content)
        self.assertIn('"component": "question_predicate"', user_message.content)
        self.assertIn("reason 不能补足缺失链路", user_message.content)
        self.assertIn("source_scope", system_message.content)
        self.assertIn("来源名称只能逐字", system_message.content)
        self.assertIn("仅覆盖", system_message.content)
        self.assertIn("消费者的准时承诺", system_message.content)
        self.assertIn("claim_usages 不是“核心 claim 打卡表”", system_message.content)
        self.assertIn("主体、特定情境和所问谓词", system_message.content)
        self.assertIn("不得只圈住复合句里受支持的半句", system_message.content)
        self.assertIn("“将探索”“部分”“可能”", system_message.content)
        self.assertIn("审计备注", system_message.content)
        self.assertIn("值为 unknown 或空白时不得写入正文", system_message.content)
        self.assertIn("某信息未出现在 selected excerpt 中", system_message.content)
        self.assertIn("卫星终端许可", system_message.content)
        self.assertIn("长稿不得靠展开 case excerpt 填字", system_message.content)
        self.assertIn("同一报道含多个消费者或患者", system_message.content)
        self.assertIn("展开支付选项就能防止额外花销", system_message.content)
        self.assertIn("给路人拍摄上传具体纠纷留出", system_message.content)
        self.assertNotIn("必须保留的范围边界", system_message.content)
        self.assertNotIn("max_tokens", user_message.content)
        self.assertTrue(any("[4/5]" in message for message in logs.output))
        self.assertTrue(any("文案生成完成" in message for message in logs.output))

    async def test_legacy_unclassified_core_sources_are_rejected_before_llm(
        self,
    ) -> None:
        research = ready_research()
        research = replace(
            research,
            evidence=tuple(
                replace(evidence, source_type="unclassified")
                for evidence in research.evidence
            ),
        )
        llm = FakeLLM([])

        with self.assertRaisesRegex(
            ScriptGenerationError,
            "Core claim source quality does not satisfy its claim_kind",
        ):
            await ScriptAgent(llm).generate(
                ScriptTask(
                    topic="旧快照缺少来源分类",
                    target_length=count_characters(VALID_SCRIPT),
                ),
                research,
            )

        self.assertEqual(llm.calls, [])

    async def test_mismatched_core_source_type_is_rejected_before_llm(self) -> None:
        research = ready_research()
        research = replace(
            research,
            evidence=(
                replace(
                    research.evidence[0],
                    source_type="reputable_reporting",
                ),
                research.evidence[1],
            ),
            claims=(
                replace(research.claims[0], claim_kind="rule_or_terms"),
                *research.claims[1:],
            ),
        )
        llm = FakeLLM([])

        with self.assertRaisesRegex(
            ScriptGenerationError,
            "Core claim source quality does not satisfy its claim_kind",
        ):
            await ScriptAgent(llm).generate(
                ScriptTask(
                    topic="二手报道不能支撑核心规则",
                    target_length=count_characters(VALID_SCRIPT),
                ),
                research,
            )

        self.assertEqual(llm.calls, [])

    async def test_core_claim_accepts_one_matching_source_among_weak_refs(self) -> None:
        research = ready_research()
        research = replace(
            research,
            evidence=(
                replace(research.evidence[0], source_type="direct_terms"),
                replace(research.evidence[1], source_type="unclassified"),
            ),
            claims=(
                replace(
                    research.claims[0],
                    evidence_ids=("E001", "E002"),
                    claim_kind="rule_or_terms",
                ),
                *research.claims[1:],
            ),
        )
        llm = FakeLLM([output_payload()])

        artifact = await ScriptAgent(llm).generate(
            ScriptTask(
                topic="强来源与弱伴随来源并存",
                target_length=count_characters(VALID_SCRIPT),
            ),
            research,
        )

        self.assertEqual(artifact.generation_attempt_count, 1)
        self.assertEqual(len(llm.calls), 1)

    async def test_repairs_length_once_without_setting_max_tokens(self) -> None:
        task = ScriptTask(
            topic="需要修复长度的热点",
            target_length=count_characters(VALID_SCRIPT),
        )
        llm = FakeLLM([output_payload("太短了"), output_payload()])

        artifact = await ScriptAgent(llm).generate(task, ready_research())

        self.assertEqual(artifact.generation_attempt_count, 2)
        self.assertEqual(len(artifact.llm_usages), 2)
        self.assertEqual(sum(item.total_tokens or 0 for item in artifact.llm_usages), 220)
        self.assertEqual(len(llm.calls), 2)
        self.assertTrue(
            all(kwargs == {"reasoning_effort": "high"} for _, kwargs in llm.calls)
        )
        self.assertIn("target-length", llm.calls[1][0][-1].content)
        self.assertIn(
            f"actual={count_characters('太短了')}",
            llm.calls[1][0][-1].content,
        )
        self.assertIn("允许区间是", llm.calls[1][0][-1].content)

    async def test_terminal_generation_error_retains_last_failure_and_usage(
        self,
    ) -> None:
        task = ScriptTask(
            topic="连续两次生成失败的热点",
            target_length=count_characters(VALID_SCRIPT),
        )
        llm = FakeLLM(
            [output_payload("太短了"), output_payload("还是太短")]
        )

        with self.assertRaises(ScriptGenerationError) as raised:
            await ScriptAgent(llm).generate(task, ready_research())

        error = raised.exception
        self.assertIn(
            "Last failure: structured_output_error: script_text is outside",
            str(error),
        )
        self.assertEqual(error.generation_attempt_count, 2)
        self.assertEqual(len(error.llm_usages), 2)
        self.assertEqual(
            tuple(item.attempt for item in error.llm_usages),
            (1, 2),
        )

    async def test_two_provider_failures_do_not_consume_content_attempts(
        self,
    ) -> None:
        task = ScriptTask(
            topic="临时服务失败后仍可生成",
            target_length=count_characters(VALID_SCRIPT),
        )
        llm = FakeLLM(
            [
                LLMProviderError("temporary outage one"),
                LLMProviderError("temporary outage two"),
                output_payload(),
            ]
        )

        artifact = await ScriptAgent(llm).generate(task, ready_research())

        self.assertEqual(artifact.generation_attempt_count, 3)
        self.assertEqual(len(llm.calls), 3)
        self.assertEqual(len(artifact.llm_usages), 1)
        self.assertEqual(artifact.llm_usages[0].attempt, 3)

    async def test_provider_retry_limit_is_initial_request_plus_two_retries(
        self,
    ) -> None:
        task = ScriptTask(
            topic="服务持续失败时有硬上限",
            target_length=count_characters(VALID_SCRIPT),
        )
        llm = FakeLLM(
            [
                LLMProviderError("provider outage one"),
                LLMProviderError("provider outage two"),
                LLMProviderError("provider outage three"),
                output_payload(),
            ]
        )

        with self.assertRaises(ScriptGenerationError) as raised:
            await ScriptAgent(llm).generate(task, ready_research())

        error = raised.exception
        self.assertEqual(error.generation_attempt_count, 3)
        self.assertEqual(len(llm.calls), 3)
        self.assertEqual(len(llm.responses), 1)
        self.assertEqual(error.llm_usages, ())
        self.assertIn(
            "Last failure: provider_error: provider outage three",
            str(error),
        )

    async def test_provider_and_structured_failures_share_a_total_request_cap(
        self,
    ) -> None:
        task = ScriptTask(
            topic="服务失败和内容修复交错",
            target_length=count_characters(VALID_SCRIPT),
        )
        llm = FakeLLM(
            [
                LLMProviderError("temporary outage one"),
                output_payload("太短了"),
                LLMProviderError("temporary outage two"),
                output_payload(),
            ]
        )

        artifact = await ScriptAgent(llm).generate(task, ready_research())

        self.assertEqual(artifact.generation_attempt_count, 4)
        self.assertEqual(len(llm.calls), 4)
        self.assertEqual(
            tuple(item.attempt for item in artifact.llm_usages),
            (2, 4),
        )
        self.assertIn("target-length", llm.calls[3][0][-1].content)

    async def test_optional_grounding_review_returns_a_revalidated_artifact(
        self,
    ) -> None:
        task = ScriptTask(
            topic="需要事实边界校对的热点",
            target_length=count_characters(VALID_SCRIPT),
        )
        reviewed_payload = review_output_payload(
            REVIEWED_SCRIPT,
            claim_usages=[
                {"claim_id": "C001", "script_quote": REVIEWED_QUOTE},
            ],
        )
        llm = FakeLLM([output_payload(), reviewed_payload])
        config = ScriptGenerationConfig(grounding_review_enabled=True)

        artifact = await ScriptAgent(llm, config=config).generate(
            task,
            ready_research(),
        )

        self.assertEqual(artifact.script_text, REVIEWED_SCRIPT)
        self.assertEqual(artifact.generation_attempt_count, 1)
        self.assertEqual(artifact.grounding_review_attempt_count, 1)
        self.assertEqual(artifact.grounding_review_status, "accepted")
        self.assertEqual(
            artifact.grounding_review_prompt_version,
            "script-grounding-review-2.5.6",
        )
        self.assertEqual(artifact.grounding_review_draft_text, VALID_SCRIPT)
        self.assertEqual(
            artifact.grounding_review_draft_character_count,
            count_characters(VALID_SCRIPT),
        )
        self.assertEqual(
            [item.stage for item in artifact.llm_usages],
            ["script.generation", "script.grounding_review"],
        )
        self.assertIn("evidence excerpt 是最终事实边界", llm.calls[1][0][0].content)
        self.assertIn("claim 自身若将事实归因", llm.calls[1][0][0].content)
        self.assertIn("消费者准时承诺", llm.calls[1][0][0].content)
        self.assertIn("不要信任或沿用 draft.claim_usages", llm.calls[1][0][0].content)
        self.assertIn("修文前先做标题链 gate", llm.calls[1][0][0].content)
        self.assertIn("一般时间窗或骑手扣款", llm.calls[1][0][0].content)
        self.assertIn("单一平台规则不得改写成无平台名", llm.calls[1][0][0].content)
        self.assertIn("务必打码才能避开侵权", llm.calls[1][0][0].content)
        self.assertIn("claim 写了 excerpt 未列出的措施", llm.calls[1][0][0].content)
        self.assertIn("已选 excerpt 没有出现某信息", llm.calls[1][0][0].content)
        self.assertIn("卫星终端未获国家许可", llm.calls[1][0][0].content)
        self.assertIn("日期、年龄、症状、价格", llm.calls[1][0][0].content)
        self.assertIn("逐案核对报道中的消费者或患者", llm.calls[1][0][0].content)
        self.assertIn("展开选项即可防止额外花销", llm.calls[1][0][0].content)
        self.assertIn("给普通路人拍摄上传", llm.calls[1][0][0].content)
        self.assertIn(
            "不能拼接成“一键开药导致误诊”",
            llm.calls[1][0][0].content,
        )
        self.assertIn("审计备注", llm.calls[1][0][0].content)
        self.assertIn(
            "值为 unknown 或空白时不得写入正文",
            llm.calls[1][0][0].content,
        )
        self.assertNotIn("不得被省略或放大", llm.calls[1][0][0].content)
        self.assertIn('"draft"', llm.calls[1][0][-1].content)
        self.assertIn('"decision"', llm.calls[1][0][-1].content)
        self.assertIn('"clause_audit"', llm.calls[1][0][-1].content)
        self.assertIn('"title_chain"', llm.calls[1][0][-1].content)
        self.assertIn(
            '"component": "question_predicate"',
            llm.calls[1][0][-1].content,
        )
        self.assertIn("rejected 不需要", llm.calls[1][0][-1].content)
        self.assertEqual(artifact.grounding_review_issues, ())
        self.assertIsNone(artifact.grounding_review_failure_reason)

    async def test_grounding_review_repairs_invalid_output_once(self) -> None:
        task = ScriptTask(
            topic="校对结构错误后修复",
            target_length=count_characters(VALID_SCRIPT),
        )
        invalid_review = output_payload("太短了")
        llm = FakeLLM(
            [
                output_payload(),
                invalid_review,
                review_output_payload(),
            ]
        )

        artifact = await ScriptAgent(
            llm,
            config=ScriptGenerationConfig(grounding_review_enabled=True),
        ).generate(task, ready_research())

        self.assertEqual(artifact.script_text, REVIEWED_SCRIPT)
        self.assertEqual(artifact.grounding_review_attempt_count, 2)
        self.assertEqual(artifact.grounding_review_status, "accepted")
        self.assertEqual(len(artifact.llm_usages), 3)
        self.assertEqual(
            [item.attempt for item in artifact.llm_usages],
            [1, 1, 2],
        )
        retry_messages = llm.calls[2][0]
        self.assertEqual(retry_messages[:2], llm.calls[1][0])
        self.assertEqual(retry_messages[-2].content, invalid_review)
        self.assertIn("structured_output_error", retry_messages[-1].content)

    async def test_grounding_review_retries_provider_failure_once(self) -> None:
        task = ScriptTask(
            topic="校对请求失败后重试",
            target_length=count_characters(VALID_SCRIPT),
        )
        llm = FakeLLM(
            [
                output_payload(),
                LLMProviderError("simulated review outage"),
                review_output_payload(),
            ]
        )

        artifact = await ScriptAgent(
            llm,
            config=ScriptGenerationConfig(grounding_review_enabled=True),
        ).generate(task, ready_research())

        self.assertEqual(artifact.grounding_review_attempt_count, 2)
        self.assertEqual(artifact.grounding_review_status, "accepted")
        self.assertEqual(
            [(item.stage, item.attempt) for item in artifact.llm_usages],
            [("script.generation", 1), ("script.grounding_review", 2)],
        )
        self.assertIn(
            "provider_error: simulated review outage",
            llm.calls[2][0][-1].content,
        )

    async def test_grounding_review_survives_two_provider_failures(self) -> None:
        task = ScriptTask(
            topic="校对连续两次请求失败后恢复",
            target_length=count_characters(VALID_SCRIPT),
        )
        llm = FakeLLM(
            [
                output_payload(),
                LLMProviderError("review outage one"),
                LLMProviderError("review outage two"),
                review_output_payload(),
            ]
        )

        artifact = await ScriptAgent(
            llm,
            config=ScriptGenerationConfig(grounding_review_enabled=True),
        ).generate(task, ready_research())

        self.assertEqual(artifact.grounding_review_attempt_count, 3)
        self.assertEqual(artifact.grounding_review_status, "accepted")
        self.assertEqual(
            [(item.stage, item.attempt) for item in artifact.llm_usages],
            [("script.generation", 1), ("script.grounding_review", 3)],
        )

    async def test_three_consecutive_review_provider_failures_fall_back(self) -> None:
        task = ScriptTask(
            topic="校对连续请求失败达到硬上限",
            target_length=count_characters(VALID_SCRIPT),
        )
        llm = FakeLLM(
            [
                output_payload(),
                LLMProviderError("review outage one"),
                LLMProviderError("review outage two"),
                LLMProviderError("review outage three"),
                review_output_payload(),
            ]
        )

        artifact = await ScriptAgent(
            llm,
            config=ScriptGenerationConfig(grounding_review_enabled=True),
        ).generate(task, ready_research())

        self.assertEqual(artifact.grounding_review_attempt_count, 3)
        self.assertEqual(artifact.grounding_review_status, "fallback")
        self.assertEqual(len(llm.calls), 4)
        self.assertIn(
            "provider_error: review outage three",
            artifact.grounding_review_failure_reason or "",
        )

    async def test_review_provider_and_content_failures_share_total_cap(self) -> None:
        task = ScriptTask(
            topic="校对请求与内容失败共享总上限",
            target_length=count_characters(VALID_SCRIPT),
        )
        llm = FakeLLM(
            [
                output_payload(),
                output_payload("太短了"),
                LLMProviderError("review outage one"),
                LLMProviderError("review outage two"),
                output_payload("仍然太短"),
                review_output_payload(),
            ]
        )

        artifact = await ScriptAgent(
            llm,
            config=ScriptGenerationConfig(grounding_review_enabled=True),
        ).generate(task, ready_research())

        self.assertEqual(artifact.grounding_review_attempt_count, 4)
        self.assertEqual(artifact.grounding_review_status, "fallback")
        self.assertEqual(len(llm.calls), 5)
        self.assertEqual(
            [item.attempt for item in artifact.llm_usages],
            [1, 1, 4],
        )

    async def test_invalid_grounding_review_falls_back_to_valid_draft(self) -> None:
        task = ScriptTask(
            topic="校对失败仍保留已验证草稿",
            target_length=count_characters(VALID_SCRIPT),
        )
        llm = FakeLLM(
            [
                output_payload(),
                output_payload("太短了"),
                output_payload("还是太短"),
            ]
        )
        config = ScriptGenerationConfig(grounding_review_enabled=True)

        artifact = await ScriptAgent(llm, config=config).generate(
            task,
            ready_research(),
        )

        self.assertEqual(artifact.script_text, VALID_SCRIPT)
        self.assertEqual(artifact.generation_attempt_count, 1)
        self.assertEqual(artifact.grounding_review_attempt_count, 2)
        self.assertEqual(artifact.grounding_review_status, "fallback")
        self.assertEqual(len(artifact.llm_usages), 3)
        self.assertIn(
            "structured_output_error",
            artifact.grounding_review_failure_reason or "",
        )

    async def test_grounding_review_can_reject_an_unfixable_research_gap(self) -> None:
        task = ScriptTask(
            topic="缺少关键证据时拒绝进入正式结果",
            target_length=count_characters(VALID_SCRIPT),
        )
        rejected = review_output_payload(
            VALID_SCRIPT,
            decision="rejected",
            issues=[
                "insufficient_evidence: 缺少回答价格变化所需的终端价格数据。"
            ],
            claim_usages=[
                {"claim_id": "C001", "script_quote": VALID_QUOTE},
            ],
        )
        llm = FakeLLM([output_payload(), rejected])

        artifact = await ScriptAgent(
            llm,
            config=ScriptGenerationConfig(grounding_review_enabled=True),
        ).generate(task, ready_research())

        self.assertEqual(artifact.script_text, VALID_SCRIPT)
        self.assertEqual(artifact.generation_attempt_count, 1)
        self.assertEqual(artifact.grounding_review_attempt_count, 1)
        self.assertEqual(artifact.grounding_review_status, "rejected")
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(artifact.grounding_review_failure_reason, "review_rejected")
        self.assertEqual(
            artifact.grounding_review_issues,
            ("insufficient_evidence: 缺少回答价格变化所需的终端价格数据。",),
        )

    def test_accepted_grounding_review_requires_exact_clause_audit(self) -> None:
        task = ScriptTask(
            topic="审计必须完整覆盖每个分句",
            target_length=count_characters(REVIEWED_SCRIPT),
        )
        claims = tuple(
            claim
            for claim in ready_research().claims
            if claim.support_status == "supported"
        )
        valid_usages = [
            {"claim_id": "C001", "script_quote": REVIEWED_QUOTE},
            {"claim_id": "C002", "script_quote": REVIEWED_CONTEXT_QUOTE},
        ]
        two_claim_audit = [
            REVIEWED_CLAUSE_AUDIT[0],
            {
                **REVIEWED_CLAUSE_AUDIT[1],
                "kind": "D",
                "claim_ids": ["C002"],
            },
            REVIEWED_CLAUSE_AUDIT[2],
        ]
        cases: tuple[tuple[str, list[dict[str, object]], list[dict[str, str]]], ...] = (
            ("missing", two_claim_audit[:-1], valid_usages),
            (
                "duplicate",
                [
                    two_claim_audit[0],
                    two_claim_audit[0],
                    two_claim_audit[2],
                ],
                valid_usages,
            ),
            (
                "unknown",
                [
                    {
                        **REVIEWED_CLAUSE_AUDIT[0],
                        "claim_ids": ["C999"],
                    },
                    *two_claim_audit[1:],
                ],
                valid_usages,
            ),
            (
                "D without claim",
                [
                    {
                        **REVIEWED_CLAUSE_AUDIT[0],
                        "claim_ids": [],
                    },
                    *two_claim_audit[1:],
                ],
                valid_usages,
            ),
            (
                "claim repeated across D clauses",
                [
                    two_claim_audit[0],
                    {
                        **two_claim_audit[1],
                        "claim_ids": ["C001"],
                    },
                    two_claim_audit[2],
                ],
                valid_usages,
            ),
            (
                "A with claim",
                [
                    *two_claim_audit[:2],
                    {
                        **two_claim_audit[2],
                        "claim_ids": ["C001"],
                    },
                ],
                valid_usages,
            ),
            (
                "detached usage",
                two_claim_audit,
                [
                    {"claim_id": "C001", "script_quote": REVIEWED_QUOTE},
                    {"claim_id": "C002", "script_quote": REVIEWED_QUOTE},
                ],
            ),
        )
        agent = ScriptAgent(FakeLLM([]))
        for label, clause_audit, claim_usages in cases:
            with self.subTest(label=label):
                response = review_output_payload(
                    claim_usages=claim_usages,
                    clause_audit=clause_audit,
                )
                with self.assertRaises(ValueError):
                    agent._parse_review_response(
                        response,
                        task=task,
                        supported_claims=claims,
                        generation_attempt_count=1,
                        llm_usages=(),
                    )

    def test_accepted_review_requires_audit_but_rejected_review_is_compatible(
        self,
    ) -> None:
        task = ScriptTask(
            topic="拒绝结果无需制造完整审计",
            target_length=count_characters(REVIEWED_SCRIPT),
        )
        claims = tuple(
            claim
            for claim in ready_research().claims
            if claim.support_status == "supported"
        )
        agent = ScriptAgent(FakeLLM([]))

        accepted = json.loads(review_output_payload())
        del accepted["clause_audit"]
        with self.assertRaisesRegex(ValueError, "requires clause_audit"):
            agent._parse_review_response(
                json.dumps(accepted, ensure_ascii=False),
                task=task,
                supported_claims=claims,
                generation_attempt_count=1,
                llm_usages=(),
            )

        rejected = json.loads(
            review_output_payload(
                decision="rejected",
                issues=["insufficient_evidence: 标题关键链路缺少直接证据。"],
            )
        )
        del rejected["clause_audit"]
        reviewed, issues = agent._parse_review_response(
            json.dumps(rejected, ensure_ascii=False),
            task=task,
            supported_claims=claims,
            generation_attempt_count=1,
            llm_usages=(),
        )

        self.assertIsNone(reviewed)
        self.assertEqual(
            issues,
            ("insufficient_evidence: 标题关键链路缺少直接证据。",),
        )

    async def test_rejects_unknown_duplicate_missing_and_non_verbatim_usages(
        self,
    ) -> None:
        invalid_usages = (
            [{"claim_id": "C999", "script_quote": VALID_QUOTE}],
            [
                {"claim_id": "C001", "script_quote": VALID_QUOTE},
                {"claim_id": "C001", "script_quote": VALID_QUOTE},
            ],
            [],
            [{"claim_id": "C001", "script_quote": "正文里没有这句话"}],
        )
        task = ScriptTask(
            topic="需要校验引用映射的热点",
            target_length=count_characters(VALID_SCRIPT),
        )
        for usages in invalid_usages:
            with self.subTest(usages=usages):
                response = output_payload(claim_usages=usages)
                llm = FakeLLM([response, response])
                with self.assertRaises(ScriptGenerationError):
                    await ScriptAgent(llm).generate(task, ready_research())

    async def test_rejects_forbidden_meta_url_citation_and_markdown_body(self) -> None:
        cases = (
            (f"这句话声称绝对安全。{VALID_SCRIPT}", ("绝对安全",)),
            (f"正文：{VALID_SCRIPT}", ()),
            (f"结尾记忆点：{VALID_SCRIPT}", ()),
            (f"先拆第一层因果。{VALID_SCRIPT}", ()),
            (f"第二层看单方压制的利弊。{VALID_SCRIPT}", ()),
            (f"第三层给权衡框架。{VALID_SCRIPT}", ()),
            (f"详情见https://example.com。{VALID_SCRIPT}", ()),
            (f"公开材料已经确认核心变化【1】。{VALID_SCRIPT}", ()),
            (f"# 说明\n{VALID_SCRIPT}", ()),
        )
        for body, forbidden_phrases in cases:
            with self.subTest(body=body):
                task = ScriptTask(
                    topic="正文格式校验",
                    target_length=count_characters(body),
                    forbidden_phrases=forbidden_phrases,
                )
                response = output_payload(
                    body,
                    claim_usages=[
                        {"claim_id": "C001", "script_quote": VALID_QUOTE}
                    ],
                )
                llm = FakeLLM([response, response])
                with self.assertRaises(ScriptGenerationError):
                    await ScriptAgent(llm).generate(task, ready_research())

    async def test_background_generation_repairs_outline_label_leakage(self) -> None:
        research = replace(ready_research(), claims=(), title_chain=())
        polluted = f"结尾记忆点：{VALID_SCRIPT}"
        llm = FakeLLM(
            [background_output_payload(polluted), background_output_payload()]
        )

        artifact = await ScriptAgent(llm).generate(
            ScriptTask(
                topic="结构标签自动修复",
                target_length=count_characters(VALID_SCRIPT),
            ),
            research,
        )

        self.assertEqual(artifact.script_text, VALID_SCRIPT)
        self.assertEqual(artifact.generation_attempt_count, 2)
        repair_instruction = llm.calls[1][0][-1].content
        self.assertIn("outline label", repair_instruction)
        self.assertIn("natural spoken transition", repair_instruction)
        self.assertIn("完整 script_text 当作草稿做一次语义 rewrite", repair_instruction)
        self.assertIn("不是机械删除标签", repair_instruction)
        self.assertIn("不新增事实", repair_instruction)

    async def test_final_hy3_rewrite_runs_even_when_background_draft_is_clean(
        self,
    ) -> None:
        research = replace(ready_research(), claims=(), title_chain=())
        llm = FakeLLM(
            [background_output_payload(), final_rewrite_payload()]
        )

        artifact = await ScriptAgent(
            llm,
            config=ScriptGenerationConfig(final_rewrite_enabled=True),
        ).generate(
            ScriptTask(
                topic="每次交付前清洗",
                target_length=count_characters(VALID_SCRIPT),
            ),
            research,
        )

        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(artifact.script_text, VALID_SCRIPT)
        self.assertEqual(artifact.final_rewrite_attempt_count, 1)
        self.assertEqual(
            artifact.final_rewrite_prompt_version,
            "script-final-rewrite-1.0.0",
        )
        self.assertEqual(artifact.final_rewrite_draft_text, VALID_SCRIPT)
        self.assertEqual(
            [item.stage for item in artifact.llm_usages],
            ["script.generation", "script.final_rewrite"],
        )
        system_message, user_message = llm.calls[1][0]
        self.assertIn("终稿清洗编辑", system_message.content)
        self.assertIn("不是机械删除几个词", system_message.content)
        self.assertIn("即使草稿已经干净", user_message.content)

    async def test_final_hy3_rewrite_cleans_outline_labels_before_delivery(
        self,
    ) -> None:
        research = replace(ready_research(), claims=(), title_chain=())
        polluted = f"结尾记忆点：{VALID_SCRIPT}"
        llm = FakeLLM(
            [background_output_payload(polluted), final_rewrite_payload()]
        )

        artifact = await ScriptAgent(
            llm,
            config=ScriptGenerationConfig(final_rewrite_enabled=True),
        ).generate(
            ScriptTask(
                topic="清除提纲标签",
                target_length=count_characters(VALID_SCRIPT),
            ),
            research,
        )

        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(artifact.final_rewrite_draft_text, polluted)
        self.assertEqual(artifact.script_text, VALID_SCRIPT)
        self.assertIsNone(OUTLINE_LABEL_PATTERN.search(artifact.script_text))

    async def test_invalid_final_hy3_rewrite_is_never_delivered(self) -> None:
        research = replace(ready_research(), claims=(), title_chain=())
        llm = FakeLLM(
            [
                background_output_payload(),
                final_rewrite_payload(f"结尾记忆点：{VALID_SCRIPT}"),
            ]
        )

        with self.assertRaisesRegex(ScriptGenerationError, "was not frozen") as caught:
            await ScriptAgent(
                llm,
                config=ScriptGenerationConfig(final_rewrite_enabled=True),
            ).generate(
                ScriptTask(
                    topic="终稿清洗失败",
                    target_length=count_characters(VALID_SCRIPT),
                ),
                research,
            )

        self.assertEqual(
            [item.stage for item in caught.exception.llm_usages],
            ["script.generation", "script.final_rewrite"],
        )

    def test_literal_physical_layers_are_not_outline_labels(self) -> None:
        bodies = (
            "商场第一层的消防机制坏了，物业正在组织检修。",
            "皮肤的第一层结构很薄，这里说的是具体组织，不是文章提纲。",
        )
        for body in bodies:
            with self.subTest(body=body):
                ScriptAgent._validate_body(
                    body,
                    ScriptTask(topic="真实层级", target_length=50),
                )

    async def test_legal_risk_language_is_left_to_the_scoring_rubric(self) -> None:
        body = (
            f"{VALID_SCRIPT}发布前务必给当事人面部打码，才能避开隐私侵权的麻烦。"
        )
        response = output_payload(
            body,
            claim_usages=[{"claim_id": "C001", "script_quote": VALID_QUOTE}],
        )
        llm = FakeLLM([response, response])

        artifact = await ScriptAgent(llm).generate(
            ScriptTask(topic="隐私建议边界", target_length=count_characters(body)),
            ready_research(),
        )

        self.assertEqual(artifact.script_text, body)

    async def test_statutory_exception_language_is_left_to_the_scoring_rubric(
        self,
    ) -> None:
        body = (
            f"{VALID_SCRIPT}个人信息保护法列有公共利益例外，"
            "这给路人拍下冲突上传留出了合法空间。"
        )
        response = output_payload(
            body,
            claim_usages=[{"claim_id": "C001", "script_quote": VALID_QUOTE}],
        )
        llm = FakeLLM([response, response])

        artifact = await ScriptAgent(llm).generate(
            ScriptTask(topic="法律例外边界", target_length=count_characters(body)),
            ready_research(),
        )

        self.assertEqual(artifact.script_text, body)

    async def test_insufficient_research_prevents_any_generation_request(self) -> None:
        llm = FakeLLM([])
        task = ScriptTask(
            topic="证据不足的热点",
            target_length=count_characters(VALID_SCRIPT),
        )

        with self.assertRaisesRegex(ScriptGenerationError, "status 'ready'"):
            await ScriptAgent(llm).generate(
                task,
                ready_research(status="insufficient_evidence"),
            )
        self.assertEqual(llm.calls, [])

    async def test_current_research_without_title_chain_is_rejected_before_llm(
        self,
    ) -> None:
        llm = FakeLLM([])
        research = replace(
            ready_research(),
            evidence_prompt_version="research-evidence-2.10.8",
            title_chain=(),
        )

        with self.assertRaisesRegex(
            ScriptGenerationError,
            "missing the persisted title chain",
        ):
            await ScriptAgent(llm).generate(
                ScriptTask(topic="缺少标题链", target_length=100),
                research,
            )

        self.assertEqual(llm.calls, [])

    async def test_title_chain_cannot_reference_non_core_or_unsupported_claim(
        self,
    ) -> None:
        llm = FakeLLM([])
        invalid_chain = (
            *ready_research().title_chain[:2],
            TitleChainPart(
                component="question_predicate",
                status="covered",
                claim_ids=("C002",),
                reason="非核心背景不能承担标题答案。",
            ),
        )
        research = replace(ready_research(), title_chain=invalid_chain)

        with self.assertRaisesRegex(
            ScriptGenerationError,
            "non-core or unsupported claim",
        ):
            await ScriptAgent(llm).generate(
                ScriptTask(topic="错误标题链", target_length=100),
                research,
            )

        self.assertEqual(llm.calls, [])

    def test_prompt_structure_adapts_across_arbitrary_target_lengths(self) -> None:
        agent = ScriptAgent(FakeLLM([]))
        research = ready_research()
        supported_claims = tuple(
            claim for claim in research.claims if claim.support_status == "supported"
        )
        cases = (
            (120, "短稿只保留一个核心判断"),
            (280, "短稿只保留一个核心判断"),
            (360, "中篇按“问题判断—关键机制或取舍—普通人影响”展开"),
            (450, "中篇按“问题判断—关键机制或取舍—普通人影响”展开"),
            (700, "长稿让因果、取舍和现实影响形成三次自然推进"),
            (1000, "长稿让因果、取舍和现实影响形成三次自然推进"),
            (5000, "长稿让因果、取舍和现实影响形成三次自然推进"),
        )
        for target_length, expected in cases:
            with self.subTest(target_length=target_length):
                messages = agent._messages(
                    ScriptTask(topic="同一选题的不同篇幅", target_length=target_length),
                    research,
                    supported_claims,
                )
                self.assertIn(expected, messages[-1].content)
                if target_length > 550:
                    self.assertIn("不能复述‘第一层、第二层、第三层’", messages[-1].content)

    async def test_duplicate_research_claim_ids_prevent_generation(self) -> None:
        llm = FakeLLM([])
        research = ready_research()
        duplicate_claim = replace(research.claims[1], claim_id="C001")
        research = replace(
            research,
            claims=(research.claims[0], duplicate_claim, research.claims[2]),
        )
        task = ScriptTask(
            topic="上游引用标识异常的热点",
            target_length=count_characters(VALID_SCRIPT),
        )

        with self.assertRaisesRegex(ScriptGenerationError, "duplicate claim IDs"):
            await ScriptAgent(llm).generate(task, research)
        self.assertEqual(llm.calls, [])

    def test_rejects_invalid_generation_configuration(self) -> None:
        configs = (
            ScriptGenerationConfig(length_tolerance_ratio=-0.01),
            ScriptGenerationConfig(length_tolerance_ratio=0.51),
            ScriptGenerationConfig(max_generation_attempts=0),
            ScriptGenerationConfig(max_generation_attempts=4),
            ScriptGenerationConfig(grounding_review_enabled=1),
            ScriptGenerationConfig(final_rewrite_enabled=1),
        )
        for config in configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    ScriptAgent(FakeLLM([]), config=config)


if __name__ == "__main__":
    unittest.main()
