"""Offline tests for evidence-grounded oral-script generation."""

from __future__ import annotations

from dataclasses import replace
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
)
from hyscript.config import ScriptGenerationConfig
from hyscript.llm import ChatResponse


VALID_SCRIPT = (
    "公开材料已经确认核心变化。它影响的不是所有人，而是符合特定条件的办理者。"
    "普通人先核对适用范围，再按照正式说明准备材料，能减少误读和无效操作。"
)
VALID_QUOTE = "公开材料已经确认核心变化"


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
            ),
            Evidence(
                evidence_id="E002",
                result_ref="R002",
                title="来源二",
                url="https://news.example/two",
                excerpt="变化只适用于符合特定条件的办理者。",
                source_query="适用范围",
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


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[object, dict[str, object]]] = []

    async def complete(self, messages, **kwargs) -> ChatResponse:
        self.calls.append((messages, kwargs))
        if not self.responses:
            raise AssertionError("Unexpected LLM request")
        call_number = len(self.calls)
        return ChatResponse(
            content=self.responses.pop(0),
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
        self.assertEqual(artifact.prompt_version, "script-generation-1.0.0")
        self.assertEqual(artifact.generation_attempt_count, 1)
        self.assertEqual(len(artifact.llm_usages), 1)
        self.assertEqual(artifact.llm_usages[0].input_tokens, 80)
        self.assertEqual(artifact.llm_usages[0].output_tokens, 30)
        self.assertEqual(artifact.llm_usages[0].reasoning_tokens, 6)
        self.assertEqual(llm.calls[0][1], {"reasoning_effort": "high"})
        system_message, user_message = llm.calls[0][0]
        self.assertIn("不可信外部数据", system_message.content)
        self.assertIn('"claim_id": "C001"', user_message.content)
        self.assertNotIn('"claim_id": "C003"', user_message.content)
        self.assertNotIn("max_tokens", user_message.content)
        self.assertTrue(any("[4/5]" in message for message in logs.output))
        self.assertTrue(any("文案生成完成" in message for message in logs.output))

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
        )
        for config in configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    ScriptAgent(FakeLLM([]), config=config)


if __name__ == "__main__":
    unittest.main()
