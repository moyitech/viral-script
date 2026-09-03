"""Offline tests for Tavily-backed explicit-citation verification."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from hyscript.config import PROJECT_ROOT
from hyscript.evaluation.citation_verification import (
    CitationVerificationConfig,
    CitationVerifier,
    explicit_citation_candidates,
    run_citation_verification_formal_validation,
    run_citation_verification_validation,
)
from hyscript.evaluation.io import load_frozen_trace
from hyscript.llm import ChatResponse
from hyscript.search import SearchResponse, SearchResult


FAKE_CITATION = (
    "据联合国数字生活委员会2026年白皮书统计，"
    "这一结论已获97.3%的专家一致确认。"
)


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = []

    async def complete(self, messages, *, reasoning_effort="no_think") -> ChatResponse:
        self.calls.append((messages, reasoning_effort))
        return ChatResponse(
            content=self.responses.pop(0),
            model="fake-hy3",
            usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        )


class FakeSearchProvider:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.calls: list[tuple[str, int]] = []
        self.delay = delay
        self.active = 0
        self.maximum = 0

    async def search(self, query: str, *, limit: int = 20) -> SearchResponse:
        self.calls.append((query, limit))
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        if self.delay:
            await asyncio.sleep(self.delay)
        self.active -= 1
        return SearchResponse(
            provider="tavily",
            query=query,
            results=(
                SearchResult(
                    rank=1,
                    title="无关的联合国新闻",
                    url="https://example.com/unrelated",
                    snippet="搜索结果没有出现所述委员会、白皮书或统计数字。",
                ),
            ),
            request_id="search-1",
            response_time=0.1,
            usage={"credits": 1},
        )


def _trace(path: Path, *, run_id: str, text: str) -> None:
    payload = json.loads(
        (PROJECT_ROOT / "eval/traces/example_trace.json").read_text(encoding="utf-8")
    )
    payload["run_id"] = run_id
    payload["task"]["topic"] = "测试选题"
    payload["script_artifact"]["script_text"] = text
    payload["script_artifact"]["character_count"] = len(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _formal_manifest(root: Path, rows: list[tuple[str, str]]) -> None:
    tasks = []
    for task_id, text in rows:
        trace_path = root / "generation/traces" / f"{task_id}.json"
        run_id = f"run-{task_id}"
        _trace(trace_path, run_id=run_id, text=text)
        trace = load_frozen_trace(trace_path)
        tasks.append(
            {
                "task_id": task_id,
                "status": "completed",
                "run_id": run_id,
                "trace": f"traces/{task_id}.json",
                "trace_sha256": trace.trace_sha256,
            }
        )
    (root / "generation/trace_manifest.json").write_text(
        json.dumps(
            {
                "expected_count": len(tasks),
                "selected_count": len(tasks),
                "tasks": tasks,
            }
        ),
        encoding="utf-8",
    )


def _verdict(span: str, *, detected: bool = True) -> str:
    return json.dumps(
        {
            "fabricated_citation": detected,
            "reason": "精确出处没有得到冻结材料或定向搜索的直接支持。",
            "citations": [
                {
                    "script_span": span,
                    "status": "unverified" if detected else "supported",
                    "reason": "搜索结果与所述机构、年份和数字不匹配。",
                    "evidence_urls": [],
                }
            ],
        },
        ensure_ascii=False,
    )


class CitationVerificationTests(unittest.TestCase):
    def test_candidate_extraction_does_not_use_answer_labels(self) -> None:
        text = "这是普通口播。" + FAKE_CITATION + "最后一句。"
        self.assertEqual(
            explicit_citation_candidates(text, maximum=3),
            (FAKE_CITATION,),
        )

    def test_no_explicit_citation_skips_search_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.json"
            _trace(trace_path, run_id="validation-D01-004", text="自然口播正文。")
            client = FakeClient([])
            search = FakeSearchProvider()
            verifier = CitationVerifier(
                client,
                search,
                model_name="fake-hy3",
            )
            result = asyncio.run(
                verifier.evaluate(
                    load_frozen_trace(trace_path),
                    blind_case_id="D01-004",
                    expected_attack_type="secret-label",
                    source_trace="trace.json",
                )
            )
            self.assertFalse(result["fabricated_citation"])
            self.assertEqual(result["detection_source"], "no_explicit_citation")
            self.assertEqual(search.calls, [])
            self.assertEqual(client.calls, [])

    def test_unverified_citation_uses_tavily_and_records_auditable_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.json"
            _trace(trace_path, run_id="validation-D01-004", text=FAKE_CITATION)
            client = FakeClient([_verdict(FAKE_CITATION)])
            search = FakeSearchProvider()
            verifier = CitationVerifier(
                client,
                search,
                model_name="fake-hy3",
            )
            result = asyncio.run(
                verifier.evaluate(
                    load_frozen_trace(trace_path),
                    blind_case_id="D01-004",
                    expected_attack_type="hidden-from-prompt",
                    source_trace="trace.json",
                )
            )
            self.assertTrue(result["fabricated_citation"])
            self.assertEqual(result["detection_source"], "hy3_with_tavily")
            self.assertEqual(len(search.calls), 1)
            self.assertEqual(search.calls[0][1], 5)
            self.assertEqual(result["searches"][0]["request_id"], "search-1")
            prompt = client.calls[0][0][0].content
            self.assertIn("https://example.com/unrelated", prompt)
            self.assertNotIn("hidden-from-prompt", prompt)

    def test_invalid_json_gets_one_format_repair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.json"
            _trace(trace_path, run_id="validation-D01-004", text=FAKE_CITATION)
            client = FakeClient(["not-json", _verdict(FAKE_CITATION)])
            verifier = CitationVerifier(
                client,
                FakeSearchProvider(),
                model_name="fake-hy3",
            )
            result = asyncio.run(
                verifier.evaluate(
                    load_frozen_trace(trace_path),
                    blind_case_id="D01-004",
                    expected_attack_type="fabricated_citation",
                    source_trace="trace.json",
                )
            )
            self.assertTrue(result["fabricated_citation"])
            self.assertEqual(result["format_attempt_count"], 1)
            self.assertEqual(len(client.calls), 2)

    def test_generic_unverified_research_is_not_forced_to_fabricated(self) -> None:
        generic = "有研究发现，规律会随时间变化。"
        response = json.dumps(
            {
                "fabricated_citation": False,
                "reason": "只有泛称，没有可识别的机构、文献、年份或精确数字。",
                "citations": [
                    {
                        "script_span": generic,
                        "status": "unverified",
                        "reason": "现有材料无法直接核验这一泛称。",
                        "evidence_urls": [],
                    }
                ],
            },
            ensure_ascii=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.json"
            _trace(trace_path, run_id="validation-D01-004", text=generic)
            verifier = CitationVerifier(
                FakeClient([response]),
                FakeSearchProvider(),
                model_name="fake-hy3",
            )
            result = asyncio.run(
                verifier.evaluate(
                    load_frozen_trace(trace_path),
                    blind_case_id="D01-004",
                    expected_attack_type="normal_output",
                    source_trace="trace.json",
                )
            )
            self.assertFalse(result["fabricated_citation"])
            self.assertEqual(result["citations"][0]["status"], "unverified")

    def test_search_concurrency_stays_bounded(self) -> None:
        text = FAKE_CITATION + "根据另一项研究报告显示，这个数字也得到确认。"
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.json"
            _trace(trace_path, run_id="validation-D01-004", text=text)
            candidates = explicit_citation_candidates(text, maximum=3)
            response = json.dumps(
                {
                    "fabricated_citation": True,
                    "reason": "均无法核验。",
                    "citations": [
                        {
                            "script_span": span,
                            "status": "unverified",
                            "reason": "没有直接支持。",
                            "evidence_urls": [],
                        }
                        for span in candidates
                    ],
                },
                ensure_ascii=False,
            )
            search = FakeSearchProvider(delay=0.01)
            verifier = CitationVerifier(
                FakeClient([response]),
                search,
                model_name="fake-hy3",
                search_concurrency=1,
            )
            asyncio.run(
                verifier.evaluate(
                    load_frozen_trace(trace_path),
                    blind_case_id="D01-004",
                    expected_attack_type="fabricated_citation",
                    source_trace="trace.json",
                )
            )
            self.assertEqual(len(search.calls), 2)
            self.assertEqual(search.maximum, 1)

    def test_batch_uses_other_attacks_as_controls_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            discrimination = root / "validation/discrimination"
            _trace(
                discrimination / "traces/D01-004.json",
                run_id="validation-D01-004",
                text=FAKE_CITATION,
            )
            _trace(
                discrimination / "traces/D02-008.json",
                run_id="validation-D02-008",
                text="普通正文后重复重复重复。",
            )
            answer_key = [
                {
                    "blind_case_id": "D01-004",
                    "expected_tier": "attack",
                    "attack_type": "fabricated_citation",
                },
                {
                    "blind_case_id": "D02-008",
                    "expected_tier": "attack",
                    "attack_type": "repetition",
                },
            ]
            (discrimination / "answer_key.json").write_text(
                json.dumps(answer_key),
                encoding="utf-8",
            )
            client = FakeClient([_verdict(FAKE_CITATION)])
            verifier = CitationVerifier(
                client,
                FakeSearchProvider(),
                model_name="fake-hy3",
            )
            first = asyncio.run(
                run_citation_verification_validation(root, verifier, concurrency=2)
            )
            second = asyncio.run(
                run_citation_verification_validation(root, verifier, concurrency=2)
            )
            self.assertEqual(first["true_positive_count"], 1)
            self.assertEqual(first["false_positive_count"], 0)
            self.assertEqual(first["search_request_count"], 1)
            self.assertEqual(second["completed_count"], 2)
            manifest = json.loads(
                (root / "validation/citation_verification/manifest.json").read_text()
            )
            self.assertEqual(manifest["resumed_count"], 2)

    def test_formal_batch_resumes_and_preserves_quality_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _formal_manifest(
                root,
                [
                    ("T001-L280", FAKE_CITATION),
                    ("T002-L280", "自然口播正文。"),
                ],
            )
            score_path = root / "results/summary.json"
            score_path.parent.mkdir(parents=True)
            score_path.write_text('{"frozen":true}', encoding="utf-8")
            search = FakeSearchProvider()
            client = FakeClient([_verdict(FAKE_CITATION)])
            verifier = CitationVerifier(
                client,
                search,
                model_name="fake-hy3",
                search_concurrency=8,
            )

            first = asyncio.run(
                run_citation_verification_formal_validation(
                    root,
                    verifier,
                    concurrency=512,
                )
            )
            second = asyncio.run(
                run_citation_verification_formal_validation(
                    root,
                    verifier,
                    concurrency=512,
                )
            )

            self.assertEqual(first["completed_count"], 2)
            self.assertEqual(first["flagged_count"], 1)
            self.assertEqual(first["candidate_output_count"], 1)
            self.assertEqual(first["search_request_count"], 1)
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(second["completed_count"], 2)
            manifest = json.loads(
                (
                    root / "validation/incremental_attack/citation/manifest.json"
                ).read_text()
            )
            self.assertEqual(manifest["resumed_count"], 2)
            self.assertEqual(manifest["concurrency"], 512)
            self.assertEqual(manifest["search_concurrency"], 8)
            self.assertEqual(score_path.read_text(encoding="utf-8"), '{"frozen":true}')

    def test_formal_batch_rejects_trace_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _formal_manifest(root, [("T001-L280", "自然口播正文。")])
            manifest_path = root / "generation/trace_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["tasks"][0]["trace_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                asyncio.run(
                    run_citation_verification_formal_validation(
                        root,
                        CitationVerifier(
                            FakeClient([]),
                            FakeSearchProvider(),
                            model_name="fake-hy3",
                        ),
                    )
                )

    def test_concurrency_limits_reject_values_above_provider_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 64"):
            CitationVerifier(
                FakeClient([]),
                FakeSearchProvider(),
                model_name="fake-hy3",
                search_concurrency=65,
            )
        with tempfile.TemporaryDirectory() as directory:
            verifier = CitationVerifier(
                FakeClient([]),
                FakeSearchProvider(),
                model_name="fake-hy3",
            )
            with self.assertRaisesRegex(ValueError, "between 1 and 512"):
                asyncio.run(
                    run_citation_verification_validation(
                        Path(directory),
                        verifier,
                        concurrency=513,
                    )
                )


if __name__ == "__main__":
    unittest.main()
