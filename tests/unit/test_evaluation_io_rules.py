"""Tests for frozen-trace loading, deterministic rules, and independent writes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
import unittest

from hyscript.artifacts.trace import RunTrace
from hyscript.config import PROJECT_ROOT
from hyscript.evaluation.io import (
    ResultWriteError,
    TraceInputError,
    load_frozen_trace,
    write_evaluation_record,
)
from hyscript.evaluation.rubric import load_rubric
from hyscript.evaluation.rules import RuleEvaluator


RUBRIC_PATH = PROJECT_ROOT / "eval/rubrics/script_quality_v1.json"


def trace_payload(
    *,
    run_id: str = "run-001",
    script_text: str = "真正影响睡眠的，不只是几点上床。白天的光照和活动量也很重要。",
    target_length: int = 31,
) -> dict:
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": "2026-08-27T00:00:00Z",
        "task": {
            "topic": "怎样改善睡眠",
            "target_length": target_length,
            "forbidden_phrases": ["包治百病"],
        },
        "queries": ["改善睡眠 光照 活动量"],
        "search_results": [{"title": "source"}],
        "selected_evidence": [
            {
                "evidence_id": "ev-1",
                "url": "https://example.com/sleep",
                "snippet": "日间光照和规律活动有助于睡眠节律。",
            }
        ],
        "claims": [
            {
                "claim_id": "claim-1",
                "text": "白天光照和活动量会影响睡眠",
                "is_core": True,
                "evidence_ids": ["ev-1"],
                "support_status": "supported",
            }
        ],
        "script_artifact": {"script_text": script_text},
    }


def write_trace(path: Path, payload: dict) -> bytes:
    content = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
    path.write_bytes(content)
    return content


class FrozenTraceIoTests(unittest.TestCase):
    def test_loads_trace_and_hashes_exact_bytes(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            content = write_trace(path, trace_payload())

            trace = load_frozen_trace(path)

        self.assertEqual(trace.run_id, "run-001")
        self.assertEqual(trace.trace_sha256, hashlib.sha256(content).hexdigest())
        self.assertEqual(trace.script_text[:4], "真正影响")

    def test_rejects_unsafe_run_id_and_missing_script_contract(self) -> None:
        with TemporaryDirectory() as directory:
            unsafe = Path(directory) / "unsafe.json"
            write_trace(unsafe, trace_payload(run_id="../escape"))
            with self.assertRaises(TraceInputError):
                load_frozen_trace(unsafe)

            missing = trace_payload()
            missing["script_artifact"] = {"text": "non-canonical"}
            write_trace(unsafe, missing)
            with self.assertRaisesRegex(TraceInputError, "script_text"):
                load_frozen_trace(unsafe)

    def test_result_write_never_changes_trace_and_is_exclusive(self) -> None:
        rubric = load_rubric(RUBRIC_PATH)
        with TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.json"
            original = write_trace(trace_path, trace_payload())
            trace = load_frozen_trace(trace_path)
            result = RuleEvaluator().evaluate(trace, rubric)
            result_path = Path(directory) / "result.json"

            write_evaluation_record(result_path, result)

            self.assertEqual(trace_path.read_bytes(), original)
            self.assertEqual(stat.S_IMODE(result_path.stat().st_mode), 0o600)
            with self.assertRaises(ResultWriteError):
                write_evaluation_record(result_path, result)

    def test_rejects_duplicate_ids_and_claims_without_core_marker(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            duplicate_evidence = trace_payload()
            duplicate_evidence["selected_evidence"].append(
                dict(duplicate_evidence["selected_evidence"][0])
            )
            write_trace(path, duplicate_evidence)
            with self.assertRaisesRegex(TraceInputError, "must be unique"):
                load_frozen_trace(path)

            no_core = trace_payload()
            no_core["claims"][0]["is_core"] = False
            write_trace(path, no_core)
            with self.assertRaisesRegex(TraceInputError, "at least one claim as core"):
                load_frozen_trace(path)

    def test_rejects_invalid_task_and_search_result_contracts(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            invalid_length = trace_payload()
            invalid_length["task"]["target_length"] = True
            write_trace(path, invalid_length)
            with self.assertRaisesRegex(TraceInputError, "positive integer"):
                load_frozen_trace(path)

            invalid_result = trace_payload()
            invalid_result["search_results"] = ["not-an-object"]
            write_trace(path, invalid_result)
            with self.assertRaisesRegex(TraceInputError, "list of objects"):
                load_frozen_trace(path)

    def test_run_trace_freeze_is_atomic_and_exclusive_by_default(self) -> None:
        trace = RunTrace(
            run_id="frozen-run",
            created_at="2026-08-27T00:00:00Z",
            task={"topic": "测试"},
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            trace.write_json(path)
            original = path.read_bytes()

            with self.assertRaises(FileExistsError):
                trace.write_json(path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)


class RuleEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rubric = load_rubric(RUBRIC_PATH)

    def _evaluate(self, payload: dict):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            write_trace(path, payload)
            return RuleEvaluator().evaluate(load_frozen_trace(path), self.rubric)

    def test_reports_length_and_complete_claim_coverage(self) -> None:
        payload = trace_payload(target_length=31)
        record = self._evaluate(payload)

        self.assertEqual(record.metrics["claim_citation_coverage"], 1.0)
        self.assertEqual(record.metrics["core_claim_support_rate"], 1.0)
        self.assertIsInstance(record.metrics["length_deviation_ratio"], float)
        self.assertFalse(record.gate_failed)

    def test_fabricated_and_unsupported_core_citation_are_gates(self) -> None:
        payload = trace_payload()
        payload["claims"][0]["evidence_ids"] = ["ev-does-not-exist"]

        record = self._evaluate(payload)
        codes = {
            finding.code for finding in record.findings if finding.severity == "gate"
        }

        self.assertIn("fabricated_citation", codes)
        self.assertIn("unsupported_core_claim", codes)

    def test_detects_meta_output_repetition_and_forbidden_phrase(self) -> None:
        payload = trace_payload(
            script_text=(
                "<think>内部推理</think>关键点剖析：这篇文案完全符合评分标准。"
                "哈哈哈哈哈哈哈哈哈哈。包治百病。"
            )
        )

        record = self._evaluate(payload)
        codes = {
            finding.code for finding in record.findings if finding.severity == "gate"
        }

        self.assertIn("reasoning_leakage", codes)
        self.assertIn("non_script_analysis", codes)
        self.assertIn("meta_evaluation", codes)
        self.assertIn("repetition_padding", codes)
        self.assertIn("forbidden_phrase", codes)

    def test_missing_target_length_is_not_guessed(self) -> None:
        payload = trace_payload()
        payload["task"].pop("target_length")

        record = self._evaluate(payload)

        self.assertIsNone(record.metrics["length_deviation_ratio"])
        self.assertIn(
            "target_length_missing", {finding.code for finding in record.findings}
        )

    def test_normal_discussion_of_scoring_or_writing_is_not_meta_output(self) -> None:
        payload = trace_payload(
            script_text="考试评分标准怎么读？先看采分点。我的写作思路是先列事实，再给结论。"
        )

        record = self._evaluate(payload)
        codes = {finding.code for finding in record.findings}

        self.assertNotIn("non_script_analysis", codes)
        self.assertNotIn("meta_evaluation", codes)


if __name__ == "__main__":
    unittest.main()
