"""Tests for batch evaluation isolation, persistence, and resume behavior."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hyscript.config import PROJECT_ROOT
from hyscript.evaluation import (
    BatchEvaluationConfig,
    BatchEvaluationRunner,
    Hy3JudgeEvaluator,
    JudgeConfig,
    RuleConfig,
    RuleEvaluator,
    load_rubric,
)
from hyscript.llm import ChatResponse


RUBRIC = load_rubric(PROJECT_ROOT / "eval/rubrics/script_quality_v2.json")


def write_trace(
    path: Path, run_id: str, script_text: str = "这是一段可以直接口播的正文。"
) -> None:
    payload = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": "2026-08-27T00:00:00Z",
        "task": {"topic": "测试选题", "target_length": 14},
        "queries": ["测试选题"],
        "search_results": [],
        "selected_evidence": [],
        "claims": [],
        "script_artifact": {"script_text": script_text},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def judge_output(script_span: str = "这是一段") -> str:
    scores = {
        dimension.dimension_id: {
            "score": 3,
            "reason": "测试评审结果。",
            "script_spans": [script_span],
            "evidence_ids": [],
        }
        for dimension in RUBRIC.judge_dimensions
    }
    return json.dumps(
        {"summary": "测试评审完成。", "scores": scores, "gates": []},
        ensure_ascii=False,
    )


class StaticJudgeClient:
    async def complete(self, messages, *, reasoning_effort="no_think"):
        return ChatResponse(
            content=judge_output(),
            model="hy3-test",
            request_id="judge-request",
            usage={"prompt_tokens": 10, "completion_tokens": 10},
        )


def judge(config: JudgeConfig | None = None, *, model: str = "hy3-test"):
    return Hy3JudgeEvaluator(
        StaticJudgeClient(),
        model_name=model,
        config=config,
        sampling_parameters={"temperature": 0.0, "top_p": 1.0},
    )


class BatchEvaluationRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_writes_separate_records_summary_and_manifest(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.json"
            output_dir = root / "results"
            write_trace(trace_path, "run-1")
            runner = BatchEvaluationRunner(
                RUBRIC,
                BatchEvaluationConfig(output_dir=output_dir),
            )

            result = await runner.run([trace_path])

            self.assertEqual(result.failed_count, 0)
            self.assertEqual(result.outcomes[0].status, "completed")
            self.assertTrue((output_dir / "items/run-1/rules.json").is_file())
            self.assertTrue((output_dir / "items/run-1/combined.json").is_file())
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["counts"]["completed"], 1)
            self.assertEqual(summary["counts_scope"], "current_invocation")
            self.assertEqual(
                summary["record_coverage"],
                {
                    "input_trace_count": 1,
                    "validated_trace_count": 1,
                    "combined_record_count": 1,
                    "unavailable_record_count": 0,
                    "complete": True,
                },
            )
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "completed")

    async def test_second_run_resumes_without_rewriting_evaluator_result(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.json"
            output_dir = root / "results"
            write_trace(trace_path, "run-2")
            config = BatchEvaluationConfig(output_dir=output_dir)
            first = await BatchEvaluationRunner(RUBRIC, config).run([trace_path])
            rules_path = output_dir / "items/run-2/rules.json"
            original = rules_path.read_bytes()

            resumed = await BatchEvaluationRunner(RUBRIC, config).run([trace_path])

            self.assertEqual(resumed.outcomes[0].status, "skipped")
            self.assertEqual(resumed.evaluation_id, first.evaluation_id)
            self.assertEqual(rules_path.read_bytes(), original)
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary["counts"],
                {"input": 1, "completed": 0, "skipped": 1, "failed": 0},
            )
            self.assertEqual(summary["record_coverage"]["combined_record_count"], 1)
            self.assertTrue(summary["record_coverage"]["complete"])

    async def test_invalid_trace_does_not_stop_valid_trace(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid.json"
            invalid_path = root / "invalid.json"
            output_dir = root / "results"
            write_trace(valid_path, "run-valid")
            invalid_path.write_text("not json", encoding="utf-8")

            result = await BatchEvaluationRunner(
                RUBRIC,
                BatchEvaluationConfig(output_dir=output_dir),
            ).run([invalid_path, valid_path])

            self.assertEqual(result.failed_count, 1)
            self.assertTrue((output_dir / "items/run-valid/combined.json").is_file())
            failures = json.loads(
                (output_dir / "failures.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failures["items"][0]["error_code"], "invalid_trace")
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary["record_coverage"],
                {
                    "input_trace_count": 2,
                    "validated_trace_count": 1,
                    "combined_record_count": 1,
                    "unavailable_record_count": 1,
                    "complete": False,
                },
            )

    async def test_trace_change_requires_explicit_overwrite(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.json"
            output_dir = root / "results"
            write_trace(trace_path, "run-3", "第一个版本。")
            await BatchEvaluationRunner(
                RUBRIC, BatchEvaluationConfig(output_dir=output_dir)
            ).run([trace_path])
            write_trace(trace_path, "run-3", "第二个版本，内容发生变化。")

            conflicted = await BatchEvaluationRunner(
                RUBRIC, BatchEvaluationConfig(output_dir=output_dir)
            ).run([trace_path])
            overwritten = await BatchEvaluationRunner(
                RUBRIC,
                BatchEvaluationConfig(output_dir=output_dir, overwrite=True),
            ).run([trace_path])

            self.assertEqual(conflicted.outcomes[0].error_code, "resume_conflict")
            self.assertEqual(overwritten.outcomes[0].status, "completed")

    async def test_evaluator_set_change_is_a_resume_conflict(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.json"
            output_dir = root / "results"
            write_trace(trace_path, "run-set")
            first = await BatchEvaluationRunner(
                RUBRIC,
                BatchEvaluationConfig(
                    output_dir=output_dir,
                    evaluators=("rules", "judge"),
                ),
                judge_evaluator=judge(),
            ).run([trace_path])
            old_combined = (output_dir / "items/run-set/combined.json").read_bytes()

            conflicted = await BatchEvaluationRunner(
                RUBRIC,
                BatchEvaluationConfig(output_dir=output_dir, evaluators=("rules",)),
            ).run([trace_path])

            self.assertEqual(conflicted.outcomes[0].error_code, "resume_conflict")
            self.assertEqual(
                (output_dir / "items/run-set/combined.json").read_bytes(),
                old_combined,
            )

            overwritten = await BatchEvaluationRunner(
                RUBRIC,
                BatchEvaluationConfig(
                    output_dir=output_dir,
                    evaluators=("rules",),
                    overwrite=True,
                ),
            ).run([trace_path])
            combined = json.loads(
                (output_dir / "items/run-set/combined.json").read_text(
                    encoding="utf-8"
                )
            )
            source_kinds = [
                source["kind"] for source in combined["metadata"]["source_evaluations"]
            ]
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )

            self.assertEqual(overwritten.evaluation_id, first.evaluation_id)
            self.assertEqual(source_kinds, ["rules"])
            self.assertIsNone(combined["metrics"]["judge_normalized_score"])
            self.assertEqual(manifest["evaluators"], ["rules"])

    async def test_rule_and_judge_config_changes_are_resume_conflicts(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.json"
            write_trace(trace_path, "run-fingerprint")

            rules_output = root / "rules-results"
            await BatchEvaluationRunner(
                RUBRIC,
                BatchEvaluationConfig(output_dir=rules_output),
            ).run([trace_path])
            changed_rules = await BatchEvaluationRunner(
                RUBRIC,
                BatchEvaluationConfig(output_dir=rules_output),
                rule_evaluator=RuleEvaluator(RuleConfig(length_tolerance_ratio=0.2)),
            ).run([trace_path])
            self.assertEqual(
                changed_rules.outcomes[0].error_code,
                "resume_conflict",
            )

            judge_output_dir = root / "judge-results"
            judge_config = BatchEvaluationConfig(
                output_dir=judge_output_dir,
                evaluators=("judge",),
            )
            await BatchEvaluationRunner(
                RUBRIC,
                judge_config,
                judge_evaluator=judge(JudgeConfig(reasoning_effort="low")),
            ).run([trace_path])
            changed_judge = await BatchEvaluationRunner(
                RUBRIC,
                judge_config,
                judge_evaluator=judge(JudgeConfig(reasoning_effort="high")),
            ).run([trace_path])
            self.assertEqual(
                changed_judge.outcomes[0].error_code,
                "resume_conflict",
            )

    async def test_stale_combined_sources_are_rebuilt_from_current_records(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.json"
            output_dir = root / "results"
            write_trace(trace_path, "run-derived")
            config = BatchEvaluationConfig(output_dir=output_dir)
            await BatchEvaluationRunner(RUBRIC, config).run([trace_path])
            rules = json.loads(
                (output_dir / "items/run-derived/rules.json").read_text(
                    encoding="utf-8"
                )
            )
            combined_path = output_dir / "items/run-derived/combined.json"
            combined = json.loads(combined_path.read_text(encoding="utf-8"))
            combined["metadata"]["source_evaluations"][0]["evaluation_id"] = "stale"
            combined_path.write_text(json.dumps(combined), encoding="utf-8")

            resumed = await BatchEvaluationRunner(RUBRIC, config).run([trace_path])
            rebuilt = json.loads(combined_path.read_text(encoding="utf-8"))

            self.assertEqual(resumed.outcomes[0].status, "skipped")
            self.assertEqual(
                rebuilt["metadata"]["source_evaluations"][0]["evaluation_id"],
                rules["evaluation_id"],
            )


if __name__ == "__main__":
    unittest.main()
