"""Tests for gate-aware aggregation semantics."""

from __future__ import annotations

from pathlib import Path
import unittest

from hyscript.config import PROJECT_ROOT
from hyscript.evaluation.aggregate import combine_evaluations, summarize_batch
from hyscript.evaluation.io import frozen_trace_from_payload
from hyscript.evaluation.models import (
    DimensionScore,
    EvaluationRecord,
    EvaluatorInfo,
    Finding,
    RubricRef,
)
from hyscript.evaluation.rubric import load_rubric


RUBRIC = load_rubric(PROJECT_ROOT / "eval/rubrics/script_quality_v2.json")
INITIAL_RUBRIC = load_rubric(
    PROJECT_ROOT / "eval/rubrics/script_quality_v1.json"
)
TRACE = frozen_trace_from_payload(
    {
        "schema_version": "1.0",
        "run_id": "aggregate-run",
        "task": {"topic": "测试"},
        "queries": [],
        "search_results": [],
        "selected_evidence": [],
        "claims": [],
        "script_artifact": {"script_text": "测试正文。"},
    },
    trace_sha256="a" * 64,
    source_path=Path("trace.json"),
)


def source_record(kind: str, *, gate: bool = False) -> EvaluationRecord:
    is_judge = kind == "judge"
    return EvaluationRecord(
        evaluation_id=f"{kind}-id",
        run_id=TRACE.run_id,
        trace_sha256=TRACE.trace_sha256,
        created_at="2026-08-27T00:00:00Z",
        evaluator=EvaluatorInfo(kind=kind, name=f"test-{kind}", version="1.0.0"),
        rubric=RubricRef(
            rubric_id=RUBRIC.rubric_id,
            version=RUBRIC.version,
            sha256=RUBRIC.sha256,
        ),
        status="completed",
        dimension_scores=tuple(
            DimensionScore(
                dimension_id=dimension.dimension_id,
                name=dimension.name,
                score=4,
                reason="测试分数。",
            )
            for dimension in RUBRIC.dimensions
            if dimension.evaluator == kind
        ),
        metrics={"weighted_average": 4.0, "normalized_score": 1.0} if is_judge else {},
        findings=(
            Finding(code="major_factual_error", severity="gate", message="测试门控。"),
        )
        if gate
        else (),
    )


class AggregateTests(unittest.TestCase):
    def test_initial_length_reward_is_added_to_seven_judge_scores(self) -> None:
        rubric_ref = RubricRef(
            INITIAL_RUBRIC.rubric_id,
            INITIAL_RUBRIC.version,
            INITIAL_RUBRIC.sha256,
        )
        judge = EvaluationRecord(
            evaluation_id="judge-v1",
            run_id=TRACE.run_id,
            trace_sha256=TRACE.trace_sha256,
            created_at="2026-08-27T00:00:00Z",
            evaluator=EvaluatorInfo("judge", "judge", "1.0.0"),
            rubric=rubric_ref,
            status="completed",
            dimension_scores=tuple(
                DimensionScore(d.dimension_id, d.name, 2, "测试分数。")
                for d in INITIAL_RUBRIC.judge_dimensions
            ),
            metrics={"weighted_average": 2.0, "normalized_score": 2 / 3},
        )
        rules = EvaluationRecord(
            evaluation_id="rules-v1",
            run_id=TRACE.run_id,
            trace_sha256=TRACE.trace_sha256,
            created_at="2026-08-27T00:00:00Z",
            evaluator=EvaluatorInfo("rules", "rules", "1.0.0"),
            rubric=rubric_ref,
            status="completed",
            dimension_scores=(
                DimensionScore("length_compliance", "字数符合度", 3, "测试分数。"),
            ),
        )

        combined = combine_evaluations(TRACE, INITIAL_RUBRIC, [judge, rules])

        self.assertEqual(combined.metrics["weighted_total"], 17.0)
        self.assertAlmostEqual(combined.metrics["final_score"], 17 / 24)

    def test_gate_preserves_diagnostic_score_but_removes_final_score(self) -> None:
        combined = combine_evaluations(
            TRACE,
            RUBRIC,
            [source_record("rules", gate=True), source_record("judge")],
        )

        self.assertTrue(combined.gate_failed)
        self.assertEqual(combined.metrics["judge_normalized_score"], 1.0)
        self.assertFalse(combined.metrics["eligible"])
        self.assertIsNone(combined.metrics["final_score"])

    def test_ungated_judge_score_is_eligible_and_summarized(self) -> None:
        combined = combine_evaluations(
            TRACE,
            RUBRIC,
            [source_record("rules"), source_record("judge")],
        )
        summary = summarize_batch([combined])

        self.assertTrue(combined.metrics["eligible"])
        self.assertEqual(combined.metrics["final_score"], 1.0)
        self.assertEqual(combined.metrics["normalized_score"], 1.0)
        self.assertEqual(len(combined.dimension_scores), 9)
        self.assertEqual(summary["eligible_count"], 1)
        self.assertEqual(summary["final_score_mean"], 1.0)

    def test_missing_rule_dimension_prevents_final_score(self) -> None:
        combined = combine_evaluations(TRACE, RUBRIC, [source_record("judge")])

        self.assertFalse(combined.metrics["eligible"])
        self.assertIsNone(combined.metrics["normalized_score"])
        self.assertIsNone(combined.metrics["final_score"])
        self.assertEqual(combined.metrics["score_coverage"], 8 / 9)

    def test_rejects_ambiguous_duplicate_evaluator_kinds(self) -> None:
        with self.assertRaisesRegex(ValueError, "one source record"):
            combine_evaluations(
                TRACE,
                RUBRIC,
                [source_record("judge"), source_record("judge")],
            )


if __name__ == "__main__":
    unittest.main()
