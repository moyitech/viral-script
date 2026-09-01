"""Tests for versioned evaluation rubrics and record contracts."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hyscript.config import PROJECT_ROOT
from hyscript.evaluation.models import (
    DimensionScore,
    EvaluationRecord,
    EvaluatorInfo,
    Finding,
    RubricRef,
    evaluation_record_from_dict,
)
from hyscript.evaluation.rubric import RubricError, load_rubric


RUBRIC_PATH = PROJECT_ROOT / "eval/rubrics/script_quality_v1.json"
CURRENT_RUBRIC_PATH = PROJECT_ROOT / "eval/rubrics/script_quality_v2.json"
HEX_DIGEST = "a" * 64


class RubricTests(unittest.TestCase):
    def test_default_rubric_has_seven_judge_dimensions_plus_length(self) -> None:
        rubric = load_rubric(RUBRIC_PATH)

        self.assertEqual(rubric.rubric_id, "script_quality")
        self.assertEqual(rubric.version, "1.1.0")
        self.assertEqual((rubric.score_min, rubric.score_max), (1, 3))
        self.assertEqual(len(rubric.dimensions), 8)
        self.assertEqual(len(set(rubric.dimension_ids)), 8)
        self.assertEqual(len(rubric.judge_dimensions), 7)
        self.assertEqual(len(rubric.rule_dimensions), 1)
        self.assertTrue(
            all(len(dimension.anchors) == 3 for dimension in rubric.dimensions)
        )
        self.assertEqual(
            rubric.rule_dimensions[0].dimension_id,
            "length_compliance",
        )
        self.assertIn("theme_information", rubric.judge_dimension_ids)
        self.assertIn("rhetoric_memorability", rubric.judge_dimension_ids)
        self.assertNotIn("factual_reference_consistency", rubric.dimension_ids)

    def test_current_rubric_adds_rule_scored_length_dimension(self) -> None:
        rubric = load_rubric(CURRENT_RUBRIC_PATH)

        self.assertEqual(rubric.version, "2.3.0")
        self.assertEqual(len(rubric.dimensions), 9)
        self.assertEqual(len(rubric.judge_dimensions), 8)
        self.assertEqual(len(rubric.rule_dimensions), 1)
        length = rubric.rule_dimensions[0]
        self.assertEqual(length.dimension_id, "length_compliance")
        self.assertEqual(length.weight, 1.0)
        self.assertIn("unsupported_core_claim", rubric.judge_gate_codes)
        self.assertIn("evidence_traceability_incomplete", rubric.judge_gate_codes)

    def test_rejects_missing_anchor_and_duplicate_dimension(self) -> None:
        payload = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))
        payload["dimensions"][0]["anchors"].pop("3")
        payload["dimensions"].append(payload["dimensions"][1])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(RubricError):
                load_rubric(path)

    def test_rejects_unknown_top_level_field(self) -> None:
        payload = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))
        payload["surprise"] = True
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RubricError, "unsupported fields"):
                load_rubric(path)

    def test_rejects_non_finite_dimension_weight(self) -> None:
        payload = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))
        payload["dimensions"][0]["weight"] = float("nan")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RubricError, "greater than zero"):
                load_rubric(path)


class EvaluationModelTests(unittest.TestCase):
    def test_dimension_score_rejects_bool_and_out_of_range(self) -> None:
        with self.assertRaises(TypeError):
            DimensionScore("topic", "Topic", True, "reason")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            DimensionScore("topic", "Topic", 5, "reason")

    def test_record_round_trips_for_resume(self) -> None:
        record = EvaluationRecord(
            evaluation_id="rules-1",
            run_id="run-1",
            trace_sha256=HEX_DIGEST,
            created_at="2026-08-27T00:00:00Z",
            evaluator=EvaluatorInfo("rules", "rules", "1.0.0"),
            rubric=RubricRef("script_quality", "1.0.0", HEX_DIGEST),
            status="completed",
            dimension_scores=(
                DimensionScore(
                    "topic_alignment",
                    "选题与要求契合度",
                    3,
                    "基本满足要求",
                    ("原文片段",),
                    (),
                ),
            ),
            findings=(Finding("notice", "info", "ok"),),
        )

        restored = evaluation_record_from_dict(record.to_dict())

        self.assertEqual(restored, record)
        self.assertFalse(restored.gate_failed)

    def test_failed_record_requires_error(self) -> None:
        with self.assertRaises(ValueError):
            EvaluationRecord(
                evaluation_id="judge-1",
                run_id="run-1",
                trace_sha256=HEX_DIGEST,
                created_at="2026-08-27T00:00:00Z",
                evaluator=EvaluatorInfo("judge", "judge", "1.0.0"),
                rubric=RubricRef("script_quality", "1.0.0", HEX_DIGEST),
                status="failed",
            )


if __name__ == "__main__":
    unittest.main()
