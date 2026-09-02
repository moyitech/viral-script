"""Tests for repeat-Judge internal consistency statistics."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from hyscript.evaluation.stability import compare_judge_runs, export_judge_stability


def _record(run_id: str, scores: tuple[int, int], normalized: float) -> dict:
    return {
        "run_id": run_id,
        "trace_sha256": ("a" if run_id == "run-1" else "b") * 64,
        "status": "completed",
        "dimension_scores": [
            {"dimension_id": "engagement", "score": scores[0]},
            {"dimension_id": "topic_alignment", "score": scores[1]},
        ],
        "metrics": {"normalized_score": normalized},
        "metadata": {
            "format_attempts": 4,
            "evaluator_fingerprint": {"sha256": "f" * 64},
        },
    }


class JudgeStabilityTests(unittest.TestCase):
    def test_compare_and_export_repeat_judge_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            repeat = root / "repeat"
            for target, evaluation_id, records in (
                (
                    baseline,
                    "first",
                    (_record("run-1", (1, 3), 2 / 3), _record("run-2", (3, 3), 1.0)),
                ),
                (
                    repeat,
                    "second",
                    (_record("run-1", (2, 3), 5 / 6), _record("run-2", (3, 3), 1.0)),
                ),
            ):
                (target / "items").mkdir(parents=True)
                (target / "summary.json").write_text(
                    json.dumps({"evaluation_id": evaluation_id}), encoding="utf-8"
                )
                for record in records:
                    item = target / "items" / record["run_id"]
                    item.mkdir()
                    (item / "hy3_judge.json").write_text(
                        json.dumps(record), encoding="utf-8"
                    )
            manifest = root / "trace_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {"run_id": "run-1", "task_id": "T001-L280", "topic": "A"},
                            {"run_id": "run-2", "task_id": "T002-L280", "topic": "B"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            summary, rows = compare_judge_runs(
                baseline, repeat, trace_manifest=manifest
            )
            self.assertEqual(summary["record_count"], 2)
            self.assertEqual(summary["overall"]["dimension_exact_agreement_rate"], 0.75)
            self.assertEqual(summary["overall"]["all_dimensions_exact_rate"], 0.5)
            self.assertEqual(summary["dimensions"]["topic_alignment"]["exact_agreement_rate"], 1.0)
            self.assertEqual(rows[0]["changed_dimensions"], "engagement")

            output = root / "output"
            export_judge_stability(
                baseline, repeat, output, trace_manifest=manifest
            )
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "full_comparison.csv").is_file())
            self.assertIn("逐维完全一致率", (output / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
