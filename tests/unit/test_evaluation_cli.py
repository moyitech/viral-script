"""Tests for offline evaluation command input discovery."""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hyscript.config import PROJECT_ROOT
from scripts.run_evaluation import _trace_paths, build_parser


class EvaluationCliTests(unittest.TestCase):
    def test_default_rubric_is_original_v1(self) -> None:
        args = build_parser().parse_args(["score", "--trace", "trace.json"])

        self.assertEqual(
            args.rubric,
            PROJECT_ROOT / "eval/rubrics/script_quality_v1.json",
        )
        self.assertEqual(args.reasoning_effort, "high")

    def test_trace_directory_skips_results_configs_and_output_tree(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.json"
            trace.write_text(
                json.dumps({"run_id": "run-1", "script_artifact": {}}),
                encoding="utf-8",
            )
            (root / "manifest.json").write_text(
                json.dumps({"evaluation_id": "eval-1", "status": "completed"}),
                encoding="utf-8",
            )
            (root / "rubric.json").write_text(
                json.dumps({"rubric_id": "quality", "dimensions": []}),
                encoding="utf-8",
            )
            result = root / "rules.json"
            result.write_text(
                json.dumps(
                    {
                        "run_id": "run-1",
                        "evaluation_id": "rules-1",
                        "evaluator": {"kind": "rules"},
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "output"
            output_dir.mkdir()
            (output_dir / "trace-like.json").write_text(
                json.dumps({"run_id": "old-output"}),
                encoding="utf-8",
            )

            paths = _trace_paths(
                Namespace(trace=None, trace_dir=root),
                output_dir=output_dir,
            )

            self.assertEqual(paths, [trace])

    def test_repeated_trace_directories_combine_primary_and_retry_inputs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary"
            retry = root / "retry"
            output = root / "output"
            primary.mkdir()
            retry.mkdir()
            output.mkdir()
            first = primary / "T01.json"
            second = retry / "T02.json"
            first.write_text(json.dumps({"run_id": "run-1"}), encoding="utf-8")
            second.write_text(json.dumps({"run_id": "run-2"}), encoding="utf-8")

            args = build_parser().parse_args(
                [
                    "score",
                    "--trace-dir",
                    str(primary),
                    "--trace-dir",
                    str(retry),
                ]
            )
            paths = _trace_paths(args, output_dir=output)

            self.assertEqual(paths, [first, second])


if __name__ == "__main__":
    unittest.main()
