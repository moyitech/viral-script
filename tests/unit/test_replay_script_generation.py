"""Offline tests for the frozen-research script replay CLI."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

from hyscript.llm import LLMCallUsage


def load_module():
    path = Path(__file__).resolve().parents[2] / "scripts/replay_script_generation.py"
    spec = importlib.util.spec_from_file_location("replay_script_generation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load replay script module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReplayScriptGenerationTests(unittest.TestCase):
    def test_formal_review_gate_flag_is_explicit(self) -> None:
        module = load_module()
        required = [
            "--source-manifest",
            "source.json",
            "--output-dir",
            "output",
            "--phase",
            "phase",
        ]
        parser = module.build_parser()

        self.assertFalse(
            parser.parse_args(required).require_grounding_review_accepted
        )
        self.assertTrue(
            parser.parse_args(
                [*required, "--require-grounding-review-accepted"]
            ).require_grounding_review_accepted
        )

    def test_selects_replayable_research_and_supports_explicit_ids(self) -> None:
        module = load_module()
        manifest = {
            "tasks": [
                {
                    "task_id": "T01",
                    "status": "completed",
                    "research_snapshot": "/tmp/T01.json",
                    "target_length": 321,
                },
                {
                    "task_id": "T02",
                    "status": "insufficient_evidence",
                    "research_snapshot": "/tmp/T02.json",
                },
                {
                    "task_id": "T03",
                    "status": "completed",
                    "research_snapshot": "/tmp/T03.json",
                },
                {
                    "task_id": "T04",
                    "status": "failed",
                    "research_status": "ready",
                    "research_snapshot": "/tmp/T04.json",
                },
                {
                    "task_id": "T05",
                    "status": "failed",
                    "research_status": None,
                    "research_snapshot": "/tmp/T05.json",
                },
            ]
        }

        replayable = module._source_tasks(manifest, ())
        self.assertEqual(
            [item["task_id"] for item in replayable],
            ["T01", "T03", "T04"],
        )
        self.assertEqual(replayable[0]["target_length"], 321)
        self.assertEqual(
            [item["task_id"] for item in module._source_tasks(manifest, ("T03",))],
            ["T03"],
        )
        self.assertEqual(
            [item["task_id"] for item in module._source_tasks(manifest, ("T04",))],
            ["T04"],
        )
        with self.assertRaisesRegex(ValueError, "Unknown task ids"):
            module._source_tasks(manifest, ("T99",))

    def test_reports_only_incremental_script_usage(self) -> None:
        module = load_module()
        script = SimpleNamespace(
            generation_attempt_count=1,
            grounding_review_attempt_count=1,
            llm_usages=(
                LLMCallUsage(
                    stage="script.generation",
                    attempt=1,
                    model="hy3",
                    request_id="request-1",
                    input_tokens=100,
                    output_tokens=50,
                    total_tokens=150,
                    reasoning_tokens=20,
                    cached_input_tokens=10,
                    raw_usage={},
                ),
            ),
        )

        self.assertEqual(
            module._script_usage(script),
            {
                "attempted_calls": 2,
                "generation_attempted_calls": 1,
                "grounding_review_attempted_calls": 1,
                "reported_calls": 1,
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "reasoning_tokens": 20,
                "cached_input_tokens": 10,
            },
        )

    def test_expands_each_topic_across_three_test_lengths(self) -> None:
        module = load_module()
        tasks = [
            {"task_id": "T01", "topic": "选题甲", "target_length": 321},
            {"task_id": "T02", "topic": "选题乙", "target_length": 654},
        ]

        expanded = module._expand_target_lengths(tasks, (280, 450, 700))

        self.assertEqual(len(expanded), 6)
        self.assertEqual(
            [item["task_id"] for item in expanded],
            [
                "T01-L280",
                "T01-L450",
                "T01-L700",
                "T02-L280",
                "T02-L450",
                "T02-L700",
            ],
        )
        self.assertEqual(
            [item["target_length"] for item in expanded[:3]],
            [280, 450, 700],
        )
        self.assertTrue(all(item["source_task_id"] in {"T01", "T02"} for item in expanded))

        with self.assertRaisesRegex(ValueError, "must not repeat"):
            module._expand_target_lengths(tasks, (280, 280))


if __name__ == "__main__":
    unittest.main()
