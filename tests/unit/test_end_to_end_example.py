"""Offline tests for user-facing end-to-end usage reporting."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

from hyscript.llm import LLMCallUsage


def load_example_module():
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "examples/04_end_to_end.py"
    spec = importlib.util.spec_from_file_location("hyscript_end_to_end_example", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load end-to-end example module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def usage(stage: str, *, input_tokens: int, output_tokens: int) -> LLMCallUsage:
    return LLMCallUsage(
        stage=stage,
        attempt=1,
        model="hy3",
        request_id=None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        reasoning_tokens=5,
        cached_input_tokens=10,
        raw_usage={},
    )


class EndToEndUsageTests(unittest.TestCase):
    def test_reports_tavily_calls_and_all_hy3_token_categories(self) -> None:
        module = load_example_module()
        research = SimpleNamespace(
            search_request_count=5,
            search_responses=(object(), object(), object(), object()),
            llm_request_count=3,
            llm_usages=(
                usage("research.query_plan", input_tokens=100, output_tokens=20),
                usage(
                    "research.evidence_selection",
                    input_tokens=200,
                    output_tokens=40,
                ),
            ),
        )
        script = SimpleNamespace(
            generation_attempt_count=2,
            llm_usages=(
                usage("script.generation", input_tokens=300, output_tokens=60),
            ),
        )

        result = module.usage_statistics(research, script)

        self.assertEqual(
            result["tavily"],
            {"attempted_calls": 5, "succeeded_calls": 4, "failed_calls": 1},
        )
        self.assertEqual(result["hy3"]["attempted_calls"], 5)
        self.assertEqual(result["hy3"]["reported_usage_calls"], 3)
        self.assertEqual(result["hy3"]["input_tokens"], 600)
        self.assertEqual(result["hy3"]["output_tokens"], 120)
        self.assertEqual(result["hy3"]["total_tokens"], 720)
        self.assertEqual(result["hy3"]["reasoning_tokens"], 15)
        self.assertEqual(result["hy3"]["cached_input_tokens"], 30)


if __name__ == "__main__":
    unittest.main()
