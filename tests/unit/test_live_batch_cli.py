"""Offline tests for the bounded live-research batch CLI."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from hyscript.agent import (
    Claim,
    ClaimUsage,
    Evidence,
    QueryPlan,
    ResearchGenerationError,
    ResearchOutcome,
    ScriptArtifact,
    ScriptGenerationError,
)
from hyscript.config import ResearchConfig, ScriptGenerationConfig
from hyscript.llm import LLMCallUsage


def load_module():
    path = Path(__file__).resolve().parents[2] / "scripts/run_live_batch.py"
    spec = importlib.util.spec_from_file_location("run_live_batch", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load live batch script module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def ready_research(topic: str) -> ResearchOutcome:
    return ResearchOutcome(
        status="ready",
        query_plan=QueryPlan(
            goal=f"核实{topic}",
            must_verify=("事实",),
            queries=(),
            current_date="2026-08-31",
        ),
        search_responses=(),
        evidence=(
            Evidence(
                evidence_id="E001",
                result_ref="R001",
                title="来源",
                url="https://example.com/source",
                excerpt="已经核实的事实。",
                source_query="查询",
            ),
        ),
        claims=(
            Claim(
                claim_id="C001",
                text="已经核实的事实。",
                evidence_ids=("E001",),
                is_core=True,
            ),
        ),
        errors=(),
        query_plan_prompt_version="query-test",
        evidence_prompt_version="evidence-test",
        llm_request_count=0,
        search_request_count=0,
    )


def generated_script(topic: str) -> ScriptArtifact:
    text = f"{topic}已有可核实的信息。"
    return ScriptArtifact(
        outline=("核心判断",),
        script_text=text,
        claim_usages=(ClaimUsage(claim_id="C001", script_quote=text),),
        character_count=len(text),
        prompt_version="script-test",
        generation_attempt_count=1,
    )


class FakeAsyncClient:
    def __init__(self, config) -> None:
        self.config = config

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None


class LiveBatchCliTests(unittest.TestCase):
    def test_concurrency_defaults_to_one_and_supports_formal_high_concurrency(
        self,
    ) -> None:
        module = load_module()
        parser = module.build_parser()
        required = [
            "--dataset",
            "dataset.json",
            "--task-spec",
            "tasks.json",
            "--output-dir",
            "output",
            "--experiment-id",
            "experiment",
            "--phase",
            "phase",
        ]

        defaults = parser.parse_args(required)
        self.assertEqual(defaults.concurrency, 1)
        self.assertFalse(hasattr(defaults, "require_grounding_review_accepted"))
        self.assertEqual(
            parser.parse_args([*required, "--task-concurrency", "512"]).task_concurrency,
            512,
        )
        self.assertEqual(
            parser.parse_args([*required, "--hy3-concurrency", "512"]).hy3_concurrency,
            512,
        )
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parser.parse_args([*required, "--require-grounding-review-accepted"])
        for invalid in ("0", "513"):
            with self.subTest(invalid=invalid), patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    parser.parse_args([*required, "--concurrency", invalid])
        for invalid in (False, 0, 513, 1.5, "2"):
            with self.subTest(runtime_invalid=invalid):
                with self.assertRaisesRegex(ValueError, "between 1 and 512"):
                    module._validate_task_concurrency(invalid)

    def test_loads_list_dataset_and_arbitrary_valid_lengths(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.json"
            task_spec_path = root / "tasks.json"
            write_json(dataset_path, ["选题一", "选题二"])
            write_json(
                task_spec_path,
                [
                    {"task_id": "short", "dataset_index": 0, "target_length": 50},
                    {"task_id": "custom", "dataset_index": 1, "target_length": 321},
                    {"task_id": "long", "dataset_index": 0, "target_length": 5000},
                ],
            )

            dataset = module._load_dataset(dataset_path)
            tasks = module._load_batch_tasks(task_spec_path, dataset)

        self.assertEqual(dataset, ("选题一", "选题二"))
        self.assertEqual(
            [item.task.target_length for item in tasks],
            [50, 321, 5000],
        )
        self.assertEqual([item.dataset_index for item in tasks], [0, 1, 0])
    def test_failed_script_attempts_remain_in_usage_accounting(self) -> None:
        module = load_module()
        research = ready_research("选题")
        error = ScriptGenerationError(
            "generation failed",
            generation_attempt_count=2,
            grounding_review_attempt_count=0,
        )

        usage = module._usage_payload(research, None, error)

        self.assertEqual(usage["hy3_attempted_calls"], 2)
        self.assertEqual(usage["script_generation_attempted_calls"], 2)
        self.assertEqual(usage["script_grounding_review_attempted_calls"], 0)

    def test_rejects_invalid_dataset_and_task_specs(self) -> None:
        module = load_module()
        cases = (
            ({"topic": "不是列表"}, [], "Dataset must be"),
            (["选题"], [], "non-empty"),
            (
                ["选题"],
                [{"task_id": "T01", "dataset_index": 0, "target_length": 49}],
                "between 50 and 5000",
            ),
            (
                ["选题"],
                [{"task_id": "T01", "dataset_index": 0, "target_length": 5001}],
                "between 50 and 5000",
            ),
            (
                ["选题"],
                [{"task_id": "../T01", "dataset_index": 0, "target_length": 450}],
                "task_id",
            ),
            (
                ["选题"],
                [
                    {"task_id": "T01", "dataset_index": 0, "target_length": 280},
                    {"task_id": "T01", "dataset_index": 0, "target_length": 700},
                ],
                "repeats task_id",
            ),
        )
        for dataset_payload, task_payload, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dataset_path = root / "dataset.json"
                task_spec_path = root / "tasks.json"
                write_json(dataset_path, dataset_payload)
                write_json(task_spec_path, task_payload)
                if not isinstance(dataset_payload, list):
                    with self.assertRaisesRegex(ValueError, expected):
                        module._load_dataset(dataset_path)
                    continue
                dataset = module._load_dataset(dataset_path)
                with self.assertRaisesRegex(ValueError, expected):
                    module._load_batch_tasks(task_spec_path, dataset)

    def test_refuses_a_nonempty_output_directory(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "output"
            output_dir.mkdir()
            (output_dir / "existing.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite or resume"):
                module._prepare_output_dir(output_dir)

    def test_runs_tasks_serially_with_fakes_and_no_external_calls(self) -> None:
        module = load_module()
        events: list[str] = []

        class FakeResearchAgent:
            def __init__(self, llm, search, *, config) -> None:
                pass

            async def research(self, task):
                events.append(f"research:{task.topic}")
                return ready_research(task.topic)

        class FakeScriptAgent:
            def __init__(self, llm, *, config) -> None:
                pass

            async def generate(self, task, research):
                events.append(f"script:{task.topic}")
                return generated_script(task.topic)

        fake_settings = SimpleNamespace(
            hy3=SimpleNamespace(model="fake-hy3", temperature=0.0, top_p=1.0),
            tavily=SimpleNamespace(),
            research=ResearchConfig(),
            script_generation=ScriptGenerationConfig(),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.json"
            task_spec_path = root / "tasks.json"
            output_dir = root / "output"
            write_json(dataset_path, ["选题甲", "选题乙"])
            write_json(
                task_spec_path,
                [
                    {"task_id": "T01", "dataset_index": 0, "target_length": 120},
                    {"task_id": "T02", "dataset_index": 1, "target_length": 1000},
                ],
            )
            args = argparse.Namespace(
                dataset=dataset_path,
                task_spec=task_spec_path,
                output_dir=output_dir,
                experiment_id="offline-test",
                phase="unit",
            )

            with (
                patch.object(module, "settings", fake_settings),
                patch.object(module, "AsyncHy3Client", FakeAsyncClient),
                patch.object(module, "AsyncTavilySearchProvider", FakeAsyncClient),
                patch.object(module, "ResearchAgent", FakeResearchAgent),
                patch.object(module, "ScriptAgent", FakeScriptAgent),
            ):
                result = asyncio.run(module._run(args))

            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            research_files = sorted((output_dir / "research").glob("*.json"))
            trace_files = sorted((output_dir / "traces").glob("*.json"))

        self.assertEqual(result, 0)
        self.assertEqual(
            events,
            [
                "research:选题甲",
                "script:选题甲",
                "research:选题乙",
                "script:选题乙",
            ],
        )
        self.assertEqual(manifest["counts"]["completed"], 2)
        self.assertEqual(
            manifest["execution"],
            {
                "task_concurrency": 1,
                "hy3_concurrency": 16,
                "search_concurrency": 8,
                "grounding_review_enabled": False,
            },
        )
        self.assertEqual(
            manifest["tasks"][0]["usage"]["hy3_attempted_calls"],
            1,
        )
        self.assertEqual(len(research_files), 2)
        self.assertEqual(len(trace_files), 2)

    def test_bounded_concurrency_preserves_task_order_and_manifest_checkpoints(
        self,
    ) -> None:
        module = load_module()
        events: list[str] = []
        active_tasks: set[str] = set()
        peak_active = 0
        first_started = asyncio.Event()
        release_first = asyncio.Event()

        class CoordinatedResearchAgent:
            def __init__(self, llm, search, *, config) -> None:
                pass

            async def research(self, task):
                nonlocal peak_active
                events.append(f"research:{task.topic}")
                active_tasks.add(task.topic)
                peak_active = max(peak_active, len(active_tasks))
                if task.topic == "选题甲":
                    first_started.set()
                    await release_first.wait()
                elif task.topic == "选题乙":
                    await first_started.wait()
                else:
                    if "script:选题乙" not in events:
                        raise AssertionError(
                            "Third task started before a slot was free."
                        )
                    release_first.set()
                return ready_research(task.topic)

        class CoordinatedScriptAgent:
            def __init__(self, llm, *, config) -> None:
                pass

            async def generate(self, task, research):
                events.append(f"script:{task.topic}")
                active_tasks.remove(task.topic)
                return generated_script(task.topic)

        fake_settings = SimpleNamespace(
            hy3=SimpleNamespace(model="fake-hy3", temperature=0.0, top_p=1.0),
            tavily=SimpleNamespace(),
            research=ResearchConfig(),
            script_generation=ScriptGenerationConfig(),
        )
        manifest_snapshots: list[dict] = []
        real_write_json = module._write_json

        def recording_write_json(path, payload, *, replace):
            if path.name == "manifest.json":
                manifest_snapshots.append(
                    json.loads(json.dumps(payload, ensure_ascii=False))
                )
            real_write_json(path, payload, replace=replace)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.json"
            task_spec_path = root / "tasks.json"
            output_dir = root / "output"
            write_json(dataset_path, ["选题甲", "选题乙", "选题丙"])
            write_json(
                task_spec_path,
                [
                    {"task_id": "T01", "dataset_index": 0, "target_length": 120},
                    {"task_id": "T02", "dataset_index": 1, "target_length": 450},
                    {"task_id": "T03", "dataset_index": 2, "target_length": 1000},
                ],
            )
            args = argparse.Namespace(
                dataset=dataset_path,
                task_spec=task_spec_path,
                output_dir=output_dir,
                experiment_id="offline-concurrency-test",
                phase="unit",
                concurrency=2,
                grounding_review=False,
            )

            async def run_with_timeout() -> int:
                return await asyncio.wait_for(module._run(args), timeout=5)

            with (
                patch.object(module, "settings", fake_settings),
                patch.object(module, "AsyncHy3Client", FakeAsyncClient),
                patch.object(module, "AsyncTavilySearchProvider", FakeAsyncClient),
                patch.object(module, "ResearchAgent", CoordinatedResearchAgent),
                patch.object(module, "ScriptAgent", CoordinatedScriptAgent),
                patch.object(module, "_write_json", recording_write_json),
            ):
                result = asyncio.run(run_with_timeout())

            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result, 0)
        self.assertEqual(peak_active, 2)
        self.assertEqual(active_tasks, set())
        self.assertLess(
            events.index("script:选题乙"),
            events.index("research:选题丙"),
        )
        self.assertEqual(
            [item["task_id"] for item in manifest["tasks"]],
            ["T01", "T02", "T03"],
        )
        self.assertEqual(
            manifest["execution"],
            {
                "task_concurrency": 2,
                "hy3_concurrency": 16,
                "search_concurrency": 8,
                "grounding_review_enabled": False,
            },
        )
        self.assertEqual(manifest["counts"]["completed"], 3)
        self.assertEqual(manifest["counts"]["pending"], 0)
        self.assertTrue(
            any(
                [item["status"] for item in snapshot["tasks"]]
                == ["running", "completed", "pending"]
                for snapshot in manifest_snapshots
            )
        )
        for snapshot in manifest_snapshots:
            self.assertEqual(
                [item["task_id"] for item in snapshot["tasks"]],
                ["T01", "T02", "T03"],
            )
            statuses = [item["status"] for item in snapshot["tasks"]]
            self.assertEqual(snapshot["counts"]["pending"], statuses.count("pending"))
            self.assertEqual(
                snapshot["counts"]["accounted"],
                len(statuses) - statuses.count("pending"),
            )

    def test_research_and_generation_share_the_global_hy3_semaphore(self) -> None:
        module = load_module()
        active = 0
        peak = 0
        release = asyncio.Event()

        async def bounded_call(owner, result):
            nonlocal active, peak
            async with owner._request_semaphore:
                active += 1
                peak = max(peak, active)
                if peak == 2:
                    release.set()
                await release.wait()
                await asyncio.sleep(0)
                active -= 1
                return result

        class BoundedResearchAgent:
            def __init__(self, llm, search, *, config) -> None:
                self._request_semaphore = None

            async def research(self, task):
                return await bounded_call(self, ready_research(task.topic))

        class BoundedScriptAgent:
            def __init__(self, llm, *, config) -> None:
                self._request_semaphore = None

            async def generate(self, task, research):
                return await bounded_call(self, generated_script(task.topic))

        fake_settings = SimpleNamespace(
            hy3=SimpleNamespace(model="fake-hy3", temperature=0.0, top_p=1.0),
            tavily=SimpleNamespace(),
            research=ResearchConfig(),
            script_generation=ScriptGenerationConfig(),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.json"
            task_spec_path = root / "tasks.json"
            output_dir = root / "output"
            write_json(dataset_path, ["选题甲", "选题乙", "选题丙"])
            write_json(
                task_spec_path,
                [
                    {
                        "task_id": f"T0{index}",
                        "dataset_index": index - 1,
                        "target_length": 450,
                    }
                    for index in range(1, 4)
                ],
            )
            args = argparse.Namespace(
                dataset=dataset_path,
                task_spec=task_spec_path,
                output_dir=output_dir,
                experiment_id="offline-shared-hy3-limit",
                phase="unit",
                task_concurrency=3,
                hy3_concurrency=2,
                search_concurrency=8,
            )
            with (
                patch.object(module, "settings", fake_settings),
                patch.object(module, "AsyncHy3Client", FakeAsyncClient),
                patch.object(module, "AsyncTavilySearchProvider", FakeAsyncClient),
                patch.object(module, "ResearchAgent", BoundedResearchAgent),
                patch.object(module, "ScriptAgent", BoundedScriptAgent),
            ):
                result = asyncio.run(module._run(args))

        self.assertEqual(result, 0)
        self.assertEqual(active, 0)
        self.assertEqual(peak, 2)

    def test_manifest_does_not_persist_unexpected_exception_details(self) -> None:
        module = load_module()

        class FailingResearchAgent:
            def __init__(self, llm, search, *, config) -> None:
                pass

            async def research(self, task):
                raise RuntimeError("api-key=do-not-store")

        class UnusedScriptAgent:
            def __init__(self, llm, *, config) -> None:
                pass

            async def generate(self, task, research):
                raise AssertionError("Script generation must not run.")

        fake_settings = SimpleNamespace(
            hy3=SimpleNamespace(model="fake-hy3", temperature=0.0, top_p=1.0),
            tavily=SimpleNamespace(),
            research=ResearchConfig(),
            script_generation=ScriptGenerationConfig(),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.json"
            task_spec_path = root / "tasks.json"
            output_dir = root / "output"
            write_json(dataset_path, ["选题"])
            write_json(
                task_spec_path,
                [{"task_id": "T01", "dataset_index": 0, "target_length": 450}],
            )
            args = argparse.Namespace(
                dataset=dataset_path,
                task_spec=task_spec_path,
                output_dir=output_dir,
                experiment_id="offline-test",
                phase="safe-error",
            )

            with (
                patch.object(module, "settings", fake_settings),
                patch.object(module, "AsyncHy3Client", FakeAsyncClient),
                patch.object(module, "AsyncTavilySearchProvider", FakeAsyncClient),
                patch.object(module, "ResearchAgent", FailingResearchAgent),
                patch.object(module, "ScriptAgent", UnusedScriptAgent),
            ):
                result = asyncio.run(module._run(args))

            manifest_text = (output_dir / "manifest.json").read_text(encoding="utf-8")
            manifest = json.loads(manifest_text)

        self.assertEqual(result, 1)
        self.assertNotIn("do-not-store", manifest_text)
        self.assertEqual(manifest["tasks"][0]["error_type"], "RuntimeError")
        self.assertEqual(
            manifest["tasks"][0]["error"],
            "Unexpected task failure. See local console logs for the exception type.",
        )

    def test_manifest_records_usage_when_research_generation_fails(self) -> None:
        module = load_module()
        reported_usage = LLMCallUsage(
            stage="research.evidence_selection",
            attempt=1,
            model="fake-hy3",
            request_id="request-1",
            input_tokens=120,
            output_tokens=30,
            total_tokens=150,
            reasoning_tokens=8,
            cached_input_tokens=20,
            raw_usage={},
        )

        class FailingResearchAgent:
            def __init__(self, llm, search, *, config) -> None:
                pass

            async def research(self, task):
                raise ResearchGenerationError(
                    "Research evidence selection failed after one retry.",
                    llm_request_count=3,
                    search_request_count=3,
                    successful_search_count=2,
                    llm_usages=(reported_usage,),
                )

        class UnusedScriptAgent:
            def __init__(self, llm, *, config) -> None:
                pass

            async def generate(self, task, research):
                raise AssertionError("Script generation must not run.")

        fake_settings = SimpleNamespace(
            hy3=SimpleNamespace(model="fake-hy3", temperature=0.0, top_p=1.0),
            tavily=SimpleNamespace(),
            research=ResearchConfig(),
            script_generation=ScriptGenerationConfig(),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.json"
            task_spec_path = root / "tasks.json"
            output_dir = root / "output"
            write_json(dataset_path, ["选题"])
            write_json(
                task_spec_path,
                [{"task_id": "T01", "dataset_index": 0, "target_length": 450}],
            )
            args = argparse.Namespace(
                dataset=dataset_path,
                task_spec=task_spec_path,
                output_dir=output_dir,
                experiment_id="offline-test",
                phase="research-failure-usage",
            )

            with (
                patch.object(module, "settings", fake_settings),
                patch.object(module, "AsyncHy3Client", FakeAsyncClient),
                patch.object(module, "AsyncTavilySearchProvider", FakeAsyncClient),
                patch.object(module, "ResearchAgent", FailingResearchAgent),
                patch.object(module, "ScriptAgent", UnusedScriptAgent),
            ):
                result = asyncio.run(module._run(args))

            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )

        usage = manifest["tasks"][0]["usage"]
        self.assertEqual(result, 1)
        self.assertEqual(manifest["tasks"][0]["status"], "failed")
        self.assertEqual(
            manifest["tasks"][0]["error_type"],
            "ResearchGenerationError",
        )
        self.assertEqual(usage["hy3_attempted_calls"], 3)
        self.assertEqual(usage["research_hy3_attempted_calls"], 3)
        self.assertEqual(usage["hy3_reported_calls"], 1)
        self.assertEqual(usage["hy3_total_tokens"], 150)
        self.assertEqual(usage["tavily_attempted_calls"], 3)
        self.assertEqual(usage["tavily_succeeded_calls"], 2)

    def test_live_batch_does_not_run_or_gate_on_grounding_review(self) -> None:
        module = load_module()
        seen_configs = []

        class FakeResearchAgent:
            def __init__(self, llm, search, *, config) -> None:
                pass

            async def research(self, task):
                return ready_research(task.topic)

        class RejectedReviewScriptAgent:
            def __init__(self, llm, *, config) -> None:
                seen_configs.append(config)

            async def generate(self, task, research):
                return replace(
                    generated_script(task.topic),
                    grounding_review_attempt_count=1,
                    grounding_review_status="rejected",
                    grounding_review_prompt_version="review-test",
                    grounding_review_issues=(
                        "insufficient_evidence: 缺少关键证据。",
                    ),
                    grounding_review_failure_reason="review_rejected",
                )

        fake_settings = SimpleNamespace(
            hy3=SimpleNamespace(model="fake-hy3", temperature=0.0, top_p=1.0),
            tavily=SimpleNamespace(),
            research=ResearchConfig(),
            script_generation=ScriptGenerationConfig(),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.json"
            task_spec_path = root / "tasks.json"
            output_dir = root / "output"
            write_json(dataset_path, ["选题"])
            write_json(
                task_spec_path,
                [{"task_id": "T01", "dataset_index": 0, "target_length": 321}],
            )
            args = argparse.Namespace(
                dataset=dataset_path,
                task_spec=task_spec_path,
                output_dir=output_dir,
                experiment_id="formal-review-gate",
                phase="unit",
                concurrency=1,
                grounding_review=False,
                require_grounding_review_accepted=True,
            )
            with (
                patch.object(module, "settings", fake_settings),
                patch.object(module, "AsyncHy3Client", FakeAsyncClient),
                patch.object(module, "AsyncTavilySearchProvider", FakeAsyncClient),
                patch.object(module, "ResearchAgent", FakeResearchAgent),
                patch.object(module, "ScriptAgent", RejectedReviewScriptAgent),
            ):
                result = asyncio.run(module._run(args))

            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            trace_files = list((output_dir / "traces").glob("*.json"))

        self.assertEqual(result, 0)
        self.assertFalse(seen_configs[0].grounding_review_enabled)
        self.assertEqual(manifest["counts"]["completed"], 1)
        self.assertEqual(manifest["counts"]["failed"], 0)
        self.assertEqual(len(trace_files), 1)


if __name__ == "__main__":
    unittest.main()
