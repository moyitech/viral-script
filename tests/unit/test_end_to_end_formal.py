"""Offline tests for the frozen-research single-shot formal experiment."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hyscript.evaluation import end_to_end_formal as e2e


def _build_baseline_fixture(root: Path) -> Path:
    baseline = root / "formal-100-v1"
    dataset = baseline / "dataset.json"
    rubric = baseline / "rubric.json"
    topics = [f"选题{index}" for index in range(1, 101)]
    e2e.write_json(dataset, topics)
    e2e.write_json(rubric, {"schema_version": "test"})

    matrix: list[dict] = []
    catalog: list[dict] = []
    for index, topic in enumerate(topics):
        topic_id = f"T{index + 1:03d}"
        catalog.append(
            {
                "topic_id": topic_id,
                "dataset_index": index,
                "topic": topic,
                "domain": "test",
                "challenge_tags": ["test"],
            }
        )
        for length in e2e.TARGET_LENGTHS:
            matrix.append(
                {
                    "task_id": f"{topic_id}-L{length}",
                    "source_task_id": topic_id,
                    "dataset_index": index,
                    "target_length": length,
                    "topic": topic,
                    "domain": "test",
                    "challenge_tags": ["test"],
                }
            )

    versions = e2e._prompt_versions()
    hashes = e2e._prompt_hashes()
    e2e.write_json(
        baseline / "experiment.json",
        {
            "dataset": "dataset.json",
            "dataset_sha256": e2e.sha256_file(dataset),
            "rubric": "rubric.json",
            "rubric_sha256": e2e.sha256_file(rubric),
            "prompt_versions": {
                "research_query_plan": versions["research_query_plan"],
            },
        },
    )
    e2e.write_json(baseline / "topics.json", catalog)
    e2e.write_json(baseline / "task_matrix.json", matrix)
    e2e.lock_runtime(baseline)
    e2e.write_json(
        baseline / "generation/research/attempt-001/manifest.json",
        {
            "system_prompt_sha256": {
                "research_query_plan": hashes["research_query_plan"],
            }
        },
    )

    research_tasks: list[dict] = []
    for index, topic in enumerate(topics, start=1):
        task_id = f"T{index:03d}"
        snapshot = (
            baseline
            / "generation/research/attempt-001/research"
            / f"{task_id}.json"
        )
        e2e.write_json(snapshot, {"task_id": task_id, "status": "ready"})
        research_tasks.append(
            {
                "task_id": task_id,
                "topic": topic,
                "target_length": 450,
                "research_status": "ready",
                "research_snapshot": (
                    f"research/attempt-001/research/{task_id}.json"
                ),
                "research_sha256": e2e.sha256_file(snapshot),
            }
        )
    e2e.write_json(
        baseline / "generation/research_manifest.json",
        {"selected_count": 100, "tasks": research_tasks},
    )

    trace = baseline / "generation/traces/fixture.json"
    e2e.write_json(
        trace,
        {
            "schema_version": "1.0",
            "run_id": "baseline-fixture",
            "task": {"topic": topics[0], "target_length": 280},
            "queries": ["测试查询"],
            "search_results": [{"title": "测试结果"}],
            "selected_evidence": [
                {
                    "evidence_id": "evidence-001",
                    "url": "https://example.com/source",
                    "title": "测试来源",
                    "snippet": "测试证据",
                }
            ],
            "claims": [
                {
                    "claim_id": "claim-001",
                    "text": "测试主张",
                    "is_core": True,
                    "evidence_ids": ["evidence-001"],
                    "support_status": "supported",
                }
            ],
            "script_artifact": {
                "script_text": "测试正文",
                "character_count": 302,
            },
            "config": {
                "request_counts": {
                    "script_llm": 4,
                    "research_llm": 3,
                    "script_candidate_llm": 3,
                    "tavily_attempted": 3,
                    "tavily_succeeded": 3,
                }
            },
            "lineage": {
                "prompt_versions": {
                    "research_query_plan": versions["research_query_plan"],
                    "research_evidence": versions["background_selection"],
                },
                "llm_calls": [
                    {"stage": "research.query_plan", "total_tokens": 30}
                ],
            },
            "latency": {"search_response_time_sum": 6.0},
            "token_usage": {},
        },
    )
    trace_tasks = [
        {"task_id": item["task_id"], "trace": "traces/fixture.json"}
        for item in matrix
    ]
    e2e.write_json(
        baseline / "generation/trace_manifest.json",
        {"tasks": trace_tasks},
    )

    score_rows = [
        {
            "task_id": item["task_id"],
            "run_id": f"baseline-{item['task_id']}",
            "topic": item["topic"],
            "target_length": item["target_length"],
            "domain": item["domain"],
            "challenge_tags": "test",
            "final_score": 1.0,
            "gate_failed": False,
            "hy3_total_tokens": 90,
            **{dimension: 3 for dimension in e2e._DIMENSIONS},
        }
        for item in matrix
    ]
    e2e.atomic_write_text(
        baseline / "results/full_results.csv",
        e2e._csv_text(score_rows),
    )
    return baseline


class EndToEndFormalTests(unittest.TestCase):
    def test_prepare_clones_the_300_task_matrix_and_locks_512(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            baseline = _build_baseline_fixture(temporary_root)
            root = temporary_root / "formal-100-e2e-test"
            config = e2e.prepare_e2e_experiment(root, baseline_dir=baseline)
            specs = json.loads((root / "task_specs.json").read_text(encoding="utf-8"))

        self.assertEqual(config["expected_trace_count"], 300)
        self.assertEqual(
            config["concurrency"],
            {
                "tasks": 300,
                "hy3": 512,
                "source_search": 8,
                "new_search": 0,
                "judge": 512,
            },
        )
        self.assertEqual(len(specs), 300)
        self.assertEqual(specs[0]["target_length"], 280)
        self.assertEqual(specs[0]["source_research_target_length"], 450)
        self.assertTrue(specs[0]["source_research_snapshot_sha256"])
        self.assertEqual(specs[-1]["task_id"], "T100-L700")

    def test_generate_writes_only_missing_tasks_with_512_hy3_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            baseline = _build_baseline_fixture(temporary_root)
            root = temporary_root / "formal-100-e2e-test"
            e2e.prepare_e2e_experiment(root, baseline_dir=baseline)
            with patch.object(e2e, "_run_command", return_value=1) as run:
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    e2e.generate_e2e_experiment(root, baseline_dir=baseline)

            specs = json.loads(
                (root / "generation/specs/attempt-001.json").read_text(
                    encoding="utf-8"
                )
            )
            command = run.call_args.args[0]

        self.assertEqual(len(specs), 300)
        self.assertEqual(command[command.index("--request-concurrency") + 1], "512")
        self.assertEqual(command[command.index("--concurrency") + 1], "300")
        self.assertEqual(
            command[command.index("--generation-mode") + 1],
            "single_shot",
        )
        self.assertNotIn("--search-concurrency", command)
        self.assertIn("replay_script_generation.py", command[1])

    def test_selects_a_valid_single_shot_trace_and_records_source_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            baseline = _build_baseline_fixture(temporary_root)
            root = temporary_root / "formal-100-e2e-test"
            config = e2e.prepare_e2e_experiment(root, baseline_dir=baseline)
            baseline_manifest = json.loads(
                (baseline / "generation/trace_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            source = (
                baseline
                / "generation"
                / baseline_manifest["tasks"][0]["trace"]
            )
            payload = json.loads(source.read_text(encoding="utf-8"))
            task_spec = json.loads(
                (root / "task_specs.json").read_text(encoding="utf-8")
            )[0]
            payload["config"]["experiment"] = {
                "experiment_id": root.name,
                "phase": "end-to-end-attempt-001",
                "mode": "frozen_research_single_shot_replay",
                "task_id": "T001-L280",
                "source_manifest_sha256": config[
                    "source_research_manifest_sha256"
                ],
                "source_research_snapshot_sha256": task_spec[
                    "source_research_snapshot_sha256"
                ],
                "script_system_prompt_sha256": config["prompt_sha256"][
                    "script_generation"
                ],
                "script_format_repair_prompt_sha256": config["prompt_sha256"][
                    "script_format_repair"
                ],
            }
            artifact = payload["script_artifact"]
            artifact.update(
                {
                    "generation_mode": "single_shot",
                    "generation_attempt_count": 1,
                    "content_generation_attempt_count": 1,
                    "format_repair_attempt_count": 0,
                    "generation_candidates": [],
                    "selected_candidate_ids": [],
                    "editor_attempt_count": 0,
                    "length_repair_attempted": False,
                    "grounding_review_attempt_count": 0,
                    "final_rewrite_attempt_count": 0,
                    "format_repair_content_preserved": True,
                    "initial_script_text_sha256": e2e._text_sha256(
                        artifact["script_text"]
                    ),
                }
            )
            payload["lineage"]["script_generation_mode"] = "single_shot"
            payload["lineage"]["prompt_versions"].update(
                {
                    "script_generation": config["prompt_versions"][
                        "script_generation"
                    ],
                    "script_format_repair": None,
                }
            )
            first_call = payload["lineage"]["llm_calls"][0]
            payload["lineage"]["llm_calls"] = [
                {**first_call, "stage": "script.generation", "attempt": 1}
            ]
            payload["config"]["request_counts"].update(
                {
                    "research_llm": 0,
                    "search": 0,
                    "script_llm": 1,
                    "script_generation_llm": 1,
                    "script_content_generation_llm": 1,
                    "script_format_repair_llm": 0,
                    "script_grounding_review_llm": 0,
                    "script_final_rewrite_llm": 0,
                    "hy3_total": 1,
                    "tavily_attempted": 0,
                    "tavily_succeeded": 0,
                    "tavily_failed": 0,
                }
            )
            attempt = root / "generation/attempt-001"
            trace = attempt / "traces/T001-L280.json"
            trace.parent.mkdir(parents=True)
            trace.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            (attempt / "manifest.json").write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "task_id": "T001-L280",
                                "status": "completed",
                                "trace": str(trace),
                                "script_usage": {},
                                "content_generation_attempt_count": 1,
                                "format_repair_attempt_count": 0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            selected = e2e.select_e2e_traces(root)

        self.assertEqual(selected["selected_count"], 1)
        self.assertEqual(selected["tasks"][0]["target_length"], 280)
        self.assertEqual(
            selected["tasks"][0]["source_research_target_length"],
            450,
        )
        self.assertEqual(selected["new_tavily_request_count"], 0)

    def test_paired_comparison_requires_and_preserves_all_300_ids(self) -> None:
        baseline: list[dict] = []
        candidate: list[dict] = []
        for topic in range(1, 101):
            for length in e2e.TARGET_LENGTHS:
                task_id = f"T{topic:03d}-L{length}"
                dimensions = {dimension: 3 for dimension in e2e._DIMENSIONS}
                baseline.append(
                    {
                        "task_id": task_id,
                        "run_id": f"baseline-{task_id}",
                        "final_score": 1.0,
                        "gate_failed": False,
                        **dimensions,
                    }
                )
                candidate.append(
                    {
                        "task_id": task_id,
                        "run_id": f"candidate-{task_id}",
                        "topic": f"选题{topic}",
                        "target_length": length,
                        "domain": "test",
                        "challenge_tags": "test",
                        "final_score": 1.0,
                        "gate_failed": False,
                        "character_count": length,
                        "absolute_length_error_ratio": 0.0,
                        "hy3_total_tokens": 100,
                        "hy3_attempted_calls": 1,
                        "content_generation_attempted_calls": 1,
                        "format_repair_attempted_calls": 0,
                        "search_attempted_calls": 3,
                        **dimensions,
                    }
                )

        rows = e2e._paired_rows(baseline, candidate)
        summary = e2e._paired_summary(rows)

        self.assertEqual(len(rows), 300)
        self.assertEqual(summary["ties"], 300)
        self.assertEqual(summary["wins"], 0)
        self.assertEqual(summary["losses"], 0)
        with self.assertRaisesRegex(ValueError, "300 exact pairs"):
            e2e._paired_rows(baseline[:-1], candidate)

    def test_baseline_resource_allocation_is_additive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline = _build_baseline_fixture(Path(directory))
            rows = e2e._baseline_rows(baseline)
            resources = e2e._resource_summary(rows)

        self.assertEqual(len(rows), 300)
        self.assertAlmostEqual(resources["hy3_total_tokens"], 30_000)
        self.assertAlmostEqual(resources["search_attempted_calls"], 300)
        self.assertAlmostEqual(resources["search_latency_seconds"], 600)
        self.assertEqual(rows[0]["character_count"], 302)

    def test_relative_path_falls_back_to_absolute_across_windows_drives(self) -> None:
        path = Path("artifact.json").resolve()
        with patch.object(
            e2e.os.path,
            "relpath",
            side_effect=ValueError("path is on a different drive"),
        ):
            value = e2e._relative(path, Path("experiment"))

        self.assertEqual(value, str(path))

    def test_evaluation_resume_summary_counts_only_last_resume_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "items/first").mkdir(parents=True)
            (root / "items/retried").mkdir(parents=True)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "started_at": "2026-09-03T01:00:00Z",
                        "last_started_at": "2026-09-03T02:00:00Z",
                    }
                ),
                encoding="utf-8",
            )
            (root / "items/first/hy3_judge.json").write_text(
                json.dumps({"created_at": "2026-09-03T01:30:00Z"}),
                encoding="utf-8",
            )
            (root / "items/retried/hy3_judge.json").write_text(
                json.dumps({"created_at": "2026-09-03T02:10:00Z"}),
                encoding="utf-8",
            )

            summary = e2e._evaluation_resume_summary(root, 2)

        self.assertTrue(summary["resume_detected"])
        self.assertEqual(summary["completed_before_last_resume"], 1)
        self.assertEqual(summary["remaining_before_last_resume"], 1)
        self.assertEqual(summary["completed_in_last_resume"], 1)


if __name__ == "__main__":
    unittest.main()
