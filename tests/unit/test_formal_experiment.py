"""Offline tests for the versioned formal experiment infrastructure."""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hyscript.agent import PlannedQuery, ResearchAgent
from hyscript.config import PROJECT_ROOT, ResearchConfig
from hyscript.evaluation.formal import (
    _relative,
    build_discrimination_traces,
    export_report,
    prepare_experiment,
    select_research,
    select_traces,
)
from hyscript.evaluation.human import (
    DIMENSIONS,
    import_human_annotations,
    quadratic_weighted_kappa,
    spearman,
)
from hyscript.llm import ChatMessage, ChatResponse
from hyscript.search import SearchResponse


class ConcurrencyLLM:
    def __init__(self) -> None:
        self.active = 0
        self.maximum = 0

    async def complete(self, messages, *, reasoning_effort="no_think") -> ChatResponse:
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return ChatResponse(content='{"ok": true}')


class ConcurrencySearch:
    def __init__(self) -> None:
        self.active = 0
        self.maximum = 0

    async def search(self, query: str, *, limit: int = 20) -> SearchResponse:
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return SearchResponse(provider="fake", query=query, results=())


class FormalExperimentTests(unittest.TestCase):
    def test_relative_path_falls_back_to_absolute_across_windows_drives(self) -> None:
        path = Path("dataset.json").resolve()
        with patch(
            "hyscript.evaluation.formal.os.path.relpath",
            side_effect=ValueError("path is on a different drive"),
        ):
            value = _relative(path, Path("experiment"))

        self.assertEqual(value, str(path))

    def test_prepare_builds_stable_100_by_three_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = prepare_experiment(
                root,
                dataset_path=PROJECT_ROOT / "eval/datasets/eval_topics_synthetic_v1.json",
                rubric_path=PROJECT_ROOT / "eval/rubrics/script_quality_v1.json",
            )
            topics = json.loads((root / "topics.json").read_text(encoding="utf-8"))
            research = json.loads((root / "research_tasks.json").read_text(encoding="utf-8"))
            matrix = json.loads((root / "task_matrix.json").read_text(encoding="utf-8"))

            self.assertEqual(config["expected_trace_count"], 300)
            self.assertEqual(len(topics), 100)
            self.assertEqual(len(research), 100)
            self.assertEqual(len(matrix), 300)
            self.assertEqual([item["task_id"] for item in matrix[:3]], ["T001-L280", "T001-L450", "T001-L700"])
            self.assertTrue(all(item["domain"] for item in topics))
            self.assertTrue(all(item["challenge_tags"] for item in topics))

    def test_attempt_selection_uses_latest_success_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_experiment(
                root,
                dataset_path=PROJECT_ROOT / "eval/datasets/eval_topics_synthetic_v1.json",
                rubric_path=PROJECT_ROOT / "eval/rubrics/script_quality_v1.json",
            )
            attempt = root / "generation/research/attempt-001"
            snapshot = attempt / "research/T001.json"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_text('{"status":"ready"}', encoding="utf-8")
            (attempt / "manifest.json").write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "task_id": "T001",
                                "topic": "存款利率继续走低，提前还贷还是留现金？",
                                "research_status": "ready",
                                "status": "completed",
                                "research_snapshot": str(snapshot),
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            selected_research = select_research(root)
            self.assertEqual(selected_research["selected_count"], 1)
            self.assertEqual(
                selected_research["tasks"][0]["research_snapshot"],
                "research/attempt-001/research/T001.json",
            )

            trace_dir = root / "generation/traces/attempt-001/traces"
            trace_dir.mkdir(parents=True)
            trace = trace_dir / "T001-L280-run-1.json"
            trace_payload = json.loads(
                (PROJECT_ROOT / "eval/traces/example_trace.json").read_text(encoding="utf-8")
            )
            trace_payload["run_id"] = "run-1"
            trace_payload["task"]["topic"] = "存款利率继续走低，提前还贷还是留现金？"
            trace_payload["task"]["target_length"] = 280
            trace_payload["config"] = {"experiment": {"task_id": "T001-L280"}}
            trace.write_text(
                json.dumps(trace_payload, ensure_ascii=False),
                encoding="utf-8",
            )
            selected_traces = select_traces(root)
            self.assertEqual(selected_traces["selected_count"], 1)
            self.assertEqual(
                selected_traces["tasks"][0]["trace"],
                "traces/attempt-001/traces/T001-L280-run-1.json",
            )

    def test_research_hy3_and_search_limits_are_global(self) -> None:
        async def scenario() -> tuple[int, int]:
            llm = ConcurrencyLLM()
            search = ConcurrencySearch()
            hy3_limit = asyncio.Semaphore(4)
            search_limit = asyncio.Semaphore(2)
            agents = [
                ResearchAgent(
                    llm,
                    search,
                    config=ResearchConfig(max_search_concurrency=3),
                    request_semaphore=hy3_limit,
                    search_semaphore=search_limit,
                )
                for _ in range(3)
            ]
            await asyncio.gather(
                *(
                    agent._request_with_retry(
                        (ChatMessage(role="user", content="test"),),
                        lambda value: value,
                        stage="test",
                        usage_stage="test",
                    )
                    for agent in agents for _ in range(4)
                )
            )
            queries = tuple(PlannedQuery(query=f"q{i}", purpose="test") for i in range(3))
            await asyncio.gather(*(agent._search_queries(queries) for agent in agents))
            return llm.maximum, search.maximum

        llm_maximum, search_maximum = asyncio.run(scenario())
        self.assertEqual(llm_maximum, 4)
        self.assertEqual(search_maximum, 2)

    def test_agreement_statistics_handle_perfect_and_inverse_orders(self) -> None:
        self.assertEqual(quadratic_weighted_kappa([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertEqual(spearman([1, 2, 3], [3, 2, 1]), -1.0)

    def test_discrimination_builder_creates_twenty_blinded_quartets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_experiment(
                root,
                dataset_path=PROJECT_ROOT / "eval/datasets/eval_topics_synthetic_v1.json",
                rubric_path=PROJECT_ROOT / "eval/rubrics/script_quality_v1.json",
            )
            matrix = json.loads((root / "task_matrix.json").read_text(encoding="utf-8"))
            tasks = []
            for item in matrix:
                path = root / "generation/traces/attempt-001/traces" / f"{item['task_id']}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": "1.0",
                            "run_id": f"run-{item['task_id']}",
                            "script_artifact": {"script_text": "一段有事实、有解释并正面回答问题的完整口播文案。", "character_count": 24},
                            "config": {"experiment": {"task_id": item["task_id"]}},
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                tasks.append(
                    {
                        **item,
                        "status": "completed",
                        "run_id": f"run-{item['task_id']}",
                        "trace": f"traces/attempt-001/traces/{item['task_id']}.json",
                        "trace_sha256": "0" * 64,
                    }
                )
                result_dir = root / "results/items" / f"run-{item['task_id']}"
                result_dir.mkdir(parents=True, exist_ok=True)
                (result_dir / "combined.json").write_text(
                    json.dumps(
                        {
                            "run_id": f"run-{item['task_id']}",
                            "gate_failed": False,
                            "metrics": {"final_score": 0.9},
                        }
                    ),
                    encoding="utf-8",
                )
            (root / "generation/trace_manifest.json").write_text(
                json.dumps({"tasks": tasks}, ensure_ascii=False), encoding="utf-8"
            )

            manifest = build_discrimination_traces(root)
            key = json.loads(
                (root / "validation/discrimination/answer_key.json").read_text(encoding="utf-8")
            )

            self.assertEqual(manifest["case_count"], 80)
            self.assertEqual(len(key), 80)
            self.assertEqual(sum(item["expected_tier"] == "attack" for item in key), 20)
            self.assertNotIn("expected_tier", manifest["tasks"][0])

    def test_report_does_not_overwrite_evaluation_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_experiment(
                root,
                dataset_path=PROJECT_ROOT / "eval/datasets/eval_topics_synthetic_v1.json",
                rubric_path=PROJECT_ROOT / "eval/rubrics/script_quality_v1.json",
            )
            snapshot = root / "generation/research/attempt-001/research/T001.json"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_text('{"search_responses": []}', encoding="utf-8")
            (root / "generation/research_manifest.json").write_text(
                json.dumps(
                    {
                        "selected_count": 1,
                        "tasks": [{"task_id": "T001", "research_snapshot": "research/attempt-001/research/T001.json", "usage": {}}],
                        "attempts": {},
                    }
                ),
                encoding="utf-8",
            )
            matrix = json.loads((root / "task_matrix.json").read_text(encoding="utf-8"))[:3]
            tasks = []
            for item in matrix:
                run_id = f"run-{item['task_id']}"
                trace = root / "generation/traces/attempt-001/traces" / f"{item['task_id']}.json"
                trace.parent.mkdir(parents=True, exist_ok=True)
                trace.write_text(
                    json.dumps(
                        {
                            "script_artifact": {"script_text": "测试正文"},
                            "token_usage": {"hy3_total_tokens": 10},
                            "latency": {},
                            "config": {"request_counts": {}},
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                tasks.append(
                    {
                        **item,
                        "status": "completed",
                        "run_id": run_id,
                        "trace": f"traces/attempt-001/traces/{item['task_id']}.json",
                        "trace_sha256": "0" * 64,
                    }
                )
                item_dir = root / "results/items" / run_id
                item_dir.mkdir(parents=True)
                (item_dir / "combined.json").write_text(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "status": "completed",
                            "gate_failed": False,
                            "metrics": {"final_score": 0.8},
                            "dimension_scores": [],
                        }
                    ),
                    encoding="utf-8",
                )
            (root / "generation/trace_manifest.json").write_text(
                json.dumps({"tasks": tasks, "attempts": {}}, ensure_ascii=False), encoding="utf-8"
            )
            evaluation_summary = root / "results/summary.json"
            evaluation_summary.write_text('{"owner":"evaluation-runner"}', encoding="utf-8")

            summary = export_report(root)

            self.assertEqual(summary["scored_records"], 3)
            self.assertEqual(evaluation_summary.read_text(encoding="utf-8"), '{"owner":"evaluation-runner"}')
            self.assertEqual(
                len((root / "results/full_results.csv").read_text(encoding="utf-8").splitlines()),
                4,
            )

    def test_human_import_writes_consensus_and_agreement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_experiment(
                root,
                dataset_path=PROJECT_ROOT / "eval/datasets/eval_topics_synthetic_v1.json",
                rubric_path=PROJECT_ROOT / "eval/rubrics/script_quality_v1.json",
            )
            tasks = [
                {
                    "run_id": f"run-{index:03d}",
                    "trace_sha256": f"{index:064x}",
                }
                for index in range(1, 51)
            ]
            generation = root / "generation"
            generation.mkdir(exist_ok=True)
            (generation / "trace_manifest.json").write_text(
                json.dumps({"tasks": tasks}), encoding="utf-8"
            )
            header = [
                "blind_id", "blind_batch", "run_id", "trace_sha256", "reviewer_id",
                *DIMENSIONS, "gate_failed", "notes",
            ]
            reviewer_paths = []
            for reviewer_id in ("reviewer-a", "reviewer-b"):
                path = root / f"{reviewer_id}.csv"
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=header)
                    writer.writeheader()
                    for index, task in enumerate(tasks, start=1):
                        writer.writerow(
                            {
                                "blind_id": f"H{index:03d}",
                                "blind_batch": "formal-human-v1",
                                "run_id": task["run_id"],
                                "trace_sha256": task["trace_sha256"],
                                "reviewer_id": reviewer_id,
                                **{dimension: 2 for dimension in DIMENSIONS},
                                "gate_failed": "false",
                                "notes": "independent blind review",
                            }
                        )
                reviewer_paths.append(path)

            result = import_human_annotations(root, reviewer_files=reviewer_paths)

            self.assertEqual(result["reviewed_count"], 50)
            self.assertEqual(result["consensus_count"], 50)
            self.assertEqual(result["pending_arbitration"], [])
            self.assertEqual(
                result["dimensions"]["topic_alignment"]["quadratic_weighted_kappa"],
                1.0,
            )
            self.assertTrue(
                (root / "validation/human/results/items/run-001/human.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
