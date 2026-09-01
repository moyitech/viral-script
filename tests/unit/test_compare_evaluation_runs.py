"""Tests for the read-only fixed-evaluation comparison report."""

from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.compare_evaluation_runs import DIMENSION_NAMES, run


_HEX = "a" * 64


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _trace_set(root: Path, *, editorial: bool) -> Path:
    run_id = "run-editor" if editorial else "run-single"
    task_id = "T01-L280"
    prompt_version = (
        "script-generation-background-editorial-2.1.0"
        if editorial
        else "script-generation-background-1.0.0"
    )
    prompt_versions = {"script_generation": prompt_version}
    experiment = {"script_system_prompt_sha256": _HEX}
    artifact: dict[str, object] = {
        "script_text": "这是一篇可评分的测试正文。",
        "reference_ids": ["E001"],
        "prompt_version": prompt_version,
        "llm_usages": [
            {
                "stage": "script.editor" if editorial else "script.generation",
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "reasoning_tokens": 5,
                "cached_input_tokens": 1,
            }
        ],
    }
    request_counts = {
        "script_generation_llm": 4 if editorial else 1,
        "script_candidate_llm": 3 if editorial else 0,
        "script_editor_llm": 1 if editorial else 0,
        "script_grounding_review_llm": 0,
    }
    if editorial:
        artifact.update(
            {
                "generation_mode": "editorial_candidates",
                "generation_candidates": [
                    {
                        "candidate_id": f"C0{index}",
                        "strategy": f"strategy-{index}",
                        "prompt_version": "script-generation-background-candidate-2.0.0",
                        "reference_ids": ["E001"],
                    }
                    for index in range(1, 4)
                ],
                "selected_candidate_ids": ["C01"],
                "editor_prompt_version": "script-generation-background-editor-2.1.0",
                "length_repair_attempted": True,
            }
        )
        prompt_versions.update(
            {
                "script_candidate": "script-generation-background-candidate-2.0.0",
                "script_editor": "script-generation-background-editor-2.1.0",
            }
        )
        experiment.update(
            {
                "script_candidate_prompt_sha256": "b" * 64,
                "script_editor_prompt_sha256": "c" * 64,
            }
        )
    trace = {
        "run_id": run_id,
        "script_artifact": artifact,
        "lineage": {
            "prompt_versions": prompt_versions,
            "script_reference_ids": ["E001"],
            "evidence_to_result_ref": {"E001": "R001"},
        },
        "config": {
            "request_counts": request_counts,
            "experiment": experiment,
        },
    }
    trace_path = root / "traces" / f"{task_id}-{run_id}.json"
    _write_json(trace_path, trace)
    manifest_path = root / "manifest.json"
    _write_json(
        manifest_path,
        {
            "tasks": [
                {
                    "task_id": task_id,
                    "status": "completed",
                    "run_id": run_id,
                    "trace": str(trace_path),
                }
            ]
        },
    )
    return manifest_path


def _evaluation(
    root: Path,
    trace_manifest: Path,
    *,
    rhetoric_score: int,
) -> None:
    generation_manifest = json.loads(trace_manifest.read_text(encoding="utf-8"))
    task = generation_manifest["tasks"][0]
    trace_path = Path(task["trace"])
    trace_sha256 = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    run_id = task["run_id"]
    scores = {dimension_id: 3 for dimension_id in DIMENSION_NAMES}
    scores["rhetoric_memorability"] = rhetoric_score
    _write_json(
        root / "manifest.json",
        {
            "rubric": {"version": "1.1.0"},
            "fingerprint": {
                "evaluators": [
                    {
                        "kind": "judge",
                        "version": "3.3.0",
                        "prompt_version": "script-quality-grounded-groups-v3.3",
                    },
                    {"kind": "rules", "version": "1.5.0"},
                ],
                "aggregator": {"version": "1.2.0"},
                "sha256": "f" * 64,
            },
            "inputs": [
                {
                    "source": f"T01-L280-{run_id}.json",
                    "run_id": run_id,
                    "trace_sha256": trace_sha256,
                }
            ],
        },
    )
    dimensions = [
        {
            "dimension_id": dimension_id,
            "score": score,
            "reason": f"{dimension_id} reason",
        }
        for dimension_id, score in scores.items()
    ]
    _write_json(
        root / "items" / run_id / "combined.json",
        {
            "status": "completed",
            "run_id": run_id,
            "trace_sha256": trace_sha256,
            "dimension_scores": dimensions,
            "metrics": {"weighted_total": 24.0},
            "gate_failed": False,
        },
    )
    _write_json(
        root / "items" / run_id / "hy3_judge.json",
        {
            "metadata": {
                "span_evidence": {
                    "engagement": {"problem_spans": []},
                    "oral_fluency": {"problem_spans": []},
                }
            }
        },
    )


class CompareEvaluationRunsTests(unittest.TestCase):
    def test_report_audits_traces_calls_and_repeat_variance(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_trace = _trace_set(root / "baseline-traces", editorial=False)
            candidate_trace = _trace_set(root / "candidate-traces", editorial=True)
            baseline_dirs = [root / "baseline-r1", root / "baseline-r2"]
            candidate_dirs = [root / "candidate-r1", root / "candidate-r2"]
            _evaluation(baseline_dirs[0], baseline_trace, rhetoric_score=2)
            _evaluation(baseline_dirs[1], baseline_trace, rhetoric_score=2)
            _evaluation(candidate_dirs[0], candidate_trace, rhetoric_score=2)
            _evaluation(candidate_dirs[1], candidate_trace, rhetoric_score=3)
            output_json = root / "report.json"
            output_md = root / "report.md"

            status = run(
                Namespace(
                    baseline_dir=baseline_dirs,
                    candidate_dir=candidate_dirs,
                    baseline_trace_manifest=[baseline_trace],
                    candidate_trace_manifest=[candidate_trace],
                    output_json=output_json,
                    output_md=output_md,
                    label="中文测试报告",
                )
            )

            self.assertEqual(status, 0)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertTrue(payload["passed"])
            self.assertEqual(
                payload["trace_audit"]["candidate"]["attempted_calls"],
                {
                    "script_generation_llm": 4,
                    "script_candidate_llm": 3,
                    "script_editor_llm": 1,
                    "script_grounding_review_llm": 0,
                },
            )
            self.assertEqual(
                payload["repeat_variance"]["candidate"]
                ["rhetoric_memorability"]["changed_item_count"],
                1,
            )
            markdown = output_md.read_text(encoding="utf-8")
            self.assertIn("生成 Trace 与调用量审计", markdown)
            self.assertIn("两次评分方差", markdown)
            self.assertIn("供后续 AI 阅读", markdown)


if __name__ == "__main__":
    unittest.main()
