"""Offline tests for the creator-facing formal quality report."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from hyscript.config import PROJECT_ROOT, load_settings
from hyscript.evaluation import JudgeConfig
from hyscript.evaluation.judge import JUDGE_EVALUATOR_NAME, JUDGE_EVALUATOR_VERSION
from hyscript.evaluation.models import (
    DimensionScore,
    EvaluationRecord,
    EvaluatorInfo,
    RubricRef,
    new_evaluation_id,
    utc_now_iso,
)
from hyscript.workflows import CreatorEvaluationWorkflow


class _FakeJudge:
    model_name = "fake-hy3"
    config = JudgeConfig(reasoning_effort="high")
    sampling_parameters = {"temperature": 0.0, "top_p": 1.0}

    def __init__(self) -> None:
        self.calls = 0

    async def evaluate(self, trace, rubric, *, request_semaphore=None):
        self.calls += 1
        excerpt = trace.script_text[:10]
        scores = tuple(
            DimensionScore(
                dimension_id=dimension.dimension_id,
                name=dimension.name,
                score=2,
                reason=f"{dimension.name}表现成立，但仍有可改进空间。",
                script_spans=(excerpt,),
            )
            for dimension in rubric.judge_dimensions
        )
        return EvaluationRecord(
            evaluation_id=new_evaluation_id("judge"),
            run_id=trace.run_id,
            trace_sha256=trace.trace_sha256,
            created_at=utc_now_iso(),
            evaluator=EvaluatorInfo(
                kind="judge",
                name=JUDGE_EVALUATOR_NAME,
                version=JUDGE_EVALUATOR_VERSION,
                model=self.model_name,
            ),
            rubric=RubricRef(
                rubric_id=rubric.rubric_id,
                version=rubric.version,
                sha256=rubric.sha256,
            ),
            status="completed",
            summary="整体可用，但表达仍可继续打磨。",
            dimension_scores=scores,
            metrics={
                "weighted_average": 2.0,
                "normalized_score": 2 / 3,
            },
            metadata={
                "span_evidence": {
                    dimension.dimension_id: {
                        "positive_spans": [excerpt],
                        "problem_spans": [],
                    }
                    for dimension in rubric.judge_dimensions
                },
                "judge_groups": [
                    {
                        "name": "content",
                        "dimension_ids": ["topic_alignment"],
                        "summary": "内容主线基本成立。",
                        "request_ids": ["must-not-reach-ui"],
                    }
                ],
                "judge_diagnostics": {
                    "口播": {
                        "oral_subscores": {
                            "朗读顺口度": {
                                "score": 2,
                                "comment": "整体可读，但局部气口仍然偏长。",
                                "positive_spans": [excerpt],
                                "problem_spans": [],
                            },
                            "口语自然度": {
                                "score": 2,
                                "comment": "有对话感，但中段仍略偏书面表达。",
                                "positive_spans": [excerpt],
                                "problem_spans": [],
                            },
                        }
                    }
                },
            },
        )


def _settings(directory: Path):
    loaded = load_settings(
        env_file=None,
        environ={
            "HY3_BASE_URL": "https://hy3.example/v1",
            "HY3_API_KEY": "test-key",
            "EMBEDDING_BASE_URL": "https://embedding.example/v1",
            "EMBEDDING_API_KEY": "test-embedding-key",
            "TAVILY_API_KEY": "test-search-key",
        },
    )
    return replace(
        loaded,
        runtime=replace(
            loaded.runtime,
            runs_dir=directory / "traces",
            evaluation_dir=directory / "evaluations",
        ),
    )


class CreatorQualityWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_report_is_complete_trace_safe_and_cached(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            trace_path = directory / "trace.json"
            shutil.copyfile(PROJECT_ROOT / "eval/traces/example_trace.json", trace_path)
            original_bytes = trace_path.read_bytes()
            judge = _FakeJudge()
            workflow = CreatorEvaluationWorkflow(
                _settings(directory),
                judge_evaluator=judge,
            )

            first = await workflow.score_trace(trace_path)
            second = await workflow.score_trace(trace_path)

            self.assertEqual(judge.calls, 1)
            self.assertFalse(first.cached)
            self.assertTrue(second.cached)
            self.assertEqual(len(first.dimensions), 8)
            self.assertIn("朗读顺口度", first.oral_subscores)
            self.assertEqual(first.judge_groups[0]["summary"], "内容主线基本成立。")
            self.assertNotIn("request_ids", first.judge_groups[0])
            self.assertEqual(trace_path.read_bytes(), original_bytes)
            self.assertTrue(
                any(
                    path.name == "combined.json"
                    for path in _settings(directory).runtime.evaluation_dir.rglob("*.json")
                )
            )
