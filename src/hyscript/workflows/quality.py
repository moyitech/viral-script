"""On-demand creator quality reports backed by the formal evaluators."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import logging
from pathlib import Path
from typing import Any, Mapping

from hyscript.config import Settings
from hyscript.evaluation import (
    BatchEvaluationConfig,
    BatchEvaluationRunner,
    EvaluationRecord,
    Hy3JudgeEvaluator,
    JudgeConfig,
    load_frozen_trace,
    load_rubric,
)
from hyscript.evaluation.io import load_json_object
from hyscript.evaluation.models import evaluation_record_from_dict
from hyscript.llm import AsyncHy3Client


logger = logging.getLogger(__name__)


class QualityReportError(RuntimeError):
    """Stable application error for a failed on-demand evaluation."""


@dataclass(frozen=True, slots=True)
class QualityDimensionReport:
    """One user-readable rubric dimension and its exact script evidence."""

    dimension_id: str
    name: str
    score: int | None
    score_max: int
    reason: str
    positive_spans: tuple[str, ...] = ()
    problem_spans: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QualityReport:
    """UI-ready view of rules, Judge diagnostics, and their combined score."""

    evaluation_id: str
    run_id: str
    summary: str
    score_percent: float | None
    eligible: bool
    gate_failed: bool
    rubric_version: str
    judge_model: str | None
    dimensions: tuple[QualityDimensionReport, ...]
    findings: tuple[dict[str, Any], ...]
    judge_groups: tuple[dict[str, Any], ...]
    oral_subscores: dict[str, dict[str, Any]]
    cached: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe bridge payload without provider request metadata."""

        return asdict(self)


def _record(path: Path) -> EvaluationRecord:
    return evaluation_record_from_dict(load_json_object(path))


def _safe_spans(
    span_evidence: Mapping[str, Any],
    dimension_id: str,
    name: str,
) -> tuple[str, ...]:
    item = span_evidence.get(dimension_id, {})
    if not isinstance(item, Mapping):
        return ()
    value = item.get(name, [])
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(text for text in value if isinstance(text, str) and text)


def _oral_subscores(judge: EvaluationRecord) -> dict[str, dict[str, Any]]:
    diagnostics = judge.metadata.get("judge_diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        return {}
    for group in diagnostics.values():
        if not isinstance(group, Mapping):
            continue
        raw_subscores = group.get("oral_subscores")
        if not isinstance(raw_subscores, Mapping):
            continue
        normalized: dict[str, dict[str, Any]] = {}
        for name, raw in raw_subscores.items():
            if not isinstance(name, str) or not isinstance(raw, Mapping):
                continue
            normalized[name] = {
                "score": raw.get("score"),
                "comment": raw.get("comment", ""),
                "positive_spans": list(raw.get("positive_spans", [])),
                "problem_spans": list(raw.get("problem_spans", [])),
            }
        return normalized
    return {}


def _quality_report(
    *,
    combined: EvaluationRecord,
    judge: EvaluationRecord,
    score_max: int,
    cached: bool,
) -> QualityReport:
    span_evidence = judge.metadata.get("span_evidence", {})
    if not isinstance(span_evidence, Mapping):
        span_evidence = {}
    dimensions = tuple(
        QualityDimensionReport(
            dimension_id=score.dimension_id,
            name=score.name,
            score=score.score,
            score_max=score_max,
            reason=score.reason,
            positive_spans=_safe_spans(
                span_evidence,
                score.dimension_id,
                "positive_spans",
            ),
            problem_spans=_safe_spans(
                span_evidence,
                score.dimension_id,
                "problem_spans",
            ),
        )
        for score in combined.dimension_scores
    )
    raw_groups = judge.metadata.get("judge_groups", [])
    judge_groups = tuple(
        {
            "name": group.get("name", ""),
            "dimension_ids": list(group.get("dimension_ids", [])),
            "summary": group.get("summary", ""),
        }
        for group in raw_groups
        if isinstance(group, Mapping)
    )
    final_score = combined.metrics.get("final_score")
    score_percent = (
        round(float(final_score) * 100, 1)
        if isinstance(final_score, (int, float)) and not isinstance(final_score, bool)
        else None
    )
    eligible = combined.metrics.get("eligible") is True
    return QualityReport(
        evaluation_id=combined.evaluation_id,
        run_id=combined.run_id,
        summary=judge.summary or combined.summary,
        score_percent=score_percent,
        eligible=eligible,
        gate_failed=combined.gate_failed,
        rubric_version=combined.rubric.version,
        judge_model=judge.evaluator.model,
        dimensions=dimensions,
        findings=tuple(asdict(finding) for finding in combined.findings),
        judge_groups=judge_groups,
        oral_subscores=_oral_subscores(judge),
        cached=cached,
    )


class CreatorEvaluationWorkflow:
    """Run the formal rubric only after a generation trace has been frozen."""

    def __init__(
        self,
        settings: Settings,
        *,
        rubric_path: Path | None = None,
        judge_evaluator: Hy3JudgeEvaluator | None = None,
    ) -> None:
        self.settings = settings
        self.rubric_path = rubric_path or (
            settings.project_root / "eval/rubrics/script_quality_v1.json"
        )
        self._judge_evaluator = judge_evaluator

    async def score_trace(self, trace_path: Path) -> QualityReport:
        """Score one immutable trace and return a cached-or-fresh quality report."""

        trace = load_frozen_trace(trace_path)
        rubric = load_rubric(self.rubric_path)
        if self._judge_evaluator is not None:
            return await self._run(
                trace_path,
                trace.run_id,
                rubric,
                self._judge_evaluator,
            )

        judge_settings = replace(
            self.settings.hy3,
            temperature=0.0,
            top_p=1.0,
        )
        async with AsyncHy3Client(judge_settings) as client:
            judge = Hy3JudgeEvaluator(
                client,
                model_name=judge_settings.model,
                config=JudgeConfig(reasoning_effort="high"),
                sampling_parameters={
                    "temperature": judge_settings.temperature,
                    "top_p": judge_settings.top_p,
                },
            )
            return await self._run(trace_path, trace.run_id, rubric, judge)

    async def _run(
        self,
        trace_path: Path,
        run_id: str,
        rubric: Any,
        judge: Hy3JudgeEvaluator,
    ) -> QualityReport:
        logger.info("正在准备正式文案评分")
        probe = BatchEvaluationRunner(
            rubric,
            BatchEvaluationConfig(
                output_dir=self.settings.runtime.evaluation_dir,
                evaluators=("rules", "judge"),
                concurrency=2,
            ),
            judge_evaluator=judge,
        )
        output_dir = (
            self.settings.runtime.evaluation_dir
            / run_id
            / probe.fingerprint.sha256
        )
        runner = BatchEvaluationRunner(
            rubric,
            replace(probe.config, output_dir=output_dir),
            judge_evaluator=judge,
        )
        logger.info("正在运行长度规则与 Hy3 七维 Judge")
        result = await runner.run((trace_path,))
        if len(result.outcomes) != 1 or result.outcomes[0].status == "failed":
            outcome = result.outcomes[0] if result.outcomes else None
            message = outcome.message if outcome and outcome.message else "评分未能完成"
            raise QualityReportError(message)

        item_dir = output_dir / "items" / run_id
        combined = _record(item_dir / "combined.json")
        judge_record = _record(item_dir / "hy3_judge.json")
        cached = result.outcomes[0].status == "skipped"
        logger.info("评分完成%s", "（已复用缓存）" if cached else "")
        return _quality_report(
            combined=combined,
            judge=judge_record,
            score_max=rubric.score_max,
            cached=cached,
        )
