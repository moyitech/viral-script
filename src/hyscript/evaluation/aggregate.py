"""Deterministic combination and batch summaries for evaluator records."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from .io import FrozenTrace
from .models import (
    EvaluationRecord,
    EvaluatorInfo,
    RubricRef,
    new_evaluation_id,
    utc_now_iso,
)
from .rubric import Rubric

AGGREGATOR_VERSION = "1.2.0"
AGGREGATOR_NAME = "gate-aware-combiner"


def combine_evaluations(
    trace: FrozenTrace,
    rubric: Rubric,
    records: Iterable[EvaluationRecord],
) -> EvaluationRecord:
    """Combine independent records without changing their original outputs."""

    sources = tuple(records)
    if not sources:
        raise ValueError("At least one evaluation record is required.")
    source_kinds = [record.evaluator.kind for record in sources]
    if "aggregate" in source_kinds:
        raise ValueError("Aggregate records cannot be used as aggregation sources.")
    duplicate_kinds = sorted(
        kind for kind, count in Counter(source_kinds).items() if count > 1
    )
    if duplicate_kinds:
        raise ValueError(
            f"Only one source record is allowed per evaluator kind: {duplicate_kinds}."
        )
    for record in sources:
        if record.run_id != trace.run_id:
            raise ValueError("Cannot combine records from different run ids.")
        if record.trace_sha256 != trace.trace_sha256:
            raise ValueError("Cannot combine records from different trace hashes.")
        if record.rubric.sha256 != rubric.sha256:
            raise ValueError("Cannot combine records from different rubric versions.")
        if record.status != "completed":
            raise ValueError("Cannot combine failed evaluation records.")

    judge_record = next(
        (record for record in sources if record.evaluator.kind == "judge"), None
    )
    dimension_scores = tuple(
        score for record in sources for score in record.dimension_scores
    )
    score_ids = [score.dimension_id for score in dimension_scores]
    duplicate_score_ids = sorted(
        dimension_id
        for dimension_id, count in Counter(score_ids).items()
        if count > 1
    )
    if duplicate_score_ids:
        raise ValueError(
            f"Dimension scores must not repeat across evaluators: {duplicate_score_ids}."
        )
    unknown_score_ids = sorted(set(score_ids) - set(rubric.dimension_ids))
    if unknown_score_ids:
        raise ValueError(f"Unknown dimension scores: {unknown_score_ids}.")
    scores_by_id = {score.dimension_id: score for score in dimension_scores}
    dimension_scores = tuple(
        scores_by_id[dimension_id]
        for dimension_id in rubric.dimension_ids
        if dimension_id in scores_by_id
    )
    score_ids = [score.dimension_id for score in dimension_scores]
    findings = tuple(finding for record in sources for finding in record.findings)
    gate_counts = Counter(
        finding.code for finding in findings if finding.severity == "gate"
    )
    judge_weighted_average = (
        judge_record.metrics.get("weighted_average") if judge_record else None
    )
    judge_normalized_score = (
        judge_record.metrics.get("normalized_score") if judge_record else None
    )
    dimensions_by_id = {
        dimension.dimension_id: dimension for dimension in rubric.dimensions
    }
    weighted_total = sum(
        score.score * dimensions_by_id[score.dimension_id].weight
        for score in dimension_scores
        if score.score is not None
    )
    evaluated_weight = sum(
        dimensions_by_id[score.dimension_id].weight
        for score in dimension_scores
        if score.score is not None
    )
    partial_weighted_average = (
        weighted_total / evaluated_weight if evaluated_weight else None
    )
    all_dimensions_present = set(score_ids) == set(rubric.dimension_ids)
    all_dimensions_evaluable = all(
        score.score is not None for score in dimension_scores
    )
    judge_comparable = judge_record is not None and judge_normalized_score is not None
    weighted_average = (
        partial_weighted_average
        if all_dimensions_present and all_dimensions_evaluable and judge_comparable
        else None
    )
    normalized_score = (
        weighted_average / rubric.score_max
        if weighted_average is not None
        else None
    )
    eligible = not gate_counts and normalized_score is not None
    metrics: dict[str, Any] = {
        "gate_count": sum(gate_counts.values()),
        "gate_codes": dict(sorted(gate_counts.items())),
        "eligible": eligible,
        "final_score": normalized_score if eligible else None,
        "weighted_total": weighted_total,
        "partial_weighted_average": partial_weighted_average,
        "weighted_average": weighted_average,
        "normalized_score": normalized_score,
        "evaluable_dimension_count": sum(
            score.score is not None for score in dimension_scores
        ),
        "score_coverage": (
            sum(score.score is not None for score in dimension_scores)
            / len(rubric.dimensions)
        ),
        "judge_weighted_average": judge_weighted_average,
        "judge_normalized_score": judge_normalized_score,
    }
    return EvaluationRecord(
        evaluation_id=new_evaluation_id("combined"),
        run_id=trace.run_id,
        trace_sha256=trace.trace_sha256,
        created_at=utc_now_iso(),
        evaluator=EvaluatorInfo(
            kind="aggregate",
            name=AGGREGATOR_NAME,
            version=AGGREGATOR_VERSION,
        ),
        rubric=RubricRef(
            rubric_id=rubric.rubric_id,
            version=rubric.version,
            sha256=rubric.sha256,
        ),
        status="completed",
        summary=(
            "Evaluation completed with non-compensable gate findings."
            if gate_counts
            else "Evaluation completed without gate findings."
        ),
        dimension_scores=dimension_scores,
        metrics=metrics,
        findings=findings,
        metadata={
            "source_evaluations": [
                {
                    "evaluation_id": record.evaluation_id,
                    "kind": record.evaluator.kind,
                    "name": record.evaluator.name,
                    "version": record.evaluator.version,
                }
                for record in sources
            ]
        },
    )


def summarize_batch(records: Iterable[EvaluationRecord]) -> dict[str, Any]:
    """Aggregate completed combined records into report-friendly statistics."""

    items = tuple(records)
    dimension_values: dict[str, list[int]] = defaultdict(list)
    gate_counts: Counter[str] = Counter()
    final_scores: list[float] = []
    for record in items:
        if record.status != "completed":
            continue
        for score in record.dimension_scores:
            if score.score is not None:
                dimension_values[score.dimension_id].append(score.score)
        gate_counts.update(
            finding.code for finding in record.findings if finding.severity == "gate"
        )
        final_score = record.metrics.get("final_score")
        if isinstance(final_score, (int, float)) and not isinstance(final_score, bool):
            final_scores.append(float(final_score))
    dimension_summary = {
        dimension_id: {
            "count": len(values),
            "mean": sum(values) / len(values),
            "distribution": {
                str(score): values.count(score)
                for score in range(min(values), max(values) + 1)
            },
        }
        for dimension_id, values in sorted(dimension_values.items())
    }
    return {
        "record_count": len(items),
        "gate_failed_count": sum(record.gate_failed for record in items),
        "eligible_count": len(final_scores),
        "final_score_mean": (
            sum(final_scores) / len(final_scores) if final_scores else None
        ),
        "gate_counts": dict(sorted(gate_counts.items())),
        "dimensions": dimension_summary,
    }


__all__ = [
    "AGGREGATOR_NAME",
    "AGGREGATOR_VERSION",
    "combine_evaluations",
    "summarize_batch",
]
