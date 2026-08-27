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

AGGREGATOR_VERSION = "1.1.0"
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
    dimension_scores = judge_record.dimension_scores if judge_record else ()
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
    eligible = not gate_counts and judge_normalized_score is not None
    metrics: dict[str, Any] = {
        "gate_count": sum(gate_counts.values()),
        "gate_codes": dict(sorted(gate_counts.items())),
        "eligible": eligible,
        "final_score": judge_normalized_score if eligible else None,
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
            "distribution": {str(score): values.count(score) for score in range(5)},
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
