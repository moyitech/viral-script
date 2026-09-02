"""Strict import and agreement statistics for blinded human annotations."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

from .formal import load_json, write_json
from .io import write_evaluation_record
from .models import (
    DimensionScore,
    EvaluationRecord,
    EvaluatorInfo,
    Finding,
    RubricRef,
    utc_now_iso,
)
from .rubric import load_rubric

DIMENSIONS = (
    "topic_alignment", "length_compliance", "theme_information", "engagement",
    "oral_fluency", "rhetoric_memorability", "logic_structure", "safety_compliance",
)


@dataclass(frozen=True, slots=True)
class HumanAnnotation:
    run_id: str
    trace_sha256: str
    reviewer_id: str
    blind_batch: str
    scores: dict[str, int]
    gate_failed: bool
    notes: str


def _boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "是"}:
        return True
    if normalized in {"0", "false", "no", "否"}:
        return False
    raise ValueError("gate_failed must be true/false, 1/0, yes/no, or 是/否.")


def load_annotations(path: Path) -> list[HumanAnnotation]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Could not read human annotation CSV: {path}") from exc
    if not rows:
        raise ValueError(f"Human annotation CSV is empty: {path}")
    annotations: list[HumanAnnotation] = []
    for index, row in enumerate(rows, start=2):
        try:
            scores = {dimension: int(row[dimension]) for dimension in DIMENSIONS}
            if any(not 1 <= score <= 3 for score in scores.values()):
                raise ValueError
            annotation = HumanAnnotation(
                run_id=row["run_id"].strip(),
                trace_sha256=row["trace_sha256"].strip(),
                reviewer_id=row["reviewer_id"].strip(),
                blind_batch=row["blind_batch"].strip(),
                scores=scores,
                gate_failed=_boolean(row["gate_failed"]),
                notes=row.get("notes", "").strip(),
            )
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Invalid human annotation at {path}:{index}") from None
        if not annotation.run_id or not annotation.reviewer_id or not annotation.blind_batch:
            raise ValueError(f"Missing human annotation identifier at {path}:{index}")
        if len(annotation.trace_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in annotation.trace_sha256
        ):
            raise ValueError(f"Invalid trace hash at {path}:{index}")
        annotations.append(annotation)
    keys = [(item.run_id, item.reviewer_id) for item in annotations]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate run_id/reviewer_id rows in {path}")
    return annotations


def quadratic_weighted_kappa(first: Sequence[int], second: Sequence[int]) -> float | None:
    if len(first) != len(second) or not first:
        raise ValueError("Kappa inputs must be non-empty and equally sized.")
    categories = (1, 2, 3)
    size = len(categories) - 1
    observed = sum(((a - b) / size) ** 2 for a, b in zip(first, second)) / len(first)
    first_counts = {value: first.count(value) for value in categories}
    second_counts = {value: second.count(value) for value in categories}
    expected = sum(
        ((a - b) / size) ** 2 * first_counts[a] * second_counts[b]
        for a in categories for b in categories
    ) / (len(first) ** 2)
    if expected == 0:
        return 1.0 if observed == 0 else None
    return 1.0 - observed / expected


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + 1 + end) / 2
        for offset in range(position, end):
            ranks[order[offset]] = rank
        position = end
    return ranks


def spearman(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) != len(second) or not first:
        raise ValueError("Spearman inputs must be non-empty and equally sized.")
    x = _ranks(first)
    y = _ranks(second)
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - x_mean) ** 2 for a in x) * sum((b - y_mean) ** 2 for b in y)
    )
    return numerator / denominator if denominator else None


def _judge_scores(results_dir: Path) -> dict[str, dict[str, int]]:
    records: dict[str, dict[str, int]] = {}
    for path in results_dir.glob("items/*/hy3_judge.json"):
        payload = load_json(path)
        if not isinstance(payload, dict) or not isinstance(payload.get("run_id"), str):
            continue
        records[payload["run_id"]] = {
            score["dimension_id"]: score["score"]
            for score in payload.get("dimension_scores", [])
            if score.get("dimension_id") in DIMENSIONS and isinstance(score.get("score"), int)
        }
    return records


def import_human_annotations(
    experiment_dir: Path,
    *,
    reviewer_files: Sequence[Path],
    arbitration_file: Path | None = None,
) -> dict[str, Any]:
    """Validate two blind reviews, require arbitration for disagreements, and export stats."""

    if len(reviewer_files) != 2:
        raise ValueError("Exactly two independent reviewer CSV files are required.")
    trace_manifest = load_json(experiment_dir / "generation/trace_manifest.json")
    trace_index = {
        task["run_id"]: task["trace_sha256"] for task in trace_manifest.get("tasks", [])
    }
    reviews = [load_annotations(path) for path in reviewer_files]
    by_reviewer = [{item.run_id: item for item in group} for group in reviews]
    if set(by_reviewer[0]) != set(by_reviewer[1]) or len(by_reviewer[0]) != 50:
        raise ValueError("Both reviewers must independently score the same 50 runs.")
    reviewer_ids = {group[0].reviewer_id for group in reviews}
    if len(reviewer_ids) != 2 or any(
        len({item.reviewer_id for item in group}) != 1 for group in reviews
    ):
        raise ValueError("Reviewer files must contain two distinct, stable reviewer_id values.")
    blind_batches = {item.blind_batch for group in reviews for item in group}
    if len(blind_batches) != 1:
        raise ValueError("Both reviewers must use the same blind_batch.")
    for group in reviews:
        for item in group:
            if trace_index.get(item.run_id) != item.trace_sha256:
                raise ValueError(f"Human annotation trace mismatch: {item.run_id}")

    arbitration = {}
    if arbitration_file is not None:
        arbitration = {item.run_id: item for item in load_annotations(arbitration_file)}
        for item in arbitration.values():
            if trace_index.get(item.run_id) != item.trace_sha256:
                raise ValueError(f"Arbitration trace mismatch: {item.run_id}")

    disagreements: list[str] = []
    consensus: dict[str, HumanAnnotation] = {}
    first_by_id, second_by_id = by_reviewer
    for run_id in sorted(first_by_id):
        first = first_by_id[run_id]
        second = second_by_id[run_id]
        differs = first.scores != second.scores or first.gate_failed != second.gate_failed
        if differs:
            disagreements.append(run_id)
            if run_id not in arbitration:
                continue
            consensus[run_id] = arbitration[run_id]
        else:
            consensus[run_id] = HumanAnnotation(
                run_id=run_id,
                trace_sha256=first.trace_sha256,
                reviewer_id="consensus",
                blind_batch=first.blind_batch,
                scores=first.scores,
                gate_failed=first.gate_failed,
                notes=first.notes or second.notes,
            )

    if arbitration:
        extra_arbitration = sorted(set(arbitration) - set(disagreements))
        if extra_arbitration:
            raise ValueError(
                "Arbitration may contain only reviewer disagreements: "
                + ", ".join(extra_arbitration)
            )
        arbitration_ids = {item.reviewer_id for item in arbitration.values()}
        if len(arbitration_ids) != 1 or arbitration_ids & reviewer_ids:
            raise ValueError("Arbitration must use one distinct third reviewer_id.")

    rubric = load_rubric((experiment_dir / load_json(experiment_dir / "experiment.json")["rubric"]).resolve())
    names = {dimension.dimension_id: dimension.name for dimension in rubric.dimensions}
    human_dir = experiment_dir / "validation/human"
    for run_id, item in consensus.items():
        findings = (
            (Finding(code="human_gate", severity="gate", message="Human consensus marked a gate failure."),)
            if item.gate_failed else ()
        )
        record = EvaluationRecord(
            evaluation_id=f"human-consensus-{run_id}",
            run_id=run_id,
            trace_sha256=item.trace_sha256,
            created_at=utc_now_iso(),
            evaluator=EvaluatorInfo(kind="human", name="blind-consensus", version="1.0"),
            rubric=RubricRef(rubric_id=rubric.rubric_id, version=rubric.version, sha256=rubric.sha256),
            status="completed",
            summary="Two-reviewer blind score with disagreement arbitration.",
            dimension_scores=tuple(
                DimensionScore(
                    dimension_id=dimension,
                    name=names[dimension],
                    score=item.scores[dimension],
                    reason=item.notes or "Blind human rating; see raw annotation record.",
                )
                for dimension in DIMENSIONS
            ),
            findings=findings,
            metrics={"score_mean": sum(item.scores.values()) / len(item.scores)},
            metadata={"blind_batch": item.blind_batch, "reviewer_id": item.reviewer_id},
        )
        path = human_dir / "results/items" / run_id / "human.json"
        if path.exists():
            existing = load_json(path)
            existing_scores = {
                score["dimension_id"]: score["score"]
                for score in existing.get("dimension_scores", [])
            }
            if (
                existing.get("trace_sha256") != item.trace_sha256
                or existing_scores != item.scores
                or existing.get("gate_failed") != item.gate_failed
            ):
                raise ValueError(f"Stored human consensus differs for {run_id}.")
        else:
            write_evaluation_record(path, record, overwrite=False)

    judge = _judge_scores(experiment_dir / "results")
    dimension_stats: dict[str, Any] = {}
    for dimension in DIMENSIONS:
        first_values = [first_by_id[run_id].scores[dimension] for run_id in sorted(first_by_id)]
        second_values = [second_by_id[run_id].scores[dimension] for run_id in sorted(first_by_id)]
        paired = [
            (item.scores[dimension], judge[run_id][dimension])
            for run_id, item in consensus.items()
            if run_id in judge and dimension in judge[run_id]
        ]
        dimension_stats[dimension] = {
            "quadratic_weighted_kappa": quadratic_weighted_kappa(first_values, second_values),
            "judge_human_count": len(paired),
            "judge_human_mae": (
                sum(abs(human - model) for human, model in paired) / len(paired) if paired else None
            ),
            "judge_human_spearman": (
                spearman([human for human, _ in paired], [model for _, model in paired])
                if paired else None
            ),
        }
    payload = {
        "schema_version": "1.0",
        "reviewed_count": len(first_by_id),
        "disagreement_count": len(disagreements),
        "pending_arbitration": sorted(set(disagreements) - set(arbitration)),
        "consensus_count": len(consensus),
        "reviewer_ids": sorted(reviewer_ids),
        "dimensions": dimension_stats,
    }
    raw_payload = [
        {
            "run_id": item.run_id,
            "trace_sha256": item.trace_sha256,
            "reviewer_id": item.reviewer_id,
            "blind_batch": item.blind_batch,
            "scores": item.scores,
            "gate_failed": item.gate_failed,
            "notes": item.notes,
        }
        for group in reviews for item in group
    ]
    write_json(human_dir / "raw_annotations.json", raw_payload)
    write_json(human_dir / "agreement.json", payload)
    return payload


__all__ = [
    "DIMENSIONS", "HumanAnnotation", "import_human_annotations", "load_annotations",
    "quadratic_weighted_kappa", "spearman",
]
