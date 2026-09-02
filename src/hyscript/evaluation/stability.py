"""Compare two immutable Hy3 Judge passes over the same frozen traces."""

from __future__ import annotations

from collections import Counter
import csv
import io
from pathlib import Path
from typing import Any

from .formal import atomic_write_text, load_json, write_json
from .human import quadratic_weighted_kappa, spearman


def _judge_records(results_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(results_dir.glob("items/*/hy3_judge.json")):
        payload = load_json(path)
        run_id = payload.get("run_id") if isinstance(payload, dict) else None
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"Judge record has no run_id: {path}")
        if run_id in records:
            raise ValueError(f"Duplicate Judge record for run_id: {run_id}")
        if payload.get("status") != "completed":
            raise ValueError(f"Judge record is not completed: {path}")
        records[run_id] = payload
    if not records:
        raise ValueError(f"No completed Hy3 Judge records found in {results_dir}")
    return records


def _scores(record: dict[str, Any]) -> dict[str, int]:
    values: dict[str, int] = {}
    for item in record.get("dimension_scores", []):
        dimension = item.get("dimension_id") if isinstance(item, dict) else None
        score = item.get("score") if isinstance(item, dict) else None
        if isinstance(dimension, str) and isinstance(score, int):
            values[dimension] = score
    return values


def _fingerprint_sha(record: dict[str, Any]) -> str | None:
    fingerprint = record.get("metadata", {}).get("evaluator_fingerprint", {})
    value = fingerprint.get("sha256") if isinstance(fingerprint, dict) else None
    return value if isinstance(value, str) else None


def _request_distribution(records: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        int(record.get("metadata", {}).get("format_attempts", 0) or 0)
        for record in records.values()
    )
    return {str(key): counts[key] for key in sorted(counts)}


def compare_judge_runs(
    baseline_dir: Path,
    repeat_dir: Path,
    *,
    trace_manifest: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return aggregate stability metrics and one comparison row per frozen trace."""

    baseline = _judge_records(baseline_dir)
    repeat = _judge_records(repeat_dir)
    if set(baseline) != set(repeat):
        missing = sorted(set(baseline) - set(repeat))
        extra = sorted(set(repeat) - set(baseline))
        raise ValueError(
            f"Judge run_id sets differ: missing={len(missing)}, extra={len(extra)}"
        )

    metadata: dict[str, dict[str, Any]] = {}
    if trace_manifest is not None:
        manifest = load_json(trace_manifest)
        for item in manifest.get("tasks", []):
            if isinstance(item, dict) and isinstance(item.get("run_id"), str):
                metadata[item["run_id"]] = item

    run_ids = sorted(baseline)
    first_fingerprints = {_fingerprint_sha(baseline[run_id]) for run_id in run_ids}
    second_fingerprints = {_fingerprint_sha(repeat[run_id]) for run_id in run_ids}
    if len(first_fingerprints) != 1 or len(second_fingerprints) != 1:
        raise ValueError("Each Judge pass must use one stable evaluator fingerprint.")
    if first_fingerprints != second_fingerprints:
        raise ValueError("Judge evaluator fingerprints differ between passes.")

    baseline_scores = {run_id: _scores(baseline[run_id]) for run_id in run_ids}
    repeat_scores = {run_id: _scores(repeat[run_id]) for run_id in run_ids}
    dimension_sets = {tuple(sorted(scores)) for scores in baseline_scores.values()}
    dimension_sets.update(tuple(sorted(scores)) for scores in repeat_scores.values())
    if len(dimension_sets) != 1:
        raise ValueError("Judge dimension coverage differs between records or passes.")
    dimensions = next(iter(dimension_sets))
    if not dimensions:
        raise ValueError("Judge records contain no scored dimensions.")

    rows: list[dict[str, Any]] = []
    all_dimension_agreements = 0
    all_dimensions_exact_count = 0
    baseline_totals: list[float] = []
    repeat_totals: list[float] = []
    for run_id in run_ids:
        first_record = baseline[run_id]
        second_record = repeat[run_id]
        if first_record.get("trace_sha256") != second_record.get("trace_sha256"):
            raise ValueError(f"Trace hash differs between Judge passes: {run_id}")
        first_values = baseline_scores[run_id]
        second_values = repeat_scores[run_id]
        differences = {
            dimension: second_values[dimension] - first_values[dimension]
            for dimension in dimensions
        }
        changed = [dimension for dimension in dimensions if differences[dimension] != 0]
        all_dimension_agreements += len(dimensions) - len(changed)
        all_dimensions_exact_count += not changed
        first_total = first_record.get("metrics", {}).get("normalized_score")
        second_total = second_record.get("metrics", {}).get("normalized_score")
        if not isinstance(first_total, (int, float)) or not isinstance(
            second_total, (int, float)
        ):
            raise ValueError(f"Judge normalized score is unavailable: {run_id}")
        baseline_totals.append(float(first_total))
        repeat_totals.append(float(second_total))
        task = metadata.get(run_id, {})
        row: dict[str, Any] = {
            "run_id": run_id,
            "trace_sha256": first_record["trace_sha256"],
            "task_id": task.get("task_id"),
            "topic": task.get("topic"),
            "target_length": task.get("target_length"),
            "baseline_normalized_score": float(first_total),
            "repeat_normalized_score": float(second_total),
            "normalized_score_delta": float(second_total) - float(first_total),
            "changed_dimension_count": len(changed),
            "changed_dimensions": "|".join(changed),
        }
        for dimension in dimensions:
            row[f"baseline_{dimension}"] = first_values[dimension]
            row[f"repeat_{dimension}"] = second_values[dimension]
            row[f"delta_{dimension}"] = differences[dimension]
        rows.append(row)

    dimension_summary: dict[str, Any] = {}
    for dimension in dimensions:
        first_values = [baseline_scores[run_id][dimension] for run_id in run_ids]
        second_values = [repeat_scores[run_id][dimension] for run_id in run_ids]
        exact = sum(first == second for first, second in zip(first_values, second_values))
        difference_counts = Counter(
            second - first for first, second in zip(first_values, second_values)
        )
        dimension_summary[dimension] = {
            "count": len(run_ids),
            "exact_agreement_count": exact,
            "exact_agreement_rate": exact / len(run_ids),
            "quadratic_weighted_kappa": quadratic_weighted_kappa(
                first_values, second_values
            ),
            "mae": sum(
                abs(first - second)
                for first, second in zip(first_values, second_values)
            ) / len(run_ids),
            "baseline_distribution": {
                str(key): value for key, value in sorted(Counter(first_values).items())
            },
            "repeat_distribution": {
                str(key): value for key, value in sorted(Counter(second_values).items())
            },
            "delta_distribution": {
                str(key): value for key, value in sorted(difference_counts.items())
            },
        }

    total_comparisons = len(run_ids) * len(dimensions)
    score_exact_count = sum(
        first == second for first, second in zip(baseline_totals, repeat_totals)
    )
    summary = {
        "schema_version": "1.0",
        "comparison": "hy3_judge_repeatability",
        "record_count": len(run_ids),
        "dimension_count": len(dimensions),
        "dimensions": dimension_summary,
        "evaluator_fingerprint_sha256": next(iter(first_fingerprints)),
        "baseline_evaluation_id": load_json(baseline_dir / "summary.json").get(
            "evaluation_id"
        ),
        "repeat_evaluation_id": load_json(repeat_dir / "summary.json").get(
            "evaluation_id"
        ),
        "baseline_request_count_distribution": _request_distribution(baseline),
        "repeat_request_count_distribution": _request_distribution(repeat),
        "overall": {
            "dimension_comparison_count": total_comparisons,
            "dimension_exact_agreement_count": all_dimension_agreements,
            "dimension_exact_agreement_rate": all_dimension_agreements
            / total_comparisons,
            "all_dimensions_exact_count": all_dimensions_exact_count,
            "all_dimensions_exact_rate": all_dimensions_exact_count / len(run_ids),
            "normalized_score_exact_count": score_exact_count,
            "normalized_score_exact_rate": score_exact_count / len(run_ids),
            "baseline_normalized_score_mean": sum(baseline_totals) / len(run_ids),
            "repeat_normalized_score_mean": sum(repeat_totals) / len(run_ids),
            "normalized_score_mean_delta": (
                sum(repeat_totals) - sum(baseline_totals)
            ) / len(run_ids),
            "normalized_score_mae": sum(
                abs(first - second)
                for first, second in zip(baseline_totals, repeat_totals)
            ) / len(run_ids),
            "normalized_score_spearman": spearman(
                baseline_totals, repeat_totals
            ),
        },
    }
    return summary, rows


def export_judge_stability(
    baseline_dir: Path,
    repeat_dir: Path,
    output_dir: Path,
    *,
    trace_manifest: Path | None = None,
) -> dict[str, Any]:
    """Write JSON, CSV, and Markdown reports for two complete Judge passes."""

    summary, rows = compare_judge_runs(
        baseline_dir, repeat_dir, trace_manifest=trace_manifest
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(output_dir / "full_comparison.csv", buffer.getvalue())

    overall = summary["overall"]
    lines = [
        "# Hy3 Judge 重复评价内部一致性",
        "",
        f"- 冻结轨迹：{summary['record_count']} 条",
        f"- Judge 维度：{summary['dimension_count']} 个",
        f"- 逐维完全一致率：{overall['dimension_exact_agreement_rate']:.4f}",
        f"- 七维全部一致率：{overall['all_dimensions_exact_rate']:.4f}",
        f"- 归一化总分完全一致率：{overall['normalized_score_exact_rate']:.4f}",
        f"- 首轮/复评平均分：{overall['baseline_normalized_score_mean']:.6f} / "
        f"{overall['repeat_normalized_score_mean']:.6f}",
        f"- 复评平均分变化：{overall['normalized_score_mean_delta']:+.6f}",
        f"- 归一化总分 MAE：{overall['normalized_score_mae']:.6f}",
        f"- 归一化总分 Spearman：{overall['normalized_score_spearman']}",
        "",
        "## 分维度结果",
        "",
        "| 维度 | 完全一致率 | 加权 Kappa | MAE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for dimension, stats in summary["dimensions"].items():
        kappa = stats["quadratic_weighted_kappa"]
        kappa_text = "null" if kappa is None else f"{kappa:.4f}"
        lines.append(
            f"| {dimension} | {stats['exact_agreement_rate']:.4f} | "
            f"{kappa_text} | {stats['mae']:.4f} |"
        )
    disagreements = sorted(
        (row for row in rows if row["changed_dimension_count"]),
        key=lambda row: (
            -int(row["changed_dimension_count"]),
            -abs(float(row["normalized_score_delta"])),
            str(row["run_id"]),
        ),
    )
    lines.extend(["", "## 主要分歧样本", ""])
    if not disagreements:
        lines.append("两轮评分完全一致。")
    else:
        for row in disagreements[:20]:
            lines.append(
                f"- `{row['task_id'] or row['run_id']}`：变化维度 "
                f"{row['changed_dimensions']}；总分差 "
                f"{float(row['normalized_score_delta']):+.6f}。"
            )
    lines.append("")
    atomic_write_text(output_dir / "report.md", "\n".join(lines))
    return summary


__all__ = ["compare_judge_runs", "export_judge_stability"]
