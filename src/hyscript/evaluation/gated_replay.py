"""Rebuild current gated results from frozen detector and score records, offline."""

from __future__ import annotations

import csv
import hashlib
import io
import os
from pathlib import Path
from typing import Any

from .gates import recorded_gates
from .io import load_frozen_trace, load_json_object, write_json_object
from .judge import Hy3JudgeEvaluator, JudgeConfig
from .models import evaluation_record_from_dict
from .rubric import load_rubric
from .runner import BatchEvaluationConfig, BatchEvaluationRunner


class _NoNetworkClient:
    async def complete(self, *args: Any, **kwargs: Any):
        raise RuntimeError("Offline replay requires an existing, matching Judge record.")


def _recorded_judge(results: Path) -> Hy3JudgeEvaluator:
    manifest = load_json_object(results / "manifest.json")
    fingerprints = manifest["fingerprint"]["evaluators"]
    fp = next(item for item in fingerprints if item["kind"] == "judge")
    return Hy3JudgeEvaluator(
        _NoNetworkClient(), model_name=fp["model"],
        config=JudgeConfig(**fp["config"]["request"]),
        sampling_parameters=fp["config"]["sampling_parameters"],
    )


def export_gated_results(rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    """Export only post-gate scores, leaving blocked score cells empty."""
    table: list[dict[str, Any]] = []
    for item in rows:
        record = evaluation_record_from_dict(load_json_object(
            output / "items" / item["run_id"] / "combined.json"
        ))
        checks = record.metadata["attack_gates"]
        row = {
            "task_id": item["task_id"], "run_id": record.run_id,
            "trace_sha256": record.trace_sha256,
            "topic": item.get("topic", ""), "domain": item.get("domain", ""),
            "target_length": item.get("target_length", ""),
            "reward_hacking_passed": checks["reward_hacking"]["passed"],
            "citation_passed": checks["citation"]["passed"],
            "gate_failed": record.gate_failed,
            "gate_reasons": "；".join(f.message for f in record.findings if f.severity == "gate"),
            "final_score": record.metrics["final_score"],
        }
        row.update({score.dimension_id: score.score for score in record.dimension_scores})
        table.append(row)
    fields = list(dict.fromkeys(key for row in table for key in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(table)
    (output / "full_results.csv").write_text(buffer.getvalue(), encoding="utf-8")
    summary = load_json_object(output / "summary.json")["aggregate"]
    (output / "report.md").write_text(
        "# 双门控正式评估\n\n"
        "所有成稿先经过 Reward-hacking 与引用风险门控，再对通过者计算八维分数。"
        "未通过者保留失败原因，质量分为空；检测异常属于未完成，不视为通过。\n\n"
        f"- 完成评估：{summary['record_count']}\n"
        f"- 门控拦截：{summary['gate_failed_count']}\n"
        f"- 有效八维分数：{summary['eligible_count']}\n"
        f"- 通过后均分：{summary['final_score_mean']}\n\n"
        "本次复用已有冻结检测与评分记录，只重新应用门控并汇总；"
        "未重新调用模型或搜索。门控与质量评分版本见 manifest.json，"
        "逐项输入哈希、门控原因与结果见 full_results.csv 和 items/。\n",
        encoding="utf-8",
    )
    return summary


async def replay_gated_evaluation(
    *, trace_manifest: Path, source_results: Path, reward_cache: Path,
    citation_cache: Path, output: Path, rubric_path: Path,
    selected_run_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Use the production runner with cache-only detectors and Judge."""
    if output.resolve() == source_results.resolve():
        raise ValueError("Gated replay must use a new output directory.")
    manifest = load_json_object(trace_manifest)
    rows: list[dict[str, Any]] = []
    paths: list[Path] = []
    for item in manifest["tasks"]:
        path = (trace_manifest.parent / item["trace"]).resolve()
        trace = load_frozen_trace(path)
        if selected_run_ids is not None and trace.run_id not in selected_run_ids:
            continue
        expected_hash = item.get("trace_sha256", trace.trace_sha256)
        if expected_hash != trace.trace_sha256:
            raise ValueError("Source manifest trace hash differs from the frozen file.")
        rows.append({**item, "run_id": trace.run_id, "topic": trace.task["topic"]})
        paths.append(path)
    if not rows or len({row["run_id"] for row in rows}) != len(rows):
        raise ValueError("Replay input must contain distinct frozen runs.")
    if selected_run_ids is not None and {row["run_id"] for row in rows} != selected_run_ids:
        raise ValueError("Replay input does not cover the selected runs.")
    runner = BatchEvaluationRunner(
        load_rubric(rubric_path),
        BatchEvaluationConfig(
            output_dir=output, evaluators=("rules", "judge"), concurrency=8,
            reuse_results_dir=source_results,
        ),
        judge_evaluator=_recorded_judge(source_results),
        gate_evaluator=recorded_gates(reward_cache, citation_cache),
    )
    result = await runner.run(paths)
    if result.failed_count:
        errors = "; ".join(outcome.message or "" for outcome in result.outcomes if outcome.status == "failed")
        raise RuntimeError(f"Gated replay incomplete ({result.failed_count}): {errors[:500]}")
    write_json_object(output / "replay_sources.json", {
        "mode": "offline_frozen_record_replay", "new_model_calls": 0, "new_search_calls": 0,
        "path_base": "directory containing replay_sources.json",
        "inputs": {
            name: {
                "path": os.path.relpath(path.resolve(), output.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in {
                "traces": trace_manifest, "scores": source_results / "manifest.json",
                "reward_hacking": reward_cache / "manifest.json", "citation": citation_cache / "manifest.json",
            }.items()
        },
    }, overwrite=True)
    return export_gated_results(rows, output)


def export_gated_comparison(baseline: Path, candidate: Path, output: Path) -> dict[str, Any]:
    def rows(directory):
        with (directory / "full_results.csv").open(encoding="utf-8", newline="") as handle:
            return {row["task_id"]: row for row in csv.DictReader(handle)}
    first, second = rows(baseline), rows(candidate)
    if set(first) != set(second):
        raise ValueError("Gated comparison requires the same task set.")
    pairs = [
        (float(first[key]["final_score"]), float(second[key]["final_score"]))
        for key in sorted(first)
        if first[key]["final_score"] and second[key]["final_score"]
    ]
    summary = {
        "task_count": len(first), "both_passed_count": len(pairs),
        "excluded_by_gate_count": len(first) - len(pairs),
        "candidate_wins": sum(b > a for a, b in pairs),
        "ties": sum(b == a for a, b in pairs),
        "candidate_losses": sum(b < a for a, b in pairs),
        "paired_mean_delta": sum(b - a for a, b in pairs) / len(pairs) if pairs else None,
    }
    write_json_object(output / "comparison.json", summary, overwrite=True)
    return summary
