"""Compare frozen baseline/candidate evaluation runs against fixed gates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Sequence


EXPECTED_RUBRIC_VERSION = "1.1.0"
EXPECTED_JUDGE_VERSION = "3.3.0"
EXPECTED_JUDGE_PROMPT = "script-quality-grounded-groups-v3.3"
EXPECTED_RULES_VERSION = "1.5.0"
EXPECTED_AGGREGATE_VERSION = "1.2.0"
TARGET_DIMENSIONS = ("engagement", "oral_fluency")
NON_TARGET_GUARDS = (
    "topic_alignment",
    "theme_information",
    "rhetoric_memorability",
    "logic_structure",
)
DIMENSION_NAMES = {
    "topic_alignment": "选题匹配度",
    "length_compliance": "字数符合度",
    "theme_information": "主题明确与信息量",
    "engagement": "吸引力",
    "oral_fluency": "口播流畅度",
    "rhetoric_memorability": "修辞与记忆点",
    "logic_structure": "语言逻辑与结构",
    "safety_compliance": "合规性",
}
_RUN_SUFFIX = re.compile(r"-run-[A-Za-z0-9T.:-]+$")
_LENGTH_SUFFIX = re.compile(r"-L(\d+)$")


@dataclass(frozen=True, slots=True)
class ScoredItem:
    task_id: str
    run_id: str
    scores: dict[str, int]
    reasons: dict[str, str]
    problem_spans: dict[str, tuple[str, ...]]
    weighted_total: float
    gate_failed: bool
    trace_sha256: str


@dataclass(frozen=True, slots=True)
class TraceAuditItem:
    task_id: str
    run_id: str
    trace_sha256: str
    generation_mode: str
    prompt_versions: dict[str, str]
    prompt_hashes: dict[str, str]
    reference_ids: tuple[str, ...]
    candidate_reference_ids: tuple[str, ...]
    attempted_calls: dict[str, int]
    reported_calls: int
    token_usage: dict[str, int]
    reused_candidate_calls: int
    reused_candidate_reported_calls: int
    reused_candidate_token_usage: dict[str, int]
    length_repair_attempted: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one or more paired fixed-Judge evaluations and write a "
            "Chinese acceptance report. This command never calls an evaluator."
        )
    )
    parser.add_argument("--baseline-dir", type=Path, action="append", required=True)
    parser.add_argument("--candidate-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--baseline-trace-manifest", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--candidate-trace-manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--label", default="三候选主编生成对比")
    return parser


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load JSON: {path}") from exc


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _task_id(source: str) -> str:
    stem = Path(source).stem
    normalized = _RUN_SUFFIX.sub("", stem)
    if not normalized:
        raise ValueError(f"Could not derive task id from source: {source}")
    return normalized


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _string_list(value: Any, *, label: str, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{label} must be a list of non-empty strings.")
    normalized = tuple(value)
    if not allow_empty and not normalized:
        raise ValueError(f"{label} must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must not repeat values.")
    return normalized


def _trace_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def _reused_candidate_usage(
    experiment: dict[str, Any],
    raw_candidates: list[Any],
    *,
    current_trace_path: Path,
    task_id: str,
) -> tuple[int, int, dict[str, int]]:
    token_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "cached_input_tokens": 0,
    }
    source_value = experiment.get("source_candidate_trace")
    if source_value is None:
        return 0, 0, token_usage
    source_sha256 = experiment.get("source_candidate_trace_sha256")
    if (
        not isinstance(source_value, str)
        or not source_value
        or not isinstance(source_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None
    ):
        raise ValueError(f"Frozen candidate source metadata is invalid: {task_id}")
    source_path = Path(source_value)
    if not source_path.is_absolute():
        source_path = current_trace_path.parent / source_path
    source_path = source_path.resolve()
    if _sha256(source_path) != source_sha256:
        raise ValueError(f"Frozen candidate source hash mismatch: {task_id}")
    source_trace = _read_json(source_path)
    source_artifact = (
        source_trace.get("script_artifact") if isinstance(source_trace, dict) else None
    )
    source_config = source_trace.get("config") if isinstance(source_trace, dict) else None
    if not isinstance(source_artifact, dict) or not isinstance(source_config, dict):
        raise ValueError(f"Frozen candidate source is incomplete: {task_id}")
    if source_artifact.get("generation_candidates") != raw_candidates:
        raise ValueError(f"Frozen candidates differ from their source trace: {task_id}")
    request_counts = source_config.get("request_counts")
    attempted_calls = (
        request_counts.get("script_candidate_llm")
        if isinstance(request_counts, dict)
        else None
    )
    if (
        isinstance(attempted_calls, bool)
        or not isinstance(attempted_calls, int)
        or attempted_calls <= 0
    ):
        raise ValueError(f"Frozen candidate source lacks call counts: {task_id}")
    source_usages = source_artifact.get("llm_usages")
    if not isinstance(source_usages, list):
        raise ValueError(f"Frozen candidate source lacks usage: {task_id}")
    candidate_usages = [
        usage
        for usage in source_usages
        if isinstance(usage, dict)
        and isinstance(usage.get("stage"), str)
        and usage["stage"].startswith("script.candidate.")
    ]
    for usage in candidate_usages:
        for key in token_usage:
            value = usage.get(key, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Frozen candidate source has invalid usage: {task_id}")
            token_usage[key] += value
    return attempted_calls, len(candidate_usages), token_usage


def _load_trace_manifest(path: Path) -> dict[str, TraceAuditItem]:
    manifest_path = path.resolve()
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("tasks"), list):
        raise ValueError(f"Invalid generation manifest: {manifest_path}")
    items: dict[str, TraceAuditItem] = {}
    for raw_task in manifest["tasks"]:
        if not isinstance(raw_task, dict):
            raise ValueError("Generation manifest contains an invalid task.")
        if raw_task.get("status") != "completed":
            continue
        task_id = raw_task.get("task_id")
        trace_value = raw_task.get("trace")
        if not isinstance(task_id, str) or not task_id or not isinstance(trace_value, str):
            raise ValueError("Generation manifest task lacks task_id/trace.")
        if task_id in items:
            raise ValueError(f"Generation manifest repeats task id: {task_id}")
        trace_path = _trace_path(manifest_path, trace_value)
        trace = _read_json(trace_path)
        if not isinstance(trace, dict):
            raise ValueError(f"Invalid generation trace: {trace_path}")
        run_id = trace.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"Generation trace lacks run_id: {trace_path}")
        manifest_run_id = raw_task.get("run_id")
        if manifest_run_id is not None and manifest_run_id != run_id:
            raise ValueError(f"Generation manifest run_id mismatch: {task_id}")

        artifact = trace.get("script_artifact")
        lineage = trace.get("lineage")
        config = trace.get("config")
        if not isinstance(artifact, dict) or not isinstance(lineage, dict):
            raise ValueError(f"Generation trace lacks script lineage: {task_id}")
        if not isinstance(config, dict):
            raise ValueError(f"Generation trace lacks config: {task_id}")
        script_text = artifact.get("script_text")
        if not isinstance(script_text, str) or not script_text.strip():
            raise ValueError(f"Generation trace lacks final script text: {task_id}")
        reference_ids = _string_list(
            artifact.get("reference_ids"),
            label=f"{task_id} final reference_ids",
        )
        lineage_reference_ids = _string_list(
            lineage.get("script_reference_ids"),
            label=f"{task_id} lineage script_reference_ids",
        )
        if reference_ids != lineage_reference_ids:
            raise ValueError(f"Generation trace reference lineage mismatch: {task_id}")
        known_reference_map = lineage.get("evidence_to_result_ref")
        if not isinstance(known_reference_map, dict) or not set(reference_ids).issubset(
            known_reference_map
        ):
            raise ValueError(f"Generation trace uses unknown final references: {task_id}")

        generation_mode = artifact.get("generation_mode") or "single"
        if generation_mode not in {"single", "editorial_candidates"}:
            raise ValueError(f"Generation trace has invalid mode: {task_id}")
        raw_prompt_versions = lineage.get("prompt_versions")
        if not isinstance(raw_prompt_versions, dict):
            raise ValueError(f"Generation trace lacks prompt versions: {task_id}")
        prompt_versions = {
            key: value
            for key, value in raw_prompt_versions.items()
            if isinstance(key, str) and isinstance(value, str) and value.strip()
        }
        if not prompt_versions.get("script_generation"):
            raise ValueError(f"Generation trace lacks script prompt version: {task_id}")
        if artifact.get("prompt_version") != prompt_versions["script_generation"]:
            raise ValueError(f"Generation trace script prompt mismatch: {task_id}")

        experiment = config.get("experiment")
        if not isinstance(experiment, dict):
            raise ValueError(f"Generation trace lacks experiment metadata: {task_id}")
        prompt_hashes = {
            key: value
            for key, value in experiment.items()
            if isinstance(key, str)
            and key.endswith("prompt_sha256")
            and isinstance(value, str)
            and re.fullmatch(r"[0-9a-f]{64}", value)
        }
        if "script_system_prompt_sha256" not in prompt_hashes:
            raise ValueError(f"Generation trace lacks script prompt hash: {task_id}")

        raw_candidates = artifact.get("generation_candidates") or []
        if not isinstance(raw_candidates, list):
            raise ValueError(f"Generation trace has invalid candidates: {task_id}")
        candidate_reference_ids: list[str] = []
        if generation_mode == "editorial_candidates":
            if len(raw_candidates) != 3:
                raise ValueError(f"Editorial trace must contain three candidates: {task_id}")
            candidate_ids: list[str] = []
            candidate_versions: set[str] = set()
            for index, raw_candidate in enumerate(raw_candidates, start=1):
                if not isinstance(raw_candidate, dict):
                    raise ValueError(f"Editorial trace has invalid candidate: {task_id}")
                candidate_id = raw_candidate.get("candidate_id")
                strategy = raw_candidate.get("strategy")
                prompt_version = raw_candidate.get("prompt_version")
                if (
                    not isinstance(candidate_id, str)
                    or not candidate_id
                    or not isinstance(strategy, str)
                    or not strategy
                    or not isinstance(prompt_version, str)
                    or not prompt_version
                ):
                    raise ValueError(
                        f"Editorial candidate {index} metadata is incomplete: {task_id}"
                    )
                candidate_ids.append(candidate_id)
                candidate_versions.add(prompt_version)
                refs = _string_list(
                    raw_candidate.get("reference_ids"),
                    label=f"{task_id} {candidate_id} reference_ids",
                )
                if not set(refs).issubset(known_reference_map):
                    raise ValueError(
                        f"Editorial candidate uses unknown references: {task_id}"
                    )
                candidate_reference_ids.extend(refs)
            if len(set(candidate_ids)) != 3 or len(candidate_versions) != 1:
                raise ValueError(f"Editorial candidate identity/version mismatch: {task_id}")
            selected_ids = _string_list(
                artifact.get("selected_candidate_ids"),
                label=f"{task_id} selected_candidate_ids",
            )
            if not set(selected_ids).issubset(candidate_ids):
                raise ValueError(f"Editorial trace selects an unknown candidate: {task_id}")
            if prompt_versions.get("script_candidate") not in candidate_versions:
                raise ValueError(f"Editorial candidate prompt lineage mismatch: {task_id}")
            if artifact.get("editor_prompt_version") != prompt_versions.get(
                "script_editor"
            ):
                raise ValueError(f"Editorial editor prompt lineage mismatch: {task_id}")
            required_hashes = {
                "script_candidate_prompt_sha256",
                "script_editor_prompt_sha256",
            }
            if not required_hashes.issubset(prompt_hashes):
                raise ValueError(f"Editorial trace lacks prompt hashes: {task_id}")
        elif raw_candidates:
            raise ValueError(f"Single-draft trace unexpectedly contains candidates: {task_id}")

        request_counts = config.get("request_counts")
        if not isinstance(request_counts, dict):
            raise ValueError(f"Generation trace lacks request counts: {task_id}")
        attempted_calls: dict[str, int] = {}
        for key in (
            "script_generation_llm",
            "script_candidate_llm",
            "script_editor_llm",
            "script_grounding_review_llm",
        ):
            value = request_counts.get(key, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Generation trace has invalid request count: {task_id}")
            attempted_calls[key] = value
        if attempted_calls["script_generation_llm"] <= 0:
            raise ValueError(f"Generation trace reports no script call: {task_id}")

        raw_usages = artifact.get("llm_usages")
        if not isinstance(raw_usages, list):
            raise ValueError(f"Generation trace lacks script usage: {task_id}")
        token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
            "cached_input_tokens": 0,
        }
        for usage in raw_usages:
            if not isinstance(usage, dict) or not isinstance(usage.get("stage"), str):
                raise ValueError(f"Generation trace has invalid script usage: {task_id}")
            for key in token_usage:
                value = usage.get(key, 0)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"Generation trace has invalid token usage: {task_id}")
                token_usage[key] += value

        reused_candidate_calls = 0
        reused_candidate_reported_calls = 0
        reused_candidate_token_usage = {key: 0 for key in token_usage}
        if generation_mode == "editorial_candidates":
            (
                reused_candidate_calls,
                reused_candidate_reported_calls,
                reused_candidate_token_usage,
            ) = _reused_candidate_usage(
                experiment,
                raw_candidates,
                current_trace_path=trace_path,
                task_id=task_id,
            )

        items[task_id] = TraceAuditItem(
            task_id=task_id,
            run_id=run_id,
            trace_sha256=_sha256(trace_path),
            generation_mode=generation_mode,
            prompt_versions=prompt_versions,
            prompt_hashes=prompt_hashes,
            reference_ids=reference_ids,
            candidate_reference_ids=tuple(candidate_reference_ids),
            attempted_calls=attempted_calls,
            reported_calls=len(raw_usages),
            token_usage=token_usage,
            reused_candidate_calls=reused_candidate_calls,
            reused_candidate_reported_calls=reused_candidate_reported_calls,
            reused_candidate_token_usage=reused_candidate_token_usage,
            length_repair_attempted=bool(artifact.get("length_repair_attempted")),
        )
    if not items:
        raise ValueError(f"Generation manifest contains no completed traces: {path}")
    return items


def _load_trace_manifests(paths: Sequence[Path]) -> dict[str, TraceAuditItem]:
    merged: dict[str, TraceAuditItem] = {}
    for path in paths:
        for task_id, item in _load_trace_manifest(path).items():
            if task_id in merged:
                raise ValueError(f"Generation manifests repeat task id: {task_id}")
            merged[task_id] = item
    if not merged:
        raise ValueError("Generation manifests contain no completed traces.")
    return merged


def _trace_audit_summary(items: dict[str, TraceAuditItem]) -> dict[str, Any]:
    values = list(items.values())
    prompt_versions: dict[str, set[str]] = {}
    prompt_hashes: dict[str, set[str]] = {}
    for item in values:
        for key, value in item.prompt_versions.items():
            prompt_versions.setdefault(key, set()).add(value)
        for key, value in item.prompt_hashes.items():
            prompt_hashes.setdefault(key, set()).add(value)
    attempted_calls = {
        key: sum(item.attempted_calls[key] for item in values)
        for key in values[0].attempted_calls
    }
    token_usage = {
        key: sum(item.token_usage[key] for item in values)
        for key in values[0].token_usage
    }
    reused_candidate_calls = sum(item.reused_candidate_calls for item in values)
    reused_candidate_reported_calls = sum(
        item.reused_candidate_reported_calls for item in values
    )
    reused_candidate_token_usage = {
        key: sum(item.reused_candidate_token_usage[key] for item in values)
        for key in values[0].reused_candidate_token_usage
    }
    logical_pipeline_calls = dict(attempted_calls)
    logical_pipeline_calls["script_generation_llm"] += reused_candidate_calls
    logical_pipeline_calls["script_candidate_llm"] += reused_candidate_calls
    logical_pipeline_token_usage = {
        key: token_usage[key] + reused_candidate_token_usage[key]
        for key in token_usage
    }
    final_reference_counts = [len(item.reference_ids) for item in values]
    return {
        "record_count": len(values),
        "generation_modes": sorted({item.generation_mode for item in values}),
        "prompt_versions": {
            key: sorted(entries) for key, entries in sorted(prompt_versions.items())
        },
        "prompt_hashes": {
            key: sorted(entries) for key, entries in sorted(prompt_hashes.items())
        },
        "attempted_calls": attempted_calls,
        "reported_calls": sum(item.reported_calls for item in values),
        "token_usage": token_usage,
        "frozen_candidate_reuse_trace_count": sum(
            item.reused_candidate_calls > 0 for item in values
        ),
        "reused_candidate_calls": reused_candidate_calls,
        "reused_candidate_reported_calls": reused_candidate_reported_calls,
        "reused_candidate_token_usage": reused_candidate_token_usage,
        "logical_pipeline_calls": logical_pipeline_calls,
        "logical_pipeline_reported_calls": (
            sum(item.reported_calls for item in values)
            + reused_candidate_reported_calls
        ),
        "logical_pipeline_token_usage": logical_pipeline_token_usage,
        "length_repair_count": sum(item.length_repair_attempted for item in values),
        "final_reference_count": sum(final_reference_counts),
        "final_references_per_trace_min": min(final_reference_counts),
        "final_references_per_trace_max": max(final_reference_counts),
        "candidate_reference_count": sum(
            len(item.candidate_reference_ids) for item in values
        ),
    }


def _fixed_fingerprint(manifest: dict[str, Any]) -> str:
    rubric = manifest.get("rubric") or {}
    if rubric.get("version") != EXPECTED_RUBRIC_VERSION:
        raise ValueError("Evaluation does not use frozen Rubric 1.1.0.")
    fingerprint = manifest.get("fingerprint") or {}
    evaluators = fingerprint.get("evaluators") or []
    judge = next(
        (item for item in evaluators if item.get("kind") == "judge"),
        None,
    )
    rules = next(
        (item for item in evaluators if item.get("kind") == "rules"),
        None,
    )
    aggregate = fingerprint.get("aggregator") or {}
    if (
        not isinstance(judge, dict)
        or judge.get("version") != EXPECTED_JUDGE_VERSION
        or judge.get("prompt_version") != EXPECTED_JUDGE_PROMPT
    ):
        raise ValueError("Evaluation does not use frozen Judge v3.3.")
    if not isinstance(rules, dict) or rules.get("version") != EXPECTED_RULES_VERSION:
        raise ValueError("Evaluation does not use frozen deterministic rules v1.5.")
    if aggregate.get("version") != EXPECTED_AGGREGATE_VERSION:
        raise ValueError("Evaluation does not use frozen aggregate v1.2.")
    sha256 = fingerprint.get("sha256")
    if not isinstance(sha256, str) or not sha256:
        raise ValueError("Evaluation fingerprint is missing its SHA-256.")
    return sha256


def _load_evaluation(path: Path) -> tuple[dict[str, ScoredItem], str]:
    root = path.resolve()
    manifest = _read_json(root / "manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid evaluation manifest: {root}")
    fingerprint = _fixed_fingerprint(manifest)
    run_to_task: dict[str, str] = {}
    run_to_trace_hash: dict[str, str] = {}
    for raw_input in manifest.get("inputs") or []:
        if not isinstance(raw_input, dict):
            raise ValueError("Evaluation manifest contains an invalid input.")
        run_id = raw_input.get("run_id")
        source = raw_input.get("source")
        trace_sha256 = raw_input.get("trace_sha256")
        if (
            not isinstance(run_id, str)
            or not isinstance(source, str)
            or not isinstance(trace_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", trace_sha256) is None
        ):
            raise ValueError(
                "Evaluation manifest input lacks run_id/source/trace_sha256."
            )
        task_id = _task_id(source)
        if task_id in run_to_task.values():
            raise ValueError(f"Evaluation repeats task id: {task_id}")
        run_to_task[run_id] = task_id
        run_to_trace_hash[run_id] = trace_sha256

    items: dict[str, ScoredItem] = {}
    for run_id, task_id in run_to_task.items():
        item_root = root / "items" / run_id
        combined = _read_json(item_root / "combined.json")
        judge = _read_json(item_root / "hy3_judge.json")
        if not isinstance(combined, dict) or combined.get("status") != "completed":
            raise ValueError(f"Combined evaluation is incomplete: {run_id}")
        raw_scores = combined.get("dimension_scores")
        if not isinstance(raw_scores, list):
            raise ValueError(f"Combined evaluation lacks dimensions: {run_id}")
        scores: dict[str, int] = {}
        reasons: dict[str, str] = {}
        for raw_score in raw_scores:
            dimension_id = raw_score.get("dimension_id")
            score = raw_score.get("score")
            reason = raw_score.get("reason")
            if not isinstance(dimension_id, str) or not isinstance(score, int):
                raise ValueError(f"Invalid dimension score: {run_id}")
            scores[dimension_id] = score
            reasons[dimension_id] = reason if isinstance(reason, str) else ""
        missing = set(DIMENSION_NAMES) - set(scores)
        if missing:
            raise ValueError(f"Evaluation lacks dimensions {sorted(missing)}: {run_id}")
        span_evidence = ((judge.get("metadata") or {}).get("span_evidence") or {})
        problem_spans = {
            dimension_id: tuple(
                span
                for span in (span_evidence.get(dimension_id) or {}).get(
                    "problem_spans", []
                )
                if isinstance(span, str)
            )
            for dimension_id in TARGET_DIMENSIONS
        }
        metrics = combined.get("metrics") or {}
        weighted_total = metrics.get("weighted_total")
        if not isinstance(weighted_total, (int, float)):
            raise ValueError(f"Evaluation lacks weighted_total: {run_id}")
        combined_trace_sha256 = combined.get("trace_sha256")
        if combined_trace_sha256 != run_to_trace_hash[run_id]:
            raise ValueError(f"Evaluation trace hash mismatch: {run_id}")
        items[task_id] = ScoredItem(
            task_id=task_id,
            run_id=run_id,
            scores=scores,
            reasons=reasons,
            problem_spans=problem_spans,
            weighted_total=float(weighted_total),
            gate_failed=bool(combined.get("gate_failed")),
            trace_sha256=combined_trace_sha256,
        )
    if not items:
        raise ValueError(f"Evaluation contains no scored items: {root}")
    return items, fingerprint


def _means(items: Sequence[ScoredItem]) -> dict[str, float]:
    return {
        dimension_id: sum(item.scores[dimension_id] for item in items) / len(items)
        for dimension_id in DIMENSION_NAMES
    }


def _length_means(items: Sequence[ScoredItem]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[ScoredItem]] = {}
    for item in items:
        match = _LENGTH_SUFFIX.search(item.task_id)
        if match is None:
            raise ValueError(f"Task id has no -L<length> suffix: {item.task_id}")
        groups.setdefault(match.group(1), []).append(item)
    return {
        length: {
            dimension_id: sum(
                item.scores[dimension_id] for item in group_items
            )
            / len(group_items)
            for dimension_id in TARGET_DIMENSIONS
        }
        for length, group_items in sorted(groups.items(), key=lambda pair: int(pair[0]))
    }


def _validate_trace_links(
    evaluation: dict[str, ScoredItem],
    traces: dict[str, TraceAuditItem],
    *,
    label: str,
) -> None:
    if set(evaluation) != set(traces):
        missing_evaluation = sorted(set(traces) - set(evaluation))
        missing_trace = sorted(set(evaluation) - set(traces))
        raise ValueError(
            f"{label} evaluation/trace task mismatch; "
            f"missing_evaluation={missing_evaluation}, missing_trace={missing_trace}."
        )
    for task_id, scored in evaluation.items():
        trace = traces[task_id]
        if scored.run_id != trace.run_id or scored.trace_sha256 != trace.trace_sha256:
            raise ValueError(f"{label} evaluation is not linked to trace: {task_id}")


def _population_variance(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Variance requires at least one value.")
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _repeat_variance(runs: Sequence[dict[str, ScoredItem]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("Repeat variance requires at least one evaluation.")
    task_ids = set(runs[0])
    if any(set(run) != task_ids for run in runs[1:]):
        raise ValueError("Repeated evaluations contain different task ids.")
    result: dict[str, Any] = {}
    for dimension_id in DIMENSION_NAMES:
        repetition_means = [
            sum(item.scores[dimension_id] for item in run.values()) / len(run)
            for run in runs
        ]
        item_variances = [
            _population_variance(
                [run[task_id].scores[dimension_id] for run in runs]
            )
            for task_id in sorted(task_ids)
        ]
        result[dimension_id] = {
            "repetition_means": repetition_means,
            "mean_variance": _population_variance(repetition_means),
            "mean_item_score_variance": sum(item_variances) / len(item_variances),
            "changed_item_count": sum(value > 0 for value in item_variances),
        }
    return result


def _compare_repetition(
    baseline: dict[str, ScoredItem],
    candidate: dict[str, ScoredItem],
    *,
    repetition: int,
) -> dict[str, Any]:
    if set(baseline) != set(candidate):
        raise ValueError("Baseline and candidate task ids do not match.")
    keys = sorted(baseline)
    baseline_items = [baseline[key] for key in keys]
    candidate_items = [candidate[key] for key in keys]
    baseline_means = _means(baseline_items)
    candidate_means = _means(candidate_items)
    length_means = _length_means(candidate_items)
    checks: dict[str, bool] = {
        "engagement_mean": candidate_means["engagement"] >= 2.90,
        "oral_fluency_mean": candidate_means["oral_fluency"] >= 2.85,
        "length_buckets": all(
            values["engagement"] >= 2.80
            and values["oral_fluency"] >= 2.70
            for values in length_means.values()
        ),
        "non_target_guard": all(
            candidate_means[dimension_id]
            >= baseline_means[dimension_id] - 0.05
            for dimension_id in NON_TARGET_GUARDS
        ),
        "safety_not_lower": (
            candidate_means["safety_compliance"]
            >= baseline_means["safety_compliance"]
        ),
        "length_not_lower": (
            candidate_means["length_compliance"]
            >= baseline_means["length_compliance"]
        ),
        "no_safety_or_length_one": all(
            item.scores["safety_compliance"] > 1
            and item.scores["length_compliance"] > 1
            for item in candidate_items
        ),
        "no_gates": not any(item.gate_failed for item in candidate_items),
        "trace_prompt_references_complete": True,
    }
    paired = {
        "wins": sum(
            candidate[key].weighted_total > baseline[key].weighted_total for key in keys
        ),
        "ties": sum(
            candidate[key].weighted_total == baseline[key].weighted_total
            for key in keys
        ),
        "losses": sum(
            candidate[key].weighted_total < baseline[key].weighted_total for key in keys
        ),
    }
    low_items = [
        {
            "task_id": key,
            "engagement": candidate[key].scores["engagement"],
            "engagement_reason": candidate[key].reasons["engagement"],
            "engagement_problem_spans": list(
                candidate[key].problem_spans["engagement"]
            ),
            "oral_fluency": candidate[key].scores["oral_fluency"],
            "oral_fluency_reason": candidate[key].reasons["oral_fluency"],
            "oral_problem_spans": list(
                candidate[key].problem_spans["oral_fluency"]
            ),
        }
        for key in keys
        if candidate[key].scores["engagement"] < 3
        or candidate[key].scores["oral_fluency"] < 3
    ]
    return {
        "repetition": repetition,
        "record_count": len(keys),
        "baseline_means": baseline_means,
        "candidate_means": candidate_means,
        "dimension_deltas": {
            dimension_id: candidate_means[dimension_id] - baseline_means[dimension_id]
            for dimension_id in DIMENSION_NAMES
        },
        "candidate_length_means": length_means,
        "paired_total_score": paired,
        "checks": checks,
        "passed": all(checks.values()),
        "low_target_items": low_items,
    }


def _markdown(payload: dict[str, Any]) -> str:
    baseline_audit = payload["trace_audit"]["baseline"]
    candidate_audit = payload["trace_audit"]["candidate"]
    baseline_variance = payload["repeat_variance"]["baseline"]
    candidate_variance = payload["repeat_variance"]["candidate"]

    def joined_versions(audit: dict[str, Any]) -> str:
        return "；".join(
            f"{key}={','.join(values)}"
            for key, values in audit["prompt_versions"].items()
        )

    def quoted_spans(values: Sequence[str]) -> str:
        normalized = [" ".join(value.split()) for value in values if value.strip()]
        return "；".join(f"“{value}”" for value in normalized) if normalized else "无"

    lines = [
        f"# {payload['label']}",
        "",
        "## 结论",
        "",
        f"- 总体验收：**{'通过' if payload['passed'] else '未通过'}**。",
        f"- 固定评测指纹：`{payload['evaluator_fingerprint']}`。",
        "- 所有重复评测都必须单独通过，不能用一次高分抵消另一次低分。",
        "",
        "## 生成 Trace 与调用量审计",
        "",
        f"- 旧流程模式：`{','.join(baseline_audit['generation_modes'])}`；"
        f"新流程模式：`{','.join(candidate_audit['generation_modes'])}`。",
        f"- 旧流程Prompt版本：{joined_versions(baseline_audit)}。",
        f"- 新流程Prompt版本：{joined_versions(candidate_audit)}。",
        f"- 最终引用ID总数：旧流程 {baseline_audit['final_reference_count']}，"
        f"新流程 {candidate_audit['final_reference_count']}；"
        f"新流程候选引用记录 {candidate_audit['candidate_reference_count']} 条。",
        f"- 唯一一次长度修复触发稿数：旧流程 {baseline_audit['length_repair_count']}，"
        f"新流程 {candidate_audit['length_repair_count']}。",
        f"- 冻结候选复用：旧流程 {baseline_audit['frozen_candidate_reuse_trace_count']} 篇，"
        f"新流程 {candidate_audit['frozen_candidate_reuse_trace_count']} 篇；"
        "复用候选的原始调用会计入逻辑全流程成本，但不会冒充本轮新请求。",
        "",
        "| 流程 | 本轮生成调用 | 复用候选调用 | 逻辑全流程调用 | 逻辑候选调用 | 主编调用 | 逻辑成功返回调用 | 逻辑总Token |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| 旧流程 | {baseline_audit['attempted_calls']['script_generation_llm']} | "
            f"{baseline_audit['reused_candidate_calls']} | "
            f"{baseline_audit['logical_pipeline_calls']['script_generation_llm']} | "
            f"{baseline_audit['logical_pipeline_calls']['script_candidate_llm']} | "
            f"{baseline_audit['attempted_calls']['script_editor_llm']} | "
            f"{baseline_audit['logical_pipeline_reported_calls']} | "
            f"{baseline_audit['logical_pipeline_token_usage']['total_tokens']} |"
        ),
        (
            f"| 新流程 | {candidate_audit['attempted_calls']['script_generation_llm']} | "
            f"{candidate_audit['reused_candidate_calls']} | "
            f"{candidate_audit['logical_pipeline_calls']['script_generation_llm']} | "
            f"{candidate_audit['logical_pipeline_calls']['script_candidate_llm']} | "
            f"{candidate_audit['attempted_calls']['script_editor_llm']} | "
            f"{candidate_audit['logical_pipeline_reported_calls']} | "
            f"{candidate_audit['logical_pipeline_token_usage']['total_tokens']} |"
        ),
        "",
        "## 两次评分方差",
        "",
        "这里的“均分方差”衡量两次整组均分是否漂移；“变动稿数”表示两次评分不一致的稿件数量。",
        "",
        "| 维度 | 旧流程均分方差 | 旧流程变动稿数 | 新流程均分方差 | 新流程变动稿数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for dimension_id, dimension_name in DIMENSION_NAMES.items():
        lines.append(
            f"| {dimension_name} | "
            f"{baseline_variance[dimension_id]['mean_variance']:.6f} | "
            f"{baseline_variance[dimension_id]['changed_item_count']} | "
            f"{candidate_variance[dimension_id]['mean_variance']:.6f} | "
            f"{candidate_variance[dimension_id]['changed_item_count']} |"
        )
    lines.append("")
    for repetition in payload["repetitions"]:
        candidate = repetition["candidate_means"]
        baseline = repetition["baseline_means"]
        paired = repetition["paired_total_score"]
        lines.extend(
            [
                f"## 第 {repetition['repetition']} 次固定评测",
                "",
                f"- 结果：**{'通过' if repetition['passed'] else '未通过'}**。",
                f"- 吸引力：{baseline['engagement']:.2f} → {candidate['engagement']:.2f}。",
                f"- 口播流畅度：{baseline['oral_fluency']:.2f} → {candidate['oral_fluency']:.2f}。",
                f"- 逐稿总分：{paired['wins']}胜、{paired['ties']}平、{paired['losses']}负。",
                "",
                "| 目标长度 | 吸引力 | 口播流畅度 |",
                "|---:|---:|---:|",
            ]
        )
        for length, values in repetition["candidate_length_means"].items():
            lines.append(
                f"| {length} | {values['engagement']:.2f} | "
                f"{values['oral_fluency']:.2f} |"
            )
        lines.extend(["", "### 验收项", ""])
        for check, passed in repetition["checks"].items():
            lines.append(f"- {'通过' if passed else '未通过'}：`{check}`")
        lines.extend(["", "### 目标维度未满分稿", ""])
        if not repetition["low_target_items"]:
            lines.append("- 无。")
        else:
            for item in repetition["low_target_items"]:
                lines.append(
                    f"- `{item['task_id']}`：吸引力 {item['engagement']}，"
                    f"口播 {item['oral_fluency']}；"
                    f"{item['engagement_reason']}；{item['oral_fluency_reason']}。"
                    f"吸引力低分片段：{quoted_spans(item['engagement_problem_spans'])}；"
                    f"口播低分片段：{quoted_spans(item['oral_problem_spans'])}。"
                )
        lines.append("")
    lines.extend(
        [
            "## 供后续 AI 阅读",
            "",
            "1. 本报告只比较已经冻结的生成Trace，不得把Judge调用并入生成流程。",
            "2. 只有所有重复评测均通过时，才能把候选流程设为生产默认。",
            "3. 未满分稿的原文片段保存在同名JSON的`low_target_items`中，后续Prompt修改应抽象成通用规则，禁止写入题目专属原句。",
            "4. `本轮生成调用`只统计当前Trace实际发出的请求；`逻辑全流程调用`再加上可核验来源Trace中的冻结候选调用。结构修复与长度修复都会计入尝试调用。",
            "5. `逻辑成功返回调用`只统计服务返回了可记录用量的本轮或来源请求；失败但已尝试的请求只进入尝试调用。",
            "6. Trace审计会逐篇校验评测`run_id`、Trace SHA-256、最终正文、Prompt版本与哈希、引用ID、候选ID和调用统计；任何缺失都会阻止报告生成。",
            "7. `均分方差`越接近0，表示两次整组评分越稳定；它不能替代每次独立通过验收门槛。",
            "8. 报告位于Git忽略目录，不得移入受跟踪文档。",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    if len(args.baseline_dir) != len(args.candidate_dir):
        raise ValueError("baseline-dir and candidate-dir counts must match.")
    baseline_traces = _load_trace_manifests(args.baseline_trace_manifest)
    candidate_traces = _load_trace_manifests(args.candidate_trace_manifest)
    if set(baseline_traces) != set(candidate_traces):
        raise ValueError("Baseline and candidate generation task ids do not match.")
    repetitions: list[dict[str, Any]] = []
    baseline_runs: list[dict[str, ScoredItem]] = []
    candidate_runs: list[dict[str, ScoredItem]] = []
    evaluator_fingerprint: str | None = None
    for index, (baseline_dir, candidate_dir) in enumerate(
        zip(args.baseline_dir, args.candidate_dir, strict=True),
        start=1,
    ):
        baseline, baseline_fingerprint = _load_evaluation(baseline_dir)
        candidate, candidate_fingerprint = _load_evaluation(candidate_dir)
        _validate_trace_links(
            baseline,
            baseline_traces,
            label=f"Baseline repetition {index}",
        )
        _validate_trace_links(
            candidate,
            candidate_traces,
            label=f"Candidate repetition {index}",
        )
        if baseline_fingerprint != candidate_fingerprint:
            raise ValueError("Baseline and candidate evaluator fingerprints differ.")
        if evaluator_fingerprint not in {None, baseline_fingerprint}:
            raise ValueError("Repeated evaluations use different fingerprints.")
        evaluator_fingerprint = baseline_fingerprint
        baseline_runs.append(baseline)
        candidate_runs.append(candidate)
        repetitions.append(
            _compare_repetition(baseline, candidate, repetition=index)
        )
    payload = {
        "schema_version": "1.0",
        "label": args.label,
        "evaluator_fingerprint": evaluator_fingerprint,
        "thresholds": {
            "engagement_mean": 2.90,
            "oral_fluency_mean": 2.85,
            "length_bucket_engagement": 2.80,
            "length_bucket_oral_fluency": 2.70,
            "non_target_max_mean_drop": 0.05,
        },
        "trace_audit": {
            "baseline": _trace_audit_summary(baseline_traces),
            "candidate": _trace_audit_summary(candidate_traces),
        },
        "repeat_variance": {
            "baseline": _repeat_variance(baseline_runs),
            "candidate": _repeat_variance(candidate_runs),
        },
        "repetitions": repetitions,
        "passed": all(item["passed"] for item in repetitions),
    }
    _atomic_write(
        args.output_json.resolve(),
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    _atomic_write(args.output_md.resolve(), _markdown(payload))
    print(
        json.dumps(
            {
                "passed": payload["passed"],
                "output_json": str(args.output_json.resolve()),
                "output_md": str(args.output_md.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["passed"] else 1


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    raise SystemExit(main())
