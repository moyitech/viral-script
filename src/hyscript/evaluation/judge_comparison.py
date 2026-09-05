"""Versioned Hy3-versus-Luna Judge comparison over frozen formal traces."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from hyscript.config import PROJECT_ROOT, get_settings

from .formal import atomic_write_text, load_json, sha256_file, write_json
from .human import quadratic_weighted_kappa, spearman
from .rubric import load_rubric
from .stability import (
    combine_judge_comparisons,
    compare_judge_runs,
    export_judge_stability,
)


COMPARISON_SCHEMA_VERSION = "1.0"
EXPECTED_TRACE_COUNT = 300
JUDGE_CONCURRENCY = 512
LUNA_MODEL_ID = "gpt-5.6-luna-cdx"
LUNA_MODEL_ALIAS = "gpt-5.6-luna"
LUNA_REASONING_EFFORT = "xhigh"
GLM_MODEL_ID = "glm-5.3-flash"
GLM_MODEL_ALIAS = "glm-5.3-flash"
GLM_REASONING_EFFORT = "max"
HY3_MODEL_ALIAS = "hy3"
HY3_REASONING_EFFORT = "high"

SOURCE_LABELS = {
    "baseline": "三候选主编基线",
    "single_shot": "端到端直接生成",
}
REPORT_DISPLAY_NAMES = {
    "luna": "GPT-5.6-Luna",
    "glm": "GLM-5.3-Flash",
}


@dataclass(frozen=True, slots=True)
class CandidateJudgeSpec:
    key: str
    model_id: str
    display_name: str
    reasoning_effort: str

    def config(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "reasoning_effort": self.reasoning_effort,
            "result_source": "new",
        }


LUNA_CANDIDATE = CandidateJudgeSpec(
    key="luna",
    model_id=LUNA_MODEL_ID,
    display_name=LUNA_MODEL_ALIAS,
    reasoning_effort=LUNA_REASONING_EFFORT,
)
GLM_CANDIDATE = CandidateJudgeSpec(
    key="glm",
    model_id=GLM_MODEL_ID,
    display_name=GLM_MODEL_ALIAS,
    reasoning_effort=GLM_REASONING_EFFORT,
)
KNOWN_CANDIDATES = {
    candidate.key: candidate for candidate in (LUNA_CANDIDATE, GLM_CANDIDATE)
}


def _candidate_spec(config: dict[str, Any]) -> CandidateJudgeSpec:
    judges = config.get("judges")
    if not isinstance(judges, dict):
        raise ValueError("Comparison Judge configuration is missing.")
    keys = [key for key in judges if key != "hy3"]
    if len(keys) != 1 or keys[0] not in KNOWN_CANDIDATES:
        raise ValueError("Comparison must contain exactly one supported candidate Judge.")
    candidate = KNOWN_CANDIDATES[keys[0]]
    if judges.get(candidate.key) != candidate.config():
        raise ValueError(f"Frozen {candidate.display_name} Judge configuration changed.")
    return candidate


def _relative(path: Path, base: Path) -> str:
    try:
        value = os.path.relpath(path.resolve(), base.resolve())
    except ValueError:
        value = str(path.resolve())
    return value.replace(os.sep, "/")


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _tree_sha256(directory: Path, filename: str) -> str:
    digest = hashlib.sha256()
    paths = sorted(directory.glob(f"items/*/{filename}"))
    if not paths:
        raise ValueError(f"No {filename} records found in {directory}")
    for path in paths:
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _manifest_items(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, list):
        raise ValueError(f"Trace manifest has no tasks list: {path}")
    items: dict[str, dict[str, Any]] = {}
    for raw in tasks:
        if not isinstance(raw, dict) or raw.get("status") != "completed":
            raise ValueError(f"Trace manifest contains an incomplete task: {path}")
        task_id = raw.get("task_id")
        run_id = raw.get("run_id")
        trace_value = raw.get("trace")
        trace_sha256 = raw.get("trace_sha256")
        if not all(isinstance(value, str) and value for value in (
            task_id,
            run_id,
            trace_value,
            trace_sha256,
        )):
            raise ValueError(f"Trace manifest task is missing identity fields: {path}")
        if task_id in items:
            raise ValueError(f"Trace manifest repeats task_id {task_id}: {path}")
        trace_path = _resolve(path.parent, trace_value)
        if not trace_path.is_file() or sha256_file(trace_path) != trace_sha256:
            raise ValueError(f"Frozen trace hash mismatch for {task_id}: {trace_path}")
        items[task_id] = {**raw, "trace_path": str(trace_path)}
    return items


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
            raise ValueError(f"Judge record is incomplete: {path}")
        records[run_id] = payload
    if not records:
        raise ValueError(f"No completed Judge records found in {results_dir}")
    return records


def _judge_fingerprint(record: dict[str, Any]) -> dict[str, Any]:
    fingerprint = record.get("metadata", {}).get("evaluator_fingerprint")
    if not isinstance(fingerprint, dict) or not isinstance(
        fingerprint.get("sha256"), str
    ):
        raise ValueError("Judge record has no evaluator fingerprint.")
    return fingerprint


def _validate_result_set(
    results_dir: Path,
    items: dict[str, dict[str, Any]],
    *,
    require_combined: bool,
    expected_model: str | None = None,
    expected_effort: str | None = None,
) -> dict[str, Any]:
    records = _judge_records(results_dir)
    expected_by_run = {item["run_id"]: item for item in items.values()}
    if set(records) != set(expected_by_run):
        raise ValueError(
            f"Judge coverage differs from the frozen manifest in {results_dir}: "
            f"expected={len(expected_by_run)} actual={len(records)}"
        )
    fingerprints: dict[str, dict[str, Any]] = {}
    for run_id, record in records.items():
        expected = expected_by_run[run_id]
        if record.get("trace_sha256") != expected["trace_sha256"]:
            raise ValueError(f"Judge trace hash mismatch for {run_id}")
        fingerprint = _judge_fingerprint(record)
        fingerprints[fingerprint["sha256"]] = fingerprint
    if len(fingerprints) != 1:
        raise ValueError(f"Judge result set contains mixed fingerprints: {results_dir}")
    fingerprint = next(iter(fingerprints.values()))
    if expected_model is not None and fingerprint.get("model") != expected_model:
        raise ValueError(
            f"Unexpected Judge model in {results_dir}: {fingerprint.get('model')}"
        )
    request = fingerprint.get("config", {}).get("request", {})
    if expected_effort is not None and request.get("reasoning_effort") != expected_effort:
        raise ValueError(
            f"Unexpected reasoning effort in {results_dir}: "
            f"{request.get('reasoning_effort')}"
        )
    if require_combined:
        combined = list(results_dir.glob("items/*/combined.json"))
        if len(combined) != len(items):
            raise ValueError(
                f"Expected {len(items)} combined records, found {len(combined)} "
                f"in {results_dir}"
            )
    manifest = results_dir / "manifest.json"
    summary = results_dir / "summary.json"
    if not manifest.is_file() or not summary.is_file():
        raise ValueError(f"Evaluation manifest or summary is missing: {results_dir}")
    return {
        "record_count": len(records),
        "judge_fingerprint": fingerprint,
        "manifest_sha256": sha256_file(manifest),
        "summary_sha256": sha256_file(summary),
        "judge_records_sha256": _tree_sha256(results_dir, "hy3_judge.json"),
        "combined_records_sha256": (
            _tree_sha256(results_dir, "combined.json") if require_combined else None
        ),
    }


def _source_descriptor(
    experiment_dir: Path,
    source_experiment: Path,
) -> dict[str, Any]:
    source_experiment = source_experiment.resolve()
    experiment_config = source_experiment / "experiment.json"
    trace_manifest = source_experiment / "generation/trace_manifest.json"
    first_results = source_experiment / "results"
    repeat_results = source_experiment / "validation/stability/repeat-001/results"
    stability_summary = source_experiment / "validation/stability/repeat-001/summary.json"
    for path in (experiment_config, trace_manifest, stability_summary):
        if not path.is_file():
            raise ValueError(f"Required source artifact is missing: {path}")
    items = _manifest_items(trace_manifest)
    if len(items) != EXPECTED_TRACE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TRACE_COUNT} frozen traces in {trace_manifest}, "
            f"found {len(items)}"
        )
    first = _validate_result_set(
        first_results,
        items,
        require_combined=True,
        expected_model=HY3_MODEL_ALIAS,
        expected_effort=HY3_REASONING_EFFORT,
    )
    repeat = _validate_result_set(
        repeat_results,
        items,
        require_combined=False,
        expected_model=HY3_MODEL_ALIAS,
        expected_effort=HY3_REASONING_EFFORT,
    )
    if (
        first["judge_fingerprint"]["sha256"]
        != repeat["judge_fingerprint"]["sha256"]
    ):
        raise ValueError(f"Hy3 first and repeat Judge fingerprints differ: {source_experiment}")
    return {
        "experiment_dir": _relative(source_experiment, experiment_dir),
        "experiment_sha256": sha256_file(experiment_config),
        "trace_manifest": _relative(trace_manifest, experiment_dir),
        "trace_manifest_sha256": sha256_file(trace_manifest),
        "trace_count": len(items),
        "hy3_first_results": _relative(first_results, experiment_dir),
        "hy3_first": first,
        "hy3_repeat_results": _relative(repeat_results, experiment_dir),
        "hy3_repeat": repeat,
        "hy3_stability_summary": _relative(stability_summary, experiment_dir),
        "hy3_stability_summary_sha256": sha256_file(stability_summary),
    }


def prepare_comparison(
    experiment_dir: Path,
    *,
    baseline_dir: Path,
    single_shot_dir: Path,
    candidate: CandidateJudgeSpec = LUNA_CANDIDATE,
) -> dict[str, Any]:
    """Freeze source artifacts and exact Judge settings without making API calls."""

    experiment_dir = experiment_dir.resolve()
    sources = {
        "baseline": _source_descriptor(experiment_dir, baseline_dir),
        "single_shot": _source_descriptor(experiment_dir, single_shot_dir),
    }
    baseline_items = _manifest_items(
        _resolve(experiment_dir, sources["baseline"]["trace_manifest"])
    )
    single_shot_items = _manifest_items(
        _resolve(experiment_dir, sources["single_shot"]["trace_manifest"])
    )
    if set(baseline_items) != set(single_shot_items):
        raise ValueError("Baseline and single-shot manifests do not contain the same task ids.")

    baseline_config = load_json((baseline_dir.resolve() / "experiment.json"))
    single_shot_config = load_json((single_shot_dir.resolve() / "experiment.json"))
    baseline_rubric = _resolve(baseline_dir.resolve(), baseline_config["rubric"])
    single_shot_rubric = _resolve(single_shot_dir.resolve(), single_shot_config["rubric"])
    rubric_sha256 = sha256_file(baseline_rubric)
    if rubric_sha256 != sha256_file(single_shot_rubric):
        raise ValueError("Source experiments use different Rubrics.")

    settings = get_settings()
    payload = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "experiment_id": experiment_dir.name,
        "design": "paired_generation_workflows_cross_judge_repeatability",
        "expected_trace_count_per_source": EXPECTED_TRACE_COUNT,
        "excluded_validation_sets": ["discrimination"],
        "rubric": _relative(baseline_rubric, experiment_dir),
        "rubric_sha256": rubric_sha256,
        "sources": sources,
        "judges": {
            "hy3": {
                "model_id": HY3_MODEL_ALIAS,
                "display_name": HY3_MODEL_ALIAS,
                "reasoning_effort": HY3_REASONING_EFFORT,
                "result_source": "existing",
            },
            candidate.key: candidate.config(),
        },
        "judge_sampling": {"temperature": 0.0, "top_p": 1.0},
        "judge_passes": 2,
        "canonical_pass": 1,
        "concurrency": {"judge": JUDGE_CONCURRENCY},
        "provider": {
            "endpoint_sha256": hashlib.sha256(
                settings.hy3.openai_base_url.encode("utf-8")
            ).hexdigest(),
            "credential_source": "HY3_API_KEY",
        },
    }
    path = experiment_dir / "experiment.json"
    if path.exists():
        if load_json(path) != payload:
            raise ValueError(
                f"Prepared comparison differs from existing experiment: {path}"
            )
    else:
        write_json(path, payload)
    return payload


def _validate_locked_sources(
    experiment_dir: Path,
    config: dict[str, Any],
    *,
    validate_endpoint: bool,
) -> dict[str, dict[str, dict[str, Any]]]:
    if config.get("schema_version") != COMPARISON_SCHEMA_VERSION:
        raise ValueError("Unsupported Judge comparison schema version.")
    if config.get("concurrency", {}).get("judge") != JUDGE_CONCURRENCY:
        raise ValueError("Judge comparison concurrency must remain 512.")
    _candidate_spec(config)
    if sha256_file(_resolve(experiment_dir, config["rubric"])) != config[
        "rubric_sha256"
    ]:
        raise ValueError("Comparison Rubric hash changed.")
    if validate_endpoint:
        current = hashlib.sha256(
            get_settings().hy3.openai_base_url.encode("utf-8")
        ).hexdigest()
        if current != config.get("provider", {}).get("endpoint_sha256"):
            raise ValueError("Judge provider endpoint changed after preparation.")

    manifests: dict[str, dict[str, dict[str, Any]]] = {}
    for source_name, source in config.get("sources", {}).items():
        experiment_path = _resolve(experiment_dir, source["experiment_dir"])
        if sha256_file(experiment_path / "experiment.json") != source[
            "experiment_sha256"
        ]:
            raise ValueError(f"Source experiment config changed: {source_name}")
        manifest_path = _resolve(experiment_dir, source["trace_manifest"])
        if sha256_file(manifest_path) != source["trace_manifest_sha256"]:
            raise ValueError(f"Source trace manifest changed: {source_name}")
        items = _manifest_items(manifest_path)
        if len(items) != EXPECTED_TRACE_COUNT:
            raise ValueError(f"Source trace count changed: {source_name}")
        manifests[source_name] = items
        for pass_name in ("first", "repeat"):
            results_dir = _resolve(
                experiment_dir, source[f"hy3_{pass_name}_results"]
            )
            locked = source[f"hy3_{pass_name}"]
            filename = "hy3_judge.json"
            if sha256_file(results_dir / "manifest.json") != locked["manifest_sha256"]:
                raise ValueError(f"Hy3 {pass_name} manifest changed: {source_name}")
            if sha256_file(results_dir / "summary.json") != locked["summary_sha256"]:
                raise ValueError(f"Hy3 {pass_name} summary changed: {source_name}")
            if _tree_sha256(results_dir, filename) != locked["judge_records_sha256"]:
                raise ValueError(f"Hy3 {pass_name} Judge records changed: {source_name}")
            combined_sha = locked.get("combined_records_sha256")
            if combined_sha is not None and _tree_sha256(
                results_dir, "combined.json"
            ) != combined_sha:
                raise ValueError(f"Hy3 combined records changed: {source_name}")
        stability_path = _resolve(
            experiment_dir, source["hy3_stability_summary"]
        )
        if sha256_file(stability_path) != source["hy3_stability_summary_sha256"]:
            raise ValueError(f"Hy3 stability summary changed: {source_name}")
    if set(manifests) != set(SOURCE_LABELS):
        raise ValueError("Comparison must contain baseline and single_shot sources.")
    if set(manifests["baseline"]) != set(manifests["single_shot"]):
        raise ValueError("Source task pairing changed.")
    return manifests


def _load_experiment(
    experiment_dir: Path,
    *,
    validate_endpoint: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    experiment_dir = experiment_dir.resolve()
    config = load_json(experiment_dir / "experiment.json")
    manifests = _validate_locked_sources(
        experiment_dir, config, validate_endpoint=validate_endpoint
    )
    return config, manifests


def _run_command(arguments: Sequence[str]) -> int:
    return subprocess.run(arguments, cwd=PROJECT_ROOT, check=False).returncode


def _candidate_results(
    experiment_dir: Path,
    source_name: str,
    pass_number: int,
    candidate: CandidateJudgeSpec,
) -> Path:
    return (
        experiment_dir
        / "results"
        / source_name
        / candidate.key
        / f"pass-{pass_number:03d}"
    )


def _score_source(
    experiment_dir: Path,
    config: dict[str, Any],
    source_name: str,
    *,
    pass_number: int,
) -> None:
    source = config["sources"][source_name]
    candidate = _candidate_spec(config)
    evaluators = "rules,judge" if pass_number == 1 else "judge"
    arguments = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_evaluation.py"),
        "score",
        "--trace-manifest",
        str(_resolve(experiment_dir, source["trace_manifest"])),
        "--rubric",
        str(_resolve(experiment_dir, config["rubric"])),
        "--evaluators",
        evaluators,
        "--output-dir",
        str(_candidate_results(experiment_dir, source_name, pass_number, candidate)),
        "--concurrency",
        str(JUDGE_CONCURRENCY),
        "--judge-model-id",
        candidate.model_id,
        "--reasoning-effort",
        candidate.reasoning_effort,
    ]
    if _run_command(arguments):
        raise RuntimeError(
            f"{candidate.display_name} pass {pass_number} is incomplete for "
            f"{source_name}; rerun to resume."
        )


def score_comparison(experiment_dir: Path) -> None:
    """Run the canonical candidate rules+Judge pass for both workflows."""

    experiment_dir = experiment_dir.resolve()
    config, manifests = _load_experiment(experiment_dir, validate_endpoint=True)
    candidate = _candidate_spec(config)
    for source_name in SOURCE_LABELS:
        _score_source(experiment_dir, config, source_name, pass_number=1)
        _validate_result_set(
            _candidate_results(experiment_dir, source_name, 1, candidate),
            manifests[source_name],
            require_combined=True,
            expected_model=candidate.model_id,
            expected_effort=candidate.reasoning_effort,
        )


def repeat_comparison(experiment_dir: Path) -> None:
    """Run the second candidate Judge-only pass for both workflows."""

    experiment_dir = experiment_dir.resolve()
    config, manifests = _load_experiment(experiment_dir, validate_endpoint=True)
    candidate = _candidate_spec(config)
    for source_name in SOURCE_LABELS:
        first = _validate_result_set(
            _candidate_results(experiment_dir, source_name, 1, candidate),
            manifests[source_name],
            require_combined=True,
            expected_model=candidate.model_id,
            expected_effort=candidate.reasoning_effort,
        )
        _score_source(experiment_dir, config, source_name, pass_number=2)
        second = _validate_result_set(
            _candidate_results(experiment_dir, source_name, 2, candidate),
            manifests[source_name],
            require_combined=False,
            expected_model=candidate.model_id,
            expected_effort=candidate.reasoning_effort,
        )
        if (
            first["judge_fingerprint"]["sha256"]
            != second["judge_fingerprint"]["sha256"]
        ):
            raise RuntimeError(
                f"{candidate.display_name} Judge fingerprint changed between passes: "
                f"{source_name}"
            )


def _scores(record: dict[str, Any]) -> dict[str, int]:
    return {
        item["dimension_id"]: item["score"]
        for item in record.get("dimension_scores", [])
        if isinstance(item, dict)
        and isinstance(item.get("dimension_id"), str)
        and isinstance(item.get("score"), int)
        and not isinstance(item.get("score"), bool)
    }


def _combined_rows(
    results_dir: Path,
    manifest: dict[str, dict[str, Any]],
    dimensions: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in results_dir.glob("items/*/combined.json"):
        payload = load_json(path)
        run_id = payload.get("run_id") if isinstance(payload, dict) else None
        if isinstance(run_id, str):
            records[run_id] = payload
    rows: dict[str, dict[str, Any]] = {}
    for task_id, task in manifest.items():
        record = records.get(task["run_id"])
        if record is None or record.get("trace_sha256") != task["trace_sha256"]:
            raise ValueError(f"Missing or mismatched combined result for {task_id}")
        scores = _scores(record)
        if set(scores) != set(dimensions):
            raise ValueError(f"Combined dimension coverage differs for {task_id}")
        rows[task_id] = {
            "task_id": task_id,
            "run_id": task["run_id"],
            "topic": task.get("topic"),
            "target_length": task.get("target_length"),
            "domain": task.get("domain"),
            "challenge_tags": "|".join(task.get("challenge_tags", [])),
            "trace_sha256": task["trace_sha256"],
            "gate_failed": bool(record.get("gate_failed")),
            "final_score": record.get("metrics", {}).get("final_score"),
            **scores,
        }
    return rows


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _quality_summary(
    rows: dict[str, dict[str, Any]], dimensions: tuple[str, ...]
) -> dict[str, Any]:
    final_scores = [
        float(row["final_score"])
        for row in rows.values()
        if isinstance(row.get("final_score"), (int, float))
        and not isinstance(row.get("final_score"), bool)
    ]
    return {
        "count": len(rows),
        "scored": len(final_scores),
        "gate_failed": sum(bool(row["gate_failed"]) for row in rows.values()),
        "final_score_mean": _mean(final_scores),
        "dimensions": {
            dimension: _mean([float(row[dimension]) for row in rows.values()])
            for dimension in dimensions
        },
    }


def _paired_workflows(
    baseline: dict[str, dict[str, Any]],
    single_shot: dict[str, dict[str, Any]],
    dimensions: tuple[str, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if set(baseline) != set(single_shot) or len(baseline) != EXPECTED_TRACE_COUNT:
        raise ValueError("Workflow comparison requires 300 exact task pairs.")
    rows: list[dict[str, Any]] = []
    for task_id in sorted(baseline):
        first = baseline[task_id]
        second = single_shot[task_id]
        first_score = first["final_score"]
        second_score = second["final_score"]
        evaluable = all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in (first_score, second_score)
        )
        row = {
            "task_id": task_id,
            "topic": first["topic"],
            "target_length": first["target_length"],
            "domain": first["domain"],
            "challenge_tags": first["challenge_tags"],
            "baseline_run_id": first["run_id"],
            "single_shot_run_id": second["run_id"],
            "baseline_final_score": first_score,
            "single_shot_final_score": second_score,
            "final_score_delta": (
                float(second_score) - float(first_score) if evaluable else None
            ),
            "baseline_gate_failed": first["gate_failed"],
            "single_shot_gate_failed": second["gate_failed"],
        }
        for dimension in dimensions:
            row[f"baseline_{dimension}"] = first[dimension]
            row[f"single_shot_{dimension}"] = second[dimension]
            row[f"delta_{dimension}"] = second[dimension] - first[dimension]
        rows.append(row)
    evaluable_rows = [row for row in rows if row["final_score_delta"] is not None]
    deltas = [float(row["final_score_delta"]) for row in evaluable_rows]
    ordered = sorted(deltas)
    median = (
        ordered[len(ordered) // 2]
        if len(ordered) % 2
        else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
    ) if ordered else None
    summary = {
        "pair_count": len(rows),
        "evaluable_pair_count": len(evaluable_rows),
        "wins": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "losses": sum(delta < 0 for delta in deltas),
        "mean_delta": _mean(deltas),
        "median_delta": median,
        "baseline_quality": _quality_summary(baseline, dimensions),
        "single_shot_quality": _quality_summary(single_shot, dimensions),
        "dimensions": {
            dimension: {
                "mean_delta": _mean(
                    [float(row[f"delta_{dimension}"]) for row in rows]
                ),
                "improved": sum(row[f"delta_{dimension}"] > 0 for row in rows),
                "unchanged": sum(row[f"delta_{dimension}"] == 0 for row in rows),
                "declined": sum(row[f"delta_{dimension}"] < 0 for row in rows),
            }
            for dimension in dimensions
        },
    }
    return summary, rows


def _cross_judge(
    hy3_dir: Path,
    candidate_dir: Path,
    manifest: dict[str, dict[str, Any]],
    judge_dimensions: tuple[str, ...],
    *,
    candidate_key: str = "luna",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hy3 = _judge_records(hy3_dir)
    candidate = _judge_records(candidate_dir)
    expected_runs = {item["run_id"] for item in manifest.values()}
    if set(hy3) != expected_runs or set(candidate) != expected_runs:
        raise ValueError("Cross-Judge comparison does not cover the same frozen traces.")
    rows: list[dict[str, Any]] = []
    hy3_totals: list[float] = []
    candidate_totals: list[float] = []
    all_exact = 0
    for task_id in sorted(manifest):
        task = manifest[task_id]
        first = hy3[task["run_id"]]
        second = candidate[task["run_id"]]
        if first.get("trace_sha256") != second.get("trace_sha256"):
            raise ValueError(f"Cross-Judge trace hash differs for {task_id}")
        first_scores = _scores(first)
        second_scores = _scores(second)
        if set(first_scores) != set(judge_dimensions) or set(second_scores) != set(
            judge_dimensions
        ):
            raise ValueError(f"Cross-Judge dimension coverage differs for {task_id}")
        first_total = first.get("metrics", {}).get("normalized_score")
        second_total = second.get("metrics", {}).get("normalized_score")
        if not isinstance(first_total, (int, float)) or not isinstance(
            second_total, (int, float)
        ):
            raise ValueError(f"Cross-Judge score is missing for {task_id}")
        changed = [
            dimension
            for dimension in judge_dimensions
            if first_scores[dimension] != second_scores[dimension]
        ]
        all_exact += not changed
        hy3_totals.append(float(first_total))
        candidate_totals.append(float(second_total))
        row = {
            "task_id": task_id,
            "run_id": task["run_id"],
            "topic": task.get("topic"),
            "target_length": task.get("target_length"),
            "trace_sha256": task["trace_sha256"],
            "hy3_normalized_score": float(first_total),
            f"{candidate_key}_normalized_score": float(second_total),
            "normalized_score_delta": float(second_total) - float(first_total),
            "changed_dimension_count": len(changed),
            "changed_dimensions": "|".join(changed),
            "hy3_gate_failed": bool(first.get("gate_failed")),
            f"{candidate_key}_gate_failed": bool(second.get("gate_failed")),
        }
        for dimension in judge_dimensions:
            row[f"hy3_{dimension}"] = first_scores[dimension]
            row[f"{candidate_key}_{dimension}"] = second_scores[dimension]
            row[f"delta_{dimension}"] = (
                second_scores[dimension] - first_scores[dimension]
            )
        rows.append(row)
    dimensions: dict[str, Any] = {}
    exact_total = 0
    for dimension in judge_dimensions:
        first_values = [_scores(hy3[item["run_id"]])[dimension] for item in manifest.values()]
        second_values = [
            _scores(candidate[item["run_id"]])[dimension]
            for item in manifest.values()
        ]
        exact = sum(left == right for left, right in zip(first_values, second_values))
        exact_total += exact
        dimensions[dimension] = {
            "exact_agreement_count": exact,
            "exact_agreement_rate": exact / len(first_values),
            "quadratic_weighted_kappa": quadratic_weighted_kappa(
                first_values, second_values
            ),
            "mae": sum(
                abs(left - right) for left, right in zip(first_values, second_values)
            ) / len(first_values),
            "hy3_distribution": dict(sorted(Counter(first_values).items())),
            f"{candidate_key}_distribution": dict(
                sorted(Counter(second_values).items())
            ),
        }
    summary = {
        "record_count": len(rows),
        "dimension_count": len(judge_dimensions),
        "hy3_fingerprint_sha256": _judge_fingerprint(next(iter(hy3.values())))[
            "sha256"
        ],
        f"{candidate_key}_fingerprint_sha256": _judge_fingerprint(
            next(iter(candidate.values()))
        )["sha256"],
        "dimensions": dimensions,
        "overall": {
            "dimension_exact_agreement_rate": exact_total
            / (len(rows) * len(judge_dimensions)),
            "all_dimensions_exact_count": all_exact,
            "all_dimensions_exact_rate": all_exact / len(rows),
            "hy3_normalized_score_mean": _mean(hy3_totals),
            f"{candidate_key}_normalized_score_mean": _mean(candidate_totals),
            "normalized_score_mean_delta": _mean(
                [right - left for left, right in zip(hy3_totals, candidate_totals)]
            ),
            "normalized_score_mae": _mean(
                [
                    abs(right - left)
                    for left, right in zip(hy3_totals, candidate_totals)
                ]
            ),
            "normalized_score_spearman": spearman(
                hy3_totals, candidate_totals
            ),
            "gate_disagreement_count": sum(
                row["hy3_gate_failed"] != row[f"{candidate_key}_gate_failed"]
                for row in rows
            ),
        },
    }
    return summary, rows


def _csv_text(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _sign(value: float | None) -> int | None:
    if value is None:
        return None
    return 1 if value > 0 else -1 if value < 0 else 0


def _main_shortfalls(
    workflow_summary: dict[str, Any], *, limit: int = 3
) -> list[str]:
    """Return the largest negative workflow deltas in deterministic order."""

    negative = [
        (dimension, float(item["mean_delta"]))
        for dimension, item in workflow_summary["dimensions"].items()
        if item.get("mean_delta") is not None and float(item["mean_delta"]) < 0
    ]
    negative.sort(key=lambda item: (item[1], item[0]))
    return [dimension for dimension, _ in negative[:limit]]


def export_comparison_report(experiment_dir: Path) -> dict[str, Any]:
    """Export canonical workflow, cross-Judge, and repeatability comparisons."""

    experiment_dir = experiment_dir.resolve()
    config, manifests = _load_experiment(experiment_dir, validate_endpoint=False)
    candidate = _candidate_spec(config)
    rubric = load_rubric(_resolve(experiment_dir, config["rubric"]))
    dimensions = rubric.dimension_ids
    judge_dimensions = rubric.judge_dimension_ids
    report_dir = experiment_dir / "report"

    workflow_summaries: dict[str, Any] = {}
    workflow_rows: dict[str, list[dict[str, Any]]] = {}
    cross_summaries: dict[str, Any] = {}
    cross_rows: dict[str, list[dict[str, Any]]] = {}
    stability: dict[str, dict[str, Any]] = {"hy3": {}, candidate.key: {}}
    stability_inputs: dict[
        str,
        dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
    ] = {"hy3": {}, candidate.key: {}}

    for source_name, source in config["sources"].items():
        candidate_first_dir = _candidate_results(
            experiment_dir, source_name, 1, candidate
        )
        candidate_second_dir = _candidate_results(
            experiment_dir, source_name, 2, candidate
        )
        _validate_result_set(
            candidate_first_dir,
            manifests[source_name],
            require_combined=True,
            expected_model=candidate.model_id,
            expected_effort=candidate.reasoning_effort,
        )
        _validate_result_set(
            candidate_second_dir,
            manifests[source_name],
            require_combined=False,
            expected_model=candidate.model_id,
            expected_effort=candidate.reasoning_effort,
        )
        hy3_first_dir = _resolve(experiment_dir, source["hy3_first_results"])
        hy3_repeat_dir = _resolve(experiment_dir, source["hy3_repeat_results"])
        cross_summaries[source_name], cross_rows[source_name] = _cross_judge(
            hy3_first_dir,
            candidate_first_dir,
            manifests[source_name],
            judge_dimensions,
            candidate_key=candidate.key,
        )
        hy3_stability, hy3_stability_rows = compare_judge_runs(
            hy3_first_dir,
            hy3_repeat_dir,
            trace_manifest=_resolve(experiment_dir, source["trace_manifest"]),
        )
        stability["hy3"][source_name] = hy3_stability
        stability_inputs["hy3"][source_name] = (
            hy3_stability,
            hy3_stability_rows,
        )
        candidate_stability, candidate_stability_rows = compare_judge_runs(
            candidate_first_dir,
            candidate_second_dir,
            trace_manifest=_resolve(experiment_dir, source["trace_manifest"]),
        )
        stability[candidate.key][source_name] = candidate_stability
        stability_inputs[candidate.key][source_name] = (
            candidate_stability,
            candidate_stability_rows,
        )
        export_judge_stability(
            candidate_first_dir,
            candidate_second_dir,
            report_dir / "stability" / f"{candidate.key}-{source_name}",
            trace_manifest=_resolve(experiment_dir, source["trace_manifest"]),
        )

    for judge_name, comparisons in stability_inputs.items():
        combined_summary, combined_rows = combine_judge_comparisons(comparisons)
        stability[judge_name]["combined"] = combined_summary
        combined_dir = report_dir / "stability" / f"{judge_name}-combined"
        write_json(combined_dir / "summary.json", combined_summary)
        atomic_write_text(
            combined_dir / "full_comparison.csv", _csv_text(combined_rows)
        )

    for judge_name in ("hy3", candidate.key):
        if judge_name == "hy3":
            baseline_dir = _resolve(
                experiment_dir, config["sources"]["baseline"]["hy3_first_results"]
            )
            single_dir = _resolve(
                experiment_dir,
                config["sources"]["single_shot"]["hy3_first_results"],
            )
        else:
            baseline_dir = _candidate_results(
                experiment_dir, "baseline", 1, candidate
            )
            single_dir = _candidate_results(
                experiment_dir, "single_shot", 1, candidate
            )
        baseline_rows = _combined_rows(
            baseline_dir, manifests["baseline"], dimensions
        )
        single_rows = _combined_rows(
            single_dir, manifests["single_shot"], dimensions
        )
        workflow_summaries[judge_name], workflow_rows[judge_name] = _paired_workflows(
            baseline_rows, single_rows, dimensions
        )

    robustness = {
        "mean_delta_sign_consistent": _sign(workflow_summaries["hy3"]["mean_delta"])
        == _sign(workflow_summaries[candidate.key]["mean_delta"]),
        "dimensions": {
            dimension: {
                "hy3_mean_delta": workflow_summaries["hy3"]["dimensions"][
                    dimension
                ]["mean_delta"],
                f"{candidate.key}_mean_delta": workflow_summaries[candidate.key][
                    "dimensions"
                ][dimension]["mean_delta"],
                "delta_of_deltas": workflow_summaries[candidate.key]["dimensions"]
                [dimension]["mean_delta"]
                - workflow_summaries["hy3"]["dimensions"][dimension]["mean_delta"],
                "sign_consistent": _sign(
                    workflow_summaries["hy3"]["dimensions"][dimension]["mean_delta"]
                )
                == _sign(
                    workflow_summaries[candidate.key]["dimensions"][dimension][
                        "mean_delta"
                    ]
                ),
            }
            for dimension in dimensions
        },
    }
    hy3_shortfalls = _main_shortfalls(workflow_summaries["hy3"])
    candidate_shortfalls = _main_shortfalls(workflow_summaries[candidate.key])
    robustness["main_shortfalls"] = {
        "method": "up_to_three_most_negative_mean_dimension_deltas",
        "hy3": hy3_shortfalls,
        candidate.key: candidate_shortfalls,
        "common": [
            dimension
            for dimension in hy3_shortfalls
            if dimension in candidate_shortfalls
        ],
        "same_set": set(hy3_shortfalls) == set(candidate_shortfalls),
    }
    summary = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "canonical_pass": 1,
        f"expected_{candidate.key}_record_count": 1200,
        "concurrency": config["concurrency"],
        "judges": config["judges"],
        "workflow_comparison": workflow_summaries,
        "cross_judge_agreement": cross_summaries,
        "stability": stability,
        "conclusion_robustness": robustness,
    }
    write_json(report_dir / "comparison_summary.json", summary)
    atomic_write_text(
        report_dir / f"{candidate.key}_paired_results.csv",
        _csv_text(workflow_rows[candidate.key]),
    )
    atomic_write_text(
        report_dir / "hy3_paired_results.csv", _csv_text(workflow_rows["hy3"])
    )
    for source_name, rows in cross_rows.items():
        atomic_write_text(
            report_dir / f"hy3_vs_{candidate.key}_{source_name}.csv", _csv_text(rows)
        )

    dimension_labels = {
        dimension.dimension_id: dimension.name for dimension in rubric.dimensions
    }
    candidate_label = REPORT_DISPLAY_NAMES.get(candidate.key, candidate.display_name)
    lines = [
        f"# Hy3 与 {candidate_label} 双 Judge 对比实验",
        "",
        "## 配置与完整性",
        "",
        f"- 冻结文案：{EXPECTED_TRACE_COUNT} 条三候选主编基线 + "
        f"{EXPECTED_TRACE_COUNT} 条端到端直接生成。",
        f"- Hy3：model_id=`hy3`，推理强度=`high`（该模型最高档）。",
        f"- {candidate_label}：model_id=`{candidate.model_id}`，"
        f"推理强度=`{candidate.reasoning_effort}`"
        "（该模型最高档）。",
        f"- {candidate_label} 新增评分：2 组 × 2 轮 × "
        f"{EXPECTED_TRACE_COUNT} = 1,200 条；"
        f"Judge 并发 {JUDGE_CONCURRENCY}。",
        "- 两个模型使用相同冻结输入、Rubric、Judge Prompt 与采样参数；"
        "推理强度按各模型最高支持档匹配，不解释为同名档位。",
        "",
        "## 首轮流程质量结论",
        "",
        "| Judge | 基线均分 | 单次生成均分 | 平均差 | 胜/平/负 | 门控失败（基线/单次） |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, display in (
        ("hy3", HY3_MODEL_ALIAS),
        (candidate.key, candidate_label),
    ):
        item = workflow_summaries[name]
        lines.append(
            f"| {display} | {item['baseline_quality']['final_score_mean']:.6f} | "
            f"{item['single_shot_quality']['final_score_mean']:.6f} | "
            f"{item['mean_delta']:+.6f} | {item['wins']}/{item['ties']}/{item['losses']} | "
            f"{item['baseline_quality']['gate_failed']}/"
            f"{item['single_shot_quality']['gate_failed']} |"
        )
    lines.extend(
        [
            "",
            "### 分维度结论稳健性",
            "",
            f"| 维度 | Hy3 平均差 | {candidate_label} 平均差 | "
            "差中差 | 方向一致 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for dimension in dimensions:
        item = robustness["dimensions"][dimension]
        lines.append(
            f"| {dimension_labels[dimension]} | {item['hy3_mean_delta']:+.4f} | "
            f"{item[f'{candidate.key}_mean_delta']:+.4f} | "
            f"{item['delta_of_deltas']:+.4f} | "
            f"{'是' if item['sign_consistent'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 首轮跨 Judge 一致性",
            "",
            "| 冻结文案组 | 逐维一致率 | 全维一致率 | Judge 总分 MAE | Spearman | 门控分歧 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for source_name, label in SOURCE_LABELS.items():
        overall = cross_summaries[source_name]["overall"]
        lines.append(
            f"| {label} | {overall['dimension_exact_agreement_rate']:.4f} | "
            f"{overall['all_dimensions_exact_rate']:.4f} | "
            f"{overall['normalized_score_mae']:.6f} | "
            f"{overall['normalized_score_spearman']} | "
            f"{overall['gate_disagreement_count']} |"
        )
    lines.extend(
        [
            "",
            "### 跨 Judge 分维度一致性",
            "",
            "二次加权 Kappa 按 1–3 档有序分数计算；当一侧评分无方差时，"
            "Kappa 可能为 0，即使原始一致率很高。",
        ]
    )
    for source_name, label in SOURCE_LABELS.items():
        lines.extend(
            [
                "",
                f"#### {label}",
                "",
                "| 维度 | 一致率 | 二次加权 Kappa | MAE |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for dimension in judge_dimensions:
            item = cross_summaries[source_name]["dimensions"][dimension]
            lines.append(
                f"| {dimension_labels[dimension]} | "
                f"{item['exact_agreement_rate']:.4f} | "
                f"{item['quadratic_weighted_kappa']:.4f} | "
                f"{item['mae']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## 两轮内部稳定性",
            "",
            "主结论将三候选主编基线与端到端直接生成合并为同一个 600 条样本集后计算；",
            "分组结果只用于诊断。合并 Spearman 由 600 对总分直接计算，不是两个分组相关系数的平均。",
            "",
            "| Judge | 冻结文案组 | 逐维一致率 | 全维一致率 | 总分 MAE | Spearman |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for judge_name, display in (
        ("hy3", HY3_MODEL_ALIAS),
        (candidate.key, candidate_label),
    ):
        overall = stability[judge_name]["combined"]["overall"]
        lines.append(
            f"| {display} | 两组合并（600 条） | "
            f"{overall['dimension_exact_agreement_rate']:.4f} | "
            f"{overall['all_dimensions_exact_rate']:.4f} | "
            f"{overall['normalized_score_mae']:.6f} | "
            f"{overall['normalized_score_spearman']} |"
        )
    lines.extend(
        [
            "",
            "### 分组诊断",
            "",
            "| Judge | 冻结文案组 | 逐维一致率 | 全维一致率 | 总分 MAE | Spearman |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for judge_name, display in (
        ("hy3", HY3_MODEL_ALIAS),
        (candidate.key, candidate_label),
    ):
        for source_name, label in SOURCE_LABELS.items():
            overall = stability[judge_name][source_name]["overall"]
            lines.append(
                f"| {display} | {label} | "
                f"{overall['dimension_exact_agreement_rate']:.4f} | "
                f"{overall['all_dimensions_exact_rate']:.4f} | "
                f"{overall['normalized_score_mae']:.6f} | "
                f"{overall['normalized_score_spearman']} |"
            )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            f"首轮平均差方向{'一致' if robustness['mean_delta_sign_consistent'] else '不一致'}。"
            f"主要短板集合{'一致' if robustness['main_shortfalls']['same_set'] else '不完全一致'}："
            "这里将相对基线平均差为负的维度按降幅排序，最多取前三项；"
            f"Hy3 为{'、'.join(dimension_labels[item] for item in hy3_shortfalls) or '无'}，"
            f"{candidate_label} 为"
            f"{'、'.join(dimension_labels[item] for item in candidate_shortfalls) or '无'}，"
            f"共同项为{'、'.join(dimension_labels[item] for item in robustness['main_shortfalls']['common']) or '无'}。"
            "本报告比较的是两个 Judge 在各自最高推理强度下，对同一批冻结文案的评价；"
            "第二轮只用于稳定性，不与首轮取均值。",
            "",
        ]
    )
    atomic_write_text(report_dir / "comparison.md", "\n".join(lines))
    return summary


__all__ = [
    "CandidateJudgeSpec",
    "COMPARISON_SCHEMA_VERSION",
    "EXPECTED_TRACE_COUNT",
    "GLM_CANDIDATE",
    "GLM_MODEL_ALIAS",
    "GLM_MODEL_ID",
    "GLM_REASONING_EFFORT",
    "JUDGE_CONCURRENCY",
    "LUNA_CANDIDATE",
    "LUNA_MODEL_ALIAS",
    "LUNA_MODEL_ID",
    "LUNA_REASONING_EFFORT",
    "export_comparison_report",
    "prepare_comparison",
    "repeat_comparison",
    "score_comparison",
]
