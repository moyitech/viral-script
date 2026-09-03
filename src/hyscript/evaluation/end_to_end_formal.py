"""Formal frozen-research single-shot generation experiment and reporting."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Iterable, Sequence

from hyscript.config import PROJECT_ROOT, get_settings
from hyscript.llm.prompts import (
    BACKGROUND_SCRIPT_FORMAT_REPAIR_PROMPT_VERSION,
    BACKGROUND_SCRIPT_FORMAT_REPAIR_SYSTEM_PROMPT,
    BACKGROUND_SCRIPT_GENERATION_PROMPT_VERSION,
    BACKGROUND_SCRIPT_GENERATION_SYSTEM_PROMPT,
    BACKGROUND_SELECTION_VERSION,
    RESEARCH_QUERY_PLAN_PROMPT_VERSION,
    RESEARCH_QUERY_PLAN_SYSTEM_PROMPT,
)

from .formal import (
    atomic_write_text,
    load_json,
    lock_runtime,
    sha256_file,
    write_json,
)
from .io import TraceInputError, load_frozen_trace
from .stability import export_judge_stability


E2E_SCHEMA_VERSION = "1.0"
EXPECTED_TOPIC_COUNT = 100
TARGET_LENGTHS = (280, 450, 700)
EXPECTED_TRACE_COUNT = EXPECTED_TOPIC_COUNT * len(TARGET_LENGTHS)
SOURCE_RESEARCH_TARGET_LENGTH = 450
DEFAULT_TASK_CONCURRENCY = 300
DEFAULT_HY3_CONCURRENCY = 512
DEFAULT_SEARCH_CONCURRENCY = 8
DEFAULT_JUDGE_CONCURRENCY = 512
MAX_HY3_CONCURRENCY = 512
EXPECTED_GENERATION_MODE = "single_shot"
_DIMENSIONS = (
    "topic_alignment",
    "length_compliance",
    "theme_information",
    "engagement",
    "oral_fluency",
    "rhetoric_memorability",
    "logic_structure",
    "safety_compliance",
)
_DIMENSION_LABELS = {
    "topic_alignment": "选题匹配度",
    "length_compliance": "字数符合度",
    "theme_information": "主题明确与信息量",
    "engagement": "吸引力",
    "oral_fluency": "口播流畅度",
    "rhetoric_memorability": "修辞与记忆点",
    "logic_structure": "语言逻辑与结构",
    "safety_compliance": "合规性",
}
_DOMAIN_LABELS = {
    "consumer_society": "消费与社会议题",
    "education": "教育",
    "environment_energy": "环境与能源",
    "finance": "金融",
    "health": "健康",
    "public_services": "公共服务",
    "technology": "科技",
    "workplace": "职场",
}
_CHALLENGE_LABELS = {
    "conflicting_interests": "利益冲突",
    "open_ended_tradeoff": "开放式权衡",
    "safety_or_compliance": "安全或合规",
    "time_sensitive": "时效性",
    "vulnerable_groups": "弱势群体",
}


def _json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _relative(path: Path, base: Path) -> str:
    resolved_path = path.resolve()
    try:
        return os.path.relpath(resolved_path, base.resolve())
    except ValueError:
        # Windows cannot express a relative path between different drives.
        return str(resolved_path)


def _write_stable_json(path: Path, payload: Any) -> None:
    content = _json_text(payload)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Prepared end-to-end input differs from existing file: {path}")
        return
    atomic_write_text(path, content)


def _attempt_number(parent: Path) -> int:
    values: list[int] = []
    if parent.exists():
        for path in parent.glob("attempt-*"):
            try:
                values.append(int(path.name.removeprefix("attempt-")))
            except ValueError:
                continue
    return max(values, default=0) + 1


def _prompt_versions() -> dict[str, str]:
    return {
        "research_query_plan": RESEARCH_QUERY_PLAN_PROMPT_VERSION,
        "background_selection": BACKGROUND_SELECTION_VERSION,
        "script_generation": BACKGROUND_SCRIPT_GENERATION_PROMPT_VERSION,
        "script_format_repair": BACKGROUND_SCRIPT_FORMAT_REPAIR_PROMPT_VERSION,
    }


def _prompt_hashes() -> dict[str, str]:
    return {
        "research_query_plan": _text_sha256(RESEARCH_QUERY_PLAN_SYSTEM_PROMPT),
        "script_generation": _text_sha256(BACKGROUND_SCRIPT_GENERATION_SYSTEM_PROMPT),
        "script_format_repair": _text_sha256(
            BACKGROUND_SCRIPT_FORMAT_REPAIR_SYSTEM_PROMPT
        ),
    }


def _validate_baseline_prompts(baseline_dir: Path) -> None:
    expected_versions = _prompt_versions()
    expected_hashes = _prompt_hashes()
    baseline_config = load_json(baseline_dir / "experiment.json")
    baseline_versions = baseline_config.get("prompt_versions", {})
    if baseline_versions.get("research_query_plan") != expected_versions[
        "research_query_plan"
    ]:
        raise ValueError("Current research prompt version differs from the baseline.")

    research_manifests = sorted(
        (baseline_dir / "generation/research").glob("attempt-*/manifest.json")
    )
    if not research_manifests:
        raise ValueError("Baseline research manifests are incomplete.")
    for path in research_manifests:
        manifest = load_json(path)
        actual = manifest.get("system_prompt_sha256", {}).get(
            "research_query_plan"
        )
        if actual != expected_hashes["research_query_plan"]:
            raise ValueError(f"Baseline research prompt hash differs: {path}")


def _source_research_records(
    source_manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    manifest = load_json(source_manifest_path)
    tasks = manifest.get("tasks") if isinstance(manifest, dict) else None
    if (
        not isinstance(tasks, list)
        or manifest.get("selected_count") != EXPECTED_TOPIC_COUNT
        or len(tasks) != EXPECTED_TOPIC_COUNT
    ):
        raise ValueError("Baseline research manifest must select exactly 100 topics.")
    records: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(tasks, start=1):
        expected_id = f"T{index:03d}"
        if (
            not isinstance(raw, dict)
            or raw.get("task_id") != expected_id
            or raw.get("target_length") != SOURCE_RESEARCH_TARGET_LENGTH
            or raw.get("research_status") != "ready"
        ):
            raise ValueError(f"Invalid baseline research record: {expected_id}")
        value = raw.get("research_snapshot")
        expected_hash = raw.get("research_sha256")
        if not isinstance(value, str) or not isinstance(expected_hash, str):
            raise ValueError(f"Missing baseline research snapshot: {expected_id}")
        snapshot = (source_manifest_path.parent / value).resolve()
        if not snapshot.is_file() or sha256_file(snapshot) != expected_hash:
            raise ValueError(f"Baseline research snapshot hash differs: {expected_id}")
        records[expected_id] = {**raw, "resolved_snapshot": str(snapshot)}
    return records


def _validate_matrix(matrix: Any) -> list[dict[str, Any]]:
    if not isinstance(matrix, list) or len(matrix) != EXPECTED_TRACE_COUNT:
        raise ValueError("Baseline task matrix must contain exactly 300 items.")
    expected_ids = [
        f"T{topic:03d}-L{length}"
        for topic in range(1, EXPECTED_TOPIC_COUNT + 1)
        for length in TARGET_LENGTHS
    ]
    actual_ids = [item.get("task_id") if isinstance(item, dict) else None for item in matrix]
    if actual_ids != expected_ids:
        raise ValueError("Baseline task matrix order or task ids changed.")
    for item in matrix:
        if (
            not isinstance(item.get("topic"), str)
            or item.get("target_length") not in TARGET_LENGTHS
            or not isinstance(item.get("dataset_index"), int)
        ):
            raise ValueError("Baseline task matrix contains an invalid item.")
    return [dict(item) for item in matrix]


def prepare_e2e_experiment(
    experiment_dir: Path,
    *,
    baseline_dir: Path,
) -> dict[str, Any]:
    """Freeze deterministic inputs for the paired 300-trace experiment."""

    experiment_dir = experiment_dir.resolve()
    baseline_dir = baseline_dir.resolve()
    _validate_baseline_prompts(baseline_dir)
    baseline = load_json(baseline_dir / "experiment.json")
    matrix = _validate_matrix(load_json(baseline_dir / "task_matrix.json"))
    topics = load_json(baseline_dir / "topics.json")
    if not isinstance(topics, list) or len(topics) != EXPECTED_TOPIC_COUNT:
        raise ValueError("Baseline topic catalog must contain exactly 100 items.")

    dataset_path = (baseline_dir / baseline["dataset"]).resolve()
    rubric_path = (baseline_dir / baseline["rubric"]).resolve()
    baseline_trace_manifest = baseline_dir / "generation/trace_manifest.json"
    source_research_manifest = baseline_dir / "generation/research_manifest.json"
    if sha256_file(dataset_path) != baseline.get("dataset_sha256"):
        raise ValueError("Baseline dataset hash is invalid.")
    if sha256_file(rubric_path) != baseline.get("rubric_sha256"):
        raise ValueError("Baseline rubric hash is invalid.")
    if not baseline_trace_manifest.is_file():
        raise ValueError("Baseline trace manifest is missing.")
    source_research = _source_research_records(source_research_manifest)

    config = {
        "schema_version": E2E_SCHEMA_VERSION,
        "experiment_id": experiment_dir.name,
        "design": "frozen_baseline_research_single_shot_end_to_end",
        "baseline_experiment": _relative(baseline_dir, experiment_dir),
        "baseline_experiment_sha256": sha256_file(baseline_dir / "experiment.json"),
        "baseline_trace_manifest_sha256": sha256_file(baseline_trace_manifest),
        "source_research_manifest": _relative(
            source_research_manifest,
            experiment_dir,
        ),
        "source_research_manifest_sha256": sha256_file(source_research_manifest),
        "dataset": _relative(dataset_path, experiment_dir),
        "dataset_sha256": sha256_file(dataset_path),
        "rubric": _relative(rubric_path, experiment_dir),
        "rubric_sha256": sha256_file(rubric_path),
        "topic_count": EXPECTED_TOPIC_COUNT,
        "target_lengths": list(TARGET_LENGTHS),
        "expected_trace_count": EXPECTED_TRACE_COUNT,
        "source_research_target_length": SOURCE_RESEARCH_TARGET_LENGTH,
        "source_tavily_concurrency": DEFAULT_SEARCH_CONCURRENCY,
        "new_query_planning_request_count": 0,
        "new_tavily_request_count": 0,
        "generation_mode": EXPECTED_GENERATION_MODE,
        "grounding_review_enabled": False,
        "judge_reasoning_effort": "high",
        "judge_sampling": {"temperature": 0.0, "top_p": 1.0},
        "concurrency": {
            "tasks": DEFAULT_TASK_CONCURRENCY,
            "hy3": DEFAULT_HY3_CONCURRENCY,
            "source_search": DEFAULT_SEARCH_CONCURRENCY,
            "new_search": 0,
            "judge": DEFAULT_JUDGE_CONCURRENCY,
        },
        "prompt_versions": _prompt_versions(),
        "prompt_sha256": _prompt_hashes(),
    }
    task_specs = [
        {
            "task_id": item["task_id"],
            "dataset_index": item["dataset_index"],
            "target_length": item["target_length"],
            "source_research_target_length": SOURCE_RESEARCH_TARGET_LENGTH,
            "source_research_snapshot": _relative(
                Path(source_research[item["source_task_id"]]["resolved_snapshot"]),
                experiment_dir,
            ),
            "source_research_snapshot_sha256": source_research[
                item["source_task_id"]
            ]["research_sha256"],
        }
        for item in matrix
    ]
    for name, payload in (
        ("experiment.json", config),
        ("topics.json", topics),
        ("task_matrix.json", matrix),
        ("task_specs.json", task_specs),
    ):
        _write_stable_json(experiment_dir / name, payload)

    runtime = lock_runtime(experiment_dir)
    baseline_runtime = load_json(baseline_dir / "runtime_lock.json")
    if runtime != baseline_runtime:
        raise ValueError("Current model/provider/generation runtime differs from baseline.")
    readme = """# Formal 100-topic frozen-research single-shot experiment v1

This paired experiment reuses the 100 frozen research snapshots from
`formal-100-v1` and expands them into 280/450/700 targets. Each output has one
content-generation call. JSON-only repair calls are unbounded but must preserve
the frozen outline, script text, and reference ids exactly. Successful traces
are immutable and reruns select only missing tasks.

No query-planning or Tavily call is made by this replay. The source research was
collected with Tavily concurrency eight. Hy3 generation and both Judge rounds
are configured with a client-side limit of 512.

```bash
uv run --no-sync python scripts/run_end_to_end_experiment.py prepare
uv run --no-sync python scripts/run_end_to_end_experiment.py generate
uv run --no-sync python scripts/run_end_to_end_experiment.py score
uv run --no-sync python scripts/run_end_to_end_experiment.py repeat
uv run --no-sync python scripts/run_end_to_end_experiment.py report
```

The completed comparison is in `report/comparison.md`; its paired table contains
exactly 300 rows in `report/paired_results.csv`.
"""
    if (experiment_dir / "README.md").exists():
        if (experiment_dir / "README.md").read_text(encoding="utf-8") != readme:
            raise ValueError("Prepared experiment README differs from existing file.")
    else:
        atomic_write_text(experiment_dir / "README.md", readme)
    return config


def _manifest_records(root: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    if not root.exists():
        return ()
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.glob("attempt-*/manifest.json")):
        manifest = load_json(path)
        tasks = manifest.get("tasks") if isinstance(manifest, dict) else None
        if isinstance(tasks, list):
            records.extend((path, item) for item in tasks if isinstance(item, dict))
    return records


def _resolve_artifact(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def _validate_e2e_trace(
    path: Path,
    *,
    expected: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    frozen = load_frozen_trace(path)
    payload = load_json(path)
    if frozen.task.get("topic") != expected["topic"]:
        raise ValueError(f"Trace topic differs for {expected['task_id']}")
    if frozen.task.get("target_length") != expected["target_length"]:
        raise ValueError(f"Trace target length differs for {expected['task_id']}")
    experiment = payload.get("config", {}).get("experiment", {})
    if experiment.get("task_id") != expected["task_id"]:
        raise ValueError(f"Trace task id differs for {expected['task_id']}")
    if experiment.get("mode") != "frozen_research_single_shot_replay":
        raise ValueError(f"Trace is not a frozen-research replay: {expected['task_id']}")
    if experiment.get("source_manifest_sha256") != config.get(
        "source_research_manifest_sha256"
    ):
        raise ValueError(f"Trace source manifest differs for {expected['task_id']}")
    if experiment.get("source_research_snapshot_sha256") != expected.get(
        "source_research_snapshot_sha256"
    ):
        raise ValueError(f"Trace research snapshot differs for {expected['task_id']}")
    if experiment.get("script_system_prompt_sha256") != config["prompt_sha256"][
        "script_generation"
    ]:
        raise ValueError(f"Trace generation prompt differs for {expected['task_id']}")
    if experiment.get("script_format_repair_prompt_sha256") != config[
        "prompt_sha256"
    ]["script_format_repair"]:
        raise ValueError(f"Trace repair prompt differs for {expected['task_id']}")
    lineage = payload.get("lineage", {})
    if lineage.get("script_generation_mode") != EXPECTED_GENERATION_MODE:
        raise ValueError(f"Trace generation mode differs for {expected['task_id']}")
    versions = lineage.get("prompt_versions", {})
    expected_versions = config["prompt_versions"]
    for key in ("research_query_plan", "research_evidence", "script_generation"):
        mapped_key = "background_selection" if key == "research_evidence" else key
        if versions.get(key) != expected_versions[mapped_key]:
            raise ValueError(f"Trace {key} prompt version differs: {expected['task_id']}")
    counts = payload.get("config", {}).get("request_counts", {})
    artifact = payload.get("script_artifact", {})
    format_repairs = artifact.get("format_repair_attempt_count")
    if (
        counts.get("research_llm") != 0
        or counts.get("tavily_attempted") != 0
        or counts.get("script_content_generation_llm") != 1
        or not isinstance(format_repairs, int)
        or isinstance(format_repairs, bool)
        or format_repairs < 0
        or counts.get("script_format_repair_llm") != format_repairs
        or artifact.get("content_generation_attempt_count") != 1
        or artifact.get("generation_attempt_count") != 1 + format_repairs
        or artifact.get("editor_attempt_count") != 0
        or artifact.get("generation_candidates") != []
        or artifact.get("selected_candidate_ids") != []
        or artifact.get("length_repair_attempted") is not False
        or artifact.get("grounding_review_attempt_count") != 0
        or artifact.get("final_rewrite_attempt_count") != 0
        or artifact.get("format_repair_content_preserved") is not True
    ):
        raise ValueError(f"Trace violates single-shot counts: {expected['task_id']}")
    script_text = artifact.get("script_text")
    initial_hash = artifact.get("initial_script_text_sha256")
    if (
        not isinstance(script_text, str)
        or initial_hash != _text_sha256(script_text)
    ):
        raise ValueError(f"Trace did not preserve initial content: {expected['task_id']}")
    stages = [
        call.get("stage")
        for call in lineage.get("llm_calls", [])
        if isinstance(call, dict)
    ]
    if stages.count("script.generation") != 1 or stages.count(
        "script.format_repair"
    ) != format_repairs:
        raise ValueError(f"Trace usage stages differ for {expected['task_id']}")
    if not frozen.queries or frozen.search_result_count < 1 or not frozen.selected_evidence:
        raise ValueError(f"Trace lacks frozen search evidence: {expected['task_id']}")
    return payload


def select_e2e_traces(experiment_dir: Path) -> dict[str, Any]:
    """Select the earliest valid success for each task without overwriting traces."""

    experiment_dir = experiment_dir.resolve()
    config = load_json(experiment_dir / "experiment.json")
    matrix = _validate_matrix(load_json(experiment_dir / "task_matrix.json"))
    specs = load_json(experiment_dir / "task_specs.json")
    spec_by_id = {
        item["task_id"]: item for item in specs if isinstance(item, dict)
    }
    if set(spec_by_id) != {item["task_id"] for item in matrix}:
        raise ValueError("Prepared source research mapping differs from task matrix.")
    expected = {
        item["task_id"]: {**item, **spec_by_id[item["task_id"]]}
        for item in matrix
    }
    selected: dict[str, dict[str, Any]] = {}
    attempts: dict[str, list[dict[str, Any]]] = {task_id: [] for task_id in expected}
    for manifest_path, record in _manifest_records(experiment_dir / "generation"):
        task_id = record.get("task_id")
        if task_id not in expected:
            continue
        attempt_record = {
            "manifest": _relative(manifest_path, experiment_dir),
            "status": record.get("status"),
            "research_status": record.get("research_status"),
            "error_type": record.get("error_type"),
            "usage": record.get("script_usage", record.get("usage", {})),
            "content_generation_attempt_count": record.get(
                "content_generation_attempt_count"
            ),
            "format_repair_attempt_count": record.get(
                "format_repair_attempt_count"
            ),
        }
        attempts[task_id].append(attempt_record)
        trace_value = record.get("trace")
        if record.get("status") != "completed" or not isinstance(trace_value, str):
            continue
        path = _resolve_artifact(manifest_path, trace_value)
        if not path.is_file():
            continue
        try:
            payload = _validate_e2e_trace(
                path,
                expected=expected[task_id],
                config=config,
            )
        except (TraceInputError, ValueError):
            continue
        if task_id in selected:
            continue
        selected[task_id] = {
            **expected[task_id],
            "status": "completed",
            "run_id": payload["run_id"],
            "trace": _relative(path, experiment_dir / "generation"),
            "trace_sha256": sha256_file(path),
            "source_manifest": _relative(manifest_path, experiment_dir / "generation"),
            "source_research_target_length": SOURCE_RESEARCH_TARGET_LENGTH,
            "source_research_snapshot": expected[task_id][
                "source_research_snapshot"
            ],
            "source_research_snapshot_sha256": expected[task_id][
                "source_research_snapshot_sha256"
            ],
            "content_generation_attempt_count": payload["script_artifact"][
                "content_generation_attempt_count"
            ],
            "format_repair_attempt_count": payload["script_artifact"][
                "format_repair_attempt_count"
            ],
            "usage": record.get("script_usage", record.get("usage", {})),
        }
    run_ids = [item["run_id"] for item in selected.values()]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Selected end-to-end traces contain duplicate run ids.")
    manifest = {
        "schema_version": E2E_SCHEMA_VERSION,
        "experiment_id": experiment_dir.name,
        "mode": "frozen_research_single_shot_end_to_end",
        "expected_count": EXPECTED_TRACE_COUNT,
        "selected_count": len(selected),
        "execution": config["concurrency"],
        "new_query_planning_request_count": 0,
        "new_tavily_request_count": 0,
        "tasks": [selected[item["task_id"]] for item in matrix if item["task_id"] in selected],
        "attempts": attempts,
    }
    write_json(experiment_dir / "generation/trace_manifest.json", manifest)
    return manifest


def _run_command(arguments: Sequence[str]) -> int:
    return subprocess.run(arguments, cwd=PROJECT_ROOT, check=False).returncode


def _validate_runtime(experiment_dir: Path, baseline_dir: Path) -> dict[str, Any]:
    config = load_json(experiment_dir / "experiment.json")
    dataset = (experiment_dir / config["dataset"]).resolve()
    rubric = (experiment_dir / config["rubric"]).resolve()
    if sha256_file(dataset) != config["dataset_sha256"]:
        raise ValueError("End-to-end dataset hash changed.")
    if sha256_file(rubric) != config["rubric_sha256"]:
        raise ValueError("End-to-end rubric hash changed.")
    if config.get("prompt_versions") != _prompt_versions():
        raise ValueError("End-to-end prompt versions changed.")
    if config.get("prompt_sha256") != _prompt_hashes():
        raise ValueError("End-to-end prompt text changed.")
    source_manifest = (
        experiment_dir / config["source_research_manifest"]
    ).resolve()
    if sha256_file(source_manifest) != config.get(
        "source_research_manifest_sha256"
    ):
        raise ValueError("Frozen research manifest hash changed.")
    _source_research_records(source_manifest)
    if config.get("new_query_planning_request_count") != 0 or config.get(
        "new_tavily_request_count"
    ) != 0:
        raise ValueError("Frozen-research experiment must not declare live search calls.")
    runtime = lock_runtime(experiment_dir)
    if runtime != load_json(baseline_dir / "runtime_lock.json"):
        raise ValueError("End-to-end runtime differs from the baseline runtime.")
    return config


def generate_e2e_experiment(
    experiment_dir: Path,
    *,
    baseline_dir: Path,
    task_concurrency: int = DEFAULT_TASK_CONCURRENCY,
    hy3_concurrency: int = DEFAULT_HY3_CONCURRENCY,
) -> dict[str, Any]:
    """Run one immutable attempt containing only tasks without a prior success."""

    if not 1 <= task_concurrency <= 512:
        raise ValueError("task-concurrency must be between 1 and 512.")
    if not 1 <= hy3_concurrency <= MAX_HY3_CONCURRENCY:
        raise ValueError("hy3-concurrency must be between 1 and 512.")
    experiment_dir = experiment_dir.resolve()
    baseline_dir = baseline_dir.resolve()
    config = _validate_runtime(experiment_dir, baseline_dir)
    selection = select_e2e_traces(experiment_dir)
    completed = {item["task_id"] for item in selection["tasks"]}
    matrix = _validate_matrix(load_json(experiment_dir / "task_matrix.json"))
    missing = [item for item in matrix if item["task_id"] not in completed]
    if not missing:
        return selection

    number = _attempt_number(experiment_dir / "generation")
    attempt = experiment_dir / "generation" / f"attempt-{number:03d}"
    spec = experiment_dir / "generation/specs" / f"attempt-{number:03d}.json"
    spec_payload = [
        {
            "task_id": item["task_id"],
            "dataset_index": item["dataset_index"],
            "target_length": item["target_length"],
        }
        for item in missing
    ]
    write_json(spec, spec_payload, replace=False)
    source_manifest = (
        experiment_dir / config["source_research_manifest"]
    ).resolve()
    arguments = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/replay_script_generation.py"),
            "--source-manifest",
            str(source_manifest),
            "--output-dir",
            str(attempt),
            "--experiment-id",
            experiment_dir.name,
            "--phase",
            f"end-to-end-attempt-{number:03d}",
            "--concurrency",
            str(task_concurrency),
            "--request-concurrency",
            str(hy3_concurrency),
            "--generation-mode",
            EXPECTED_GENERATION_MODE,
        ]
    for length in TARGET_LENGTHS:
        arguments.extend(("--target-length", str(length)))
    for item in missing:
        arguments.extend(("--output-task-id", item["task_id"]))
    return_code = _run_command(arguments)
    selection = select_e2e_traces(experiment_dir)
    if return_code or selection["selected_count"] != EXPECTED_TRACE_COUNT:
        raise RuntimeError(
            "End-to-end generation is incomplete; rerun generate to retry only missing tasks."
        )
    return selection


def _assert_result_coverage(results_dir: Path, expected: int, filename: str) -> None:
    paths = list(results_dir.glob(f"items/*/{filename}"))
    if len(paths) != expected:
        raise RuntimeError(
            f"Expected {expected} {filename} records, found {len(paths)} in {results_dir}."
        )


def _judge_fingerprints(results_dir: Path) -> set[str]:
    values: set[str] = set()
    for path in results_dir.glob("items/*/hy3_judge.json"):
        payload = load_json(path)
        value = payload.get("metadata", {}).get("evaluator_fingerprint", {}).get("sha256")
        if isinstance(value, str):
            values.add(value)
    return values


def score_e2e_experiment(
    experiment_dir: Path,
    *,
    baseline_dir: Path,
    judge_concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
) -> None:
    if not 1 <= judge_concurrency <= MAX_HY3_CONCURRENCY:
        raise ValueError("judge-concurrency must be between 1 and 512.")
    experiment_dir = experiment_dir.resolve()
    config = _validate_runtime(experiment_dir, baseline_dir.resolve())
    selection = select_e2e_traces(experiment_dir)
    if selection["selected_count"] != EXPECTED_TRACE_COUNT:
        raise ValueError("All 300 end-to-end traces must be frozen before scoring.")
    rubric = (experiment_dir / config["rubric"]).resolve()
    return_code = _run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_evaluation.py"),
            "score",
            "--trace-manifest",
            str(experiment_dir / "generation/trace_manifest.json"),
            "--rubric",
            str(rubric),
            "--evaluators",
            "rules,judge",
            "--output-dir",
            str(experiment_dir / "results"),
            "--concurrency",
            str(judge_concurrency),
            "--reasoning-effort",
            "high",
        ]
    )
    if return_code:
        raise RuntimeError("End-to-end scoring is incomplete; rerun score to resume it.")
    _assert_result_coverage(experiment_dir / "results", EXPECTED_TRACE_COUNT, "combined.json")
    baseline_fingerprints = _judge_fingerprints(baseline_dir.resolve() / "results")
    candidate_fingerprints = _judge_fingerprints(experiment_dir / "results")
    if len(baseline_fingerprints) != 1 or candidate_fingerprints != baseline_fingerprints:
        raise RuntimeError("Candidate Judge fingerprint differs from the baseline.")


def repeat_e2e_judge(
    experiment_dir: Path,
    *,
    baseline_dir: Path,
    judge_concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
) -> dict[str, Any]:
    if not 1 <= judge_concurrency <= MAX_HY3_CONCURRENCY:
        raise ValueError("judge-concurrency must be between 1 and 512.")
    experiment_dir = experiment_dir.resolve()
    config = _validate_runtime(experiment_dir, baseline_dir.resolve())
    selection = select_e2e_traces(experiment_dir)
    if selection["selected_count"] != EXPECTED_TRACE_COUNT:
        raise ValueError("All 300 end-to-end traces must be frozen before Judge repeat.")
    _assert_result_coverage(experiment_dir / "results", EXPECTED_TRACE_COUNT, "hy3_judge.json")
    rubric = (experiment_dir / config["rubric"]).resolve()
    output = experiment_dir / "validation/stability/repeat-001"
    results = output / "results"
    return_code = _run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_evaluation.py"),
            "score",
            "--trace-manifest",
            str(experiment_dir / "generation/trace_manifest.json"),
            "--rubric",
            str(rubric),
            "--evaluators",
            "judge",
            "--output-dir",
            str(results),
            "--concurrency",
            str(judge_concurrency),
            "--reasoning-effort",
            "high",
        ]
    )
    if return_code:
        raise RuntimeError("End-to-end Judge repeat is incomplete; rerun repeat to resume it.")
    _assert_result_coverage(results, EXPECTED_TRACE_COUNT, "hy3_judge.json")
    return export_judge_stability(
        experiment_dir / "results",
        results,
        output,
        trace_manifest=experiment_dir / "generation/trace_manifest.json",
    )


def _combined_records(results_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in results_dir.glob("items/*/combined.json"):
        payload = load_json(path)
        run_id = payload.get("run_id") if isinstance(payload, dict) else None
        if isinstance(run_id, str):
            records[run_id] = payload
    return records


def _usage_by_stage(trace: dict[str, Any], prefix: str) -> int:
    total = 0
    for call in trace.get("lineage", {}).get("llm_calls", []):
        if not isinstance(call, dict) or not str(call.get("stage", "")).startswith(prefix):
            continue
        value = call.get("total_tokens")
        if isinstance(value, int) and not isinstance(value, bool):
            total += value
    return total


def _candidate_rows(experiment_dir: Path) -> list[dict[str, Any]]:
    manifest = load_json(experiment_dir / "generation/trace_manifest.json")
    records = _combined_records(experiment_dir / "results")
    rows: list[dict[str, Any]] = []
    for task in manifest.get("tasks", []):
        record = records.get(task["run_id"])
        if record is None:
            raise ValueError(f"Missing candidate score for {task['task_id']}")
        trace = load_json(experiment_dir / "generation" / task["trace"])
        scores = {
            item["dimension_id"]: item.get("score")
            for item in record.get("dimension_scores", [])
            if isinstance(item, dict) and isinstance(item.get("dimension_id"), str)
        }
        character_count = trace.get("script_artifact", {}).get("character_count")
        target_length = task["target_length"]
        counts = trace.get("config", {}).get("request_counts", {})
        row = {
            "task_id": task["task_id"],
            "source_task_id": task["source_task_id"],
            "dataset_index": task["dataset_index"],
            "topic": task["topic"],
            "target_length": target_length,
            "source_research_target_length": task[
                "source_research_target_length"
            ],
            "domain": task["domain"],
            "challenge_tags": "|".join(task["challenge_tags"]),
            "run_id": task["run_id"],
            "trace_sha256": task["trace_sha256"],
            "gate_failed": bool(record.get("gate_failed")),
            "final_score": record.get("metrics", {}).get("final_score"),
            "character_count": character_count,
            "absolute_length_error_ratio": (
                abs(character_count - target_length) / target_length
                if isinstance(character_count, int)
                else None
            ),
            "research_hy3_total_tokens": _usage_by_stage(trace, "research."),
            "script_hy3_total_tokens": _usage_by_stage(trace, "script."),
            "hy3_total_tokens": trace.get("token_usage", {}).get("hy3_total_tokens"),
            "hy3_attempted_calls": counts.get("hy3_total"),
            "content_generation_attempted_calls": counts.get(
                "script_content_generation_llm"
            ),
            "format_repair_attempted_calls": counts.get(
                "script_format_repair_llm"
            ),
            "search_attempted_calls": counts.get("tavily_attempted"),
            "search_succeeded_calls": counts.get("tavily_succeeded"),
            "search_latency_seconds": trace.get("latency", {}).get(
                "search_response_time_sum"
            ),
        }
        row.update({dimension: scores.get(dimension) for dimension in _DIMENSIONS})
        rows.append(row)
    if len(rows) != EXPECTED_TRACE_COUNT:
        raise ValueError("Candidate report requires exactly 300 scored rows.")
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_text(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _quality_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scores = [
        float(row["final_score"])
        for row in rows
        if row.get("final_score") not in (None, "")
    ]
    return {
        "count": len(rows),
        "scored": len(scores),
        "gate_failed": sum(bool(row.get("gate_failed")) for row in rows),
        "final_score_mean": _mean(scores),
        "dimensions": {
            dimension: _mean(
                [
                    float(row[dimension])
                    for row in rows
                    if row.get(dimension) not in (None, "")
                ]
            )
            for dimension in _DIMENSIONS
        },
    }


def _resource_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def numeric_values(key: str) -> list[float]:
        return [
            float(row[key])
            for row in rows
            if row.get(key) not in (None, "")
        ]

    character_counts = numeric_values("character_count")
    length_errors = numeric_values("absolute_length_error_ratio")
    return {
        "character_count_mean": _mean(character_counts),
        "absolute_length_error_ratio_mean": _mean(length_errors),
        "hy3_attempted_calls": sum(numeric_values("hy3_attempted_calls")),
        "content_generation_attempted_calls": sum(
            numeric_values("content_generation_attempted_calls")
        ),
        "format_repair_attempted_calls": sum(
            numeric_values("format_repair_attempted_calls")
        ),
        "hy3_total_tokens": sum(numeric_values("hy3_total_tokens")),
        "search_attempted_calls": sum(numeric_values("search_attempted_calls")),
        "search_latency_seconds": sum(numeric_values("search_latency_seconds")),
    }


def _profile_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary = _quality_summary(rows)
    summary["resources"] = _resource_summary(rows)
    return summary


def _group_summary(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return {name: _profile_summary(items) for name, items in sorted(groups.items())}


def _challenge_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for tag in str(row.get("challenge_tags", "")).split("|"):
            if tag:
                groups.setdefault(tag, []).append(row)
    return {name: _profile_summary(items) for name, items in sorted(groups.items())}


def _judge_usage(results_dir: Path) -> dict[str, Any]:
    calls = 0
    total_tokens = 0
    for path in results_dir.glob("items/*/hy3_judge.json"):
        payload = load_json(path)
        attempts = payload.get("metadata", {}).get("attempts", [])
        if isinstance(attempts, list):
            calls += len(attempts)
        usage = payload.get("metadata", {}).get("usage", {})
        value = usage.get("total_tokens") if isinstance(usage, dict) else None
        if isinstance(value, int) and not isinstance(value, bool):
            total_tokens += value
    return {"reported_calls": calls, "total_tokens": total_tokens}


def _evaluation_resume_summary(results_dir: Path, expected: int) -> dict[str, Any]:
    manifest = load_json(results_dir / "manifest.json")
    started_at = manifest.get("started_at")
    last_started_at = manifest.get("last_started_at")
    resumed = (
        isinstance(started_at, str)
        and isinstance(last_started_at, str)
        and last_started_at != started_at
    )
    completed_in_last_resume = 0
    if resumed:
        for path in results_dir.glob("items/*/hy3_judge.json"):
            created_at = load_json(path).get("created_at")
            if isinstance(created_at, str) and created_at >= last_started_at:
                completed_in_last_resume += 1
    return {
        "resume_detected": resumed,
        "completed_before_last_resume": (
            expected - completed_in_last_resume if resumed else expected
        ),
        "remaining_before_last_resume": completed_in_last_resume if resumed else 0,
        "completed_in_last_resume": completed_in_last_resume,
    }


def _attempt_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    all_attempts = [
        attempt
        for values in manifest.get("attempts", {}).values()
        for attempt in values
    ]
    first_attempts = [
        values[0]
        for values in manifest.get("attempts", {}).values()
        if values
    ]
    return {
        "attempt_batch_count": max(
            (len(values) for values in manifest.get("attempts", {}).values()),
            default=0,
        ),
        "attempt_record_count": len(all_attempts),
        "first_attempt_completed": sum(
            item.get("status") == "completed" for item in first_attempts
        ),
        "first_attempt_failed": sum(
            item.get("status") != "completed" for item in first_attempts
        ),
        "failed_attempt_records": sum(
            item.get("status") != "completed" for item in all_attempts
        ),
        "selected_format_repair_calls": sum(
            int(task.get("format_repair_attempt_count", 0) or 0)
            for task in manifest.get("tasks", [])
        ),
        "all_attempt_format_repair_calls": sum(
            int(item.get("format_repair_attempt_count", 0) or 0)
            for item in all_attempts
        ),
        "all_attempt_hy3_tokens": sum(
            int(item.get("usage", {}).get("total_tokens", 0) or 0)
            for item in all_attempts
        ),
        "all_attempt_search_calls": sum(
            int(item.get("usage", {}).get("tavily_attempted_calls", 0) or 0)
            for item in all_attempts
        ),
    }


def _baseline_rows(baseline_dir: Path) -> list[dict[str, Any]]:
    manifest = load_json(baseline_dir / "generation/trace_manifest.json")
    tasks = {item["task_id"]: item for item in manifest.get("tasks", [])}
    if len(tasks) != EXPECTED_TRACE_COUNT:
        raise ValueError("Baseline trace manifest must contain exactly 300 tasks.")
    rows: list[dict[str, Any]] = []
    for raw in _read_csv(baseline_dir / "results/full_results.csv"):
        row: dict[str, Any] = dict(raw)
        for key in ("target_length", *_DIMENSIONS):
            if row.get(key) not in (None, ""):
                row[key] = int(row[key])
        if row.get("final_score") not in (None, ""):
            row["final_score"] = float(row["final_score"])
        row["gate_failed"] = str(row.get("gate_failed", "")).lower() == "true"
        task = tasks.get(row["task_id"])
        if task is None:
            raise ValueError(f"Missing baseline trace for {row['task_id']}")
        trace = load_json(baseline_dir / "generation" / task["trace"])
        target_length = int(row["target_length"])
        character_count = trace.get("script_artifact", {}).get("character_count")
        counts = trace.get("config", {}).get("request_counts", {})
        allocation = float(len(TARGET_LENGTHS))
        research_tokens = _usage_by_stage(trace, "research.") / allocation
        script_tokens = float(raw.get("hy3_total_tokens") or 0)
        row.update(
            {
                "character_count": character_count,
                "absolute_length_error_ratio": (
                    abs(character_count - target_length) / target_length
                    if isinstance(character_count, int)
                    else None
                ),
                # The baseline shares one research run across three lengths. Allocate
                # one third to each paired row so grouped totals remain additive.
                "research_hy3_total_tokens": research_tokens,
                "script_hy3_total_tokens": script_tokens,
                "hy3_total_tokens": script_tokens + research_tokens,
                "hy3_attempted_calls": float(counts.get("script_llm", 0) or 0)
                + float(counts.get("research_llm", 0) or 0) / allocation,
                "content_generation_attempted_calls": float(
                    counts.get("script_candidate_llm", 0)
                    or counts.get("script_generation_llm", 0)
                    or 0
                ),
                "format_repair_attempted_calls": float(
                    counts.get("script_format_repair_llm", 0) or 0
                ),
                "search_attempted_calls": float(
                    counts.get("tavily_attempted", 0) or 0
                )
                / allocation,
                "search_succeeded_calls": float(
                    counts.get("tavily_succeeded", 0) or 0
                )
                / allocation,
                "search_latency_seconds": float(
                    trace.get("latency", {}).get("search_response_time_sum", 0) or 0
                )
                / allocation,
            }
        )
        rows.append(row)
    if len(rows) != EXPECTED_TRACE_COUNT:
        raise ValueError("Baseline report must contain exactly 300 scored rows.")
    return rows


def _paired_rows(
    baseline_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline = {row["task_id"]: row for row in baseline_rows}
    candidate = {row["task_id"]: row for row in candidate_rows}
    if len(baseline) != EXPECTED_TRACE_COUNT or set(baseline) != set(candidate):
        raise ValueError("Baseline and candidate task ids do not form 300 exact pairs.")
    rows: list[dict[str, Any]] = []
    for task_id in baseline:
        first = baseline[task_id]
        second = candidate[task_id]
        baseline_score = first.get("final_score")
        candidate_score = second.get("final_score")
        evaluable = isinstance(baseline_score, (int, float)) and isinstance(
            candidate_score, (int, float)
        )
        delta = float(candidate_score) - float(baseline_score) if evaluable else None
        result = (
            "win" if evaluable and delta > 0
            else "loss" if evaluable and delta < 0
            else "tie" if evaluable
            else "unavailable"
        )
        row = {
            "task_id": task_id,
            "topic": second["topic"],
            "target_length": second["target_length"],
            "domain": second["domain"],
            "challenge_tags": second["challenge_tags"],
            "baseline_run_id": first["run_id"],
            "candidate_run_id": second["run_id"],
            "baseline_final_score": baseline_score,
            "candidate_final_score": candidate_score,
            "final_score_delta": delta,
            "paired_result": result,
            "baseline_gate_failed": first["gate_failed"],
            "candidate_gate_failed": second["gate_failed"],
            "baseline_character_count": first.get("character_count"),
            "candidate_character_count": second["character_count"],
            "baseline_absolute_length_error_ratio": first.get(
                "absolute_length_error_ratio"
            ),
            "candidate_absolute_length_error_ratio": second[
                "absolute_length_error_ratio"
            ],
            "baseline_hy3_attempted_calls": first.get("hy3_attempted_calls"),
            "candidate_hy3_attempted_calls": second.get("hy3_attempted_calls"),
            "baseline_content_generation_attempted_calls": first.get(
                "content_generation_attempted_calls"
            ),
            "candidate_content_generation_attempted_calls": second.get(
                "content_generation_attempted_calls"
            ),
            "baseline_format_repair_attempted_calls": first.get(
                "format_repair_attempted_calls"
            ),
            "candidate_format_repair_attempted_calls": second.get(
                "format_repair_attempted_calls"
            ),
            "baseline_hy3_total_tokens": first.get("hy3_total_tokens"),
            "candidate_hy3_total_tokens": second["hy3_total_tokens"],
            "baseline_search_attempted_calls": first.get("search_attempted_calls"),
            "candidate_search_attempted_calls": second["search_attempted_calls"],
            "baseline_search_latency_seconds": first.get("search_latency_seconds"),
            "candidate_search_latency_seconds": second.get("search_latency_seconds"),
        }
        for dimension in _DIMENSIONS:
            left = first.get(dimension)
            right = second.get(dimension)
            row[f"baseline_{dimension}"] = left
            row[f"candidate_{dimension}"] = right
            row[f"delta_{dimension}"] = (
                int(right) - int(left)
                if left not in (None, "") and right not in (None, "")
                else None
            )
        rows.append(row)
    return rows


def _paired_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    deltas = [
        float(row["final_score_delta"])
        for row in rows
        if row.get("final_score_delta") is not None
    ]
    return {
        "pair_count": len(rows),
        "evaluable_pair_count": len(deltas),
        "wins": sum(row["paired_result"] == "win" for row in rows),
        "ties": sum(row["paired_result"] == "tie" for row in rows),
        "losses": sum(row["paired_result"] == "loss" for row in rows),
        "mean_delta": _mean(deltas),
        "median_delta": statistics.median(deltas) if deltas else None,
        "dimensions": {
            dimension: {
                "mean_delta": _mean(
                    [float(row[f"delta_{dimension}"]) for row in rows]
                ),
                "improved": sum(row[f"delta_{dimension}"] > 0 for row in rows),
                "unchanged": sum(row[f"delta_{dimension}"] == 0 for row in rows),
                "declined": sum(row[f"delta_{dimension}"] < 0 for row in rows),
            }
            for dimension in _DIMENSIONS
        },
    }


def _append_group_table(
    lines: list[str],
    *,
    title: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    labels: dict[str, str] | None = None,
) -> None:
    if set(baseline) != set(candidate):
        raise ValueError(f"Baseline and candidate {title} groups differ.")
    lines.extend(
        [
            "",
            f"### {title}",
            "",
            "| 分组 | 输出数 | 平均分 基线/单次生成 | 平均实际字数 基线/单次生成 | "
            "搜索请求 历史基线/本次新增 | 搜索延迟秒 历史基线/本次新增 | "
            "Hy3 请求 基线/单次生成 | Hy3 token 基线/单次生成 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in baseline:
        first = baseline[name]
        second = candidate[name]
        first_resources = first["resources"]
        second_resources = second["resources"]
        label = labels.get(name, name) if labels else name
        lines.append(
            f"| {label} | {first['count']} | "
            f"{first['final_score_mean']:.4f} / {second['final_score_mean']:.4f} | "
            f"{first_resources['character_count_mean']:.1f} / "
            f"{second_resources['character_count_mean']:.1f} | "
            f"{first_resources['search_attempted_calls']:.0f} / "
            f"{second_resources['search_attempted_calls']:.0f} | "
            f"{first_resources['search_latency_seconds']:.2f} / "
            f"{second_resources['search_latency_seconds']:.2f} | "
            f"{first_resources['hy3_attempted_calls']:.1f} / "
            f"{second_resources['hy3_attempted_calls']:.0f} | "
            f"{first_resources['hy3_total_tokens']:.0f} / "
            f"{second_resources['hy3_total_tokens']:.0f} |"
        )


def export_e2e_report(
    experiment_dir: Path,
    *,
    baseline_dir: Path,
) -> dict[str, Any]:
    """Export candidate, paired, resource, and Judge-stability comparisons."""

    experiment_dir = experiment_dir.resolve()
    baseline_dir = baseline_dir.resolve()
    _validate_runtime(experiment_dir, baseline_dir)
    manifest = select_e2e_traces(experiment_dir)
    if manifest["selected_count"] != EXPECTED_TRACE_COUNT:
        raise ValueError("Report requires 300 selected end-to-end traces.")
    _assert_result_coverage(experiment_dir / "results", EXPECTED_TRACE_COUNT, "combined.json")
    stability_path = experiment_dir / "validation/stability/repeat-001/summary.json"
    if not stability_path.is_file():
        raise ValueError("Report requires a completed Judge repeat summary.")

    candidate_rows = _candidate_rows(experiment_dir)
    baseline_rows = _baseline_rows(baseline_dir)
    paired_rows = _paired_rows(baseline_rows, candidate_rows)
    paired = _paired_summary(paired_rows)
    candidate_quality = _quality_summary(candidate_rows)
    baseline_quality = _quality_summary(baseline_rows)
    first_judge = _judge_usage(experiment_dir / "results")
    first_judge["resume"] = _evaluation_resume_summary(
        experiment_dir / "results", EXPECTED_TRACE_COUNT
    )
    repeat_results = experiment_dir / "validation/stability/repeat-001/results"
    repeat_judge = _judge_usage(repeat_results)
    repeat_judge["resume"] = _evaluation_resume_summary(
        repeat_results, EXPECTED_TRACE_COUNT
    )
    candidate_resources = {
        "selected_research_hy3_tokens": sum(
            int(row["research_hy3_total_tokens"] or 0) for row in candidate_rows
        ),
        "selected_script_hy3_tokens": sum(
            int(row["script_hy3_total_tokens"] or 0) for row in candidate_rows
        ),
        "selected_hy3_tokens": sum(
            int(row["hy3_total_tokens"] or 0) for row in candidate_rows
        ),
        "selected_search_calls": sum(
            int(row["search_attempted_calls"] or 0) for row in candidate_rows
        ),
        "selected_search_latency_seconds": sum(
            float(row["search_latency_seconds"] or 0) for row in candidate_rows
        ),
        "selected_content_generation_calls": sum(
            int(row["content_generation_attempted_calls"] or 0)
            for row in candidate_rows
        ),
        "selected_format_repair_calls": sum(
            int(row["format_repair_attempted_calls"] or 0)
            for row in candidate_rows
        ),
        "first_judge": first_judge,
        "repeat_judge": repeat_judge,
        "attempts": _attempt_summary(manifest),
    }
    baseline_resources = load_json(baseline_dir / "report/analysis_summary.json")
    candidate_stability = load_json(stability_path)
    baseline_stability = load_json(
        baseline_dir / "validation/stability/repeat-001/summary.json"
    )
    by_length = {
        "baseline": _group_summary(baseline_rows, "target_length"),
        "candidate": _group_summary(candidate_rows, "target_length"),
    }
    by_domain = {
        "baseline": _group_summary(baseline_rows, "domain"),
        "candidate": _group_summary(candidate_rows, "domain"),
    }
    by_challenge = {
        "baseline": _challenge_summary(baseline_rows),
        "candidate": _challenge_summary(candidate_rows),
    }
    summary = {
        "schema_version": E2E_SCHEMA_VERSION,
        "baseline_experiment": baseline_dir.name,
        "candidate_experiment": experiment_dir.name,
        "concurrency": load_json(experiment_dir / "experiment.json")["concurrency"],
        "baseline_quality": baseline_quality,
        "candidate_quality": candidate_quality,
        "paired": paired,
        "by_length": by_length,
        "by_domain": by_domain,
        "by_challenge": by_challenge,
        "baseline_resources": baseline_resources,
        "candidate_resources": candidate_resources,
        "baseline_judge_stability": baseline_stability,
        "candidate_judge_stability": candidate_stability,
    }
    report_dir = experiment_dir / "report"
    write_json(report_dir / "comparison_summary.json", summary)
    atomic_write_text(
        experiment_dir / "results/full_results.csv",
        _csv_text(candidate_rows),
    )
    atomic_write_text(report_dir / "paired_results.csv", _csv_text(paired_rows))

    baseline_mean = baseline_quality["final_score_mean"]
    candidate_mean = candidate_quality["final_score_mean"]
    lines = [
        "# 端到端单次生成实验（复用基线冻结检索）",
        "",
        f"`{experiment_dir.name}` 与 `{baseline_dir.name}` 严格配对对比。",
        "",
        "## 完整性与配置",
        "",
        f"- 配对样本：{paired['evaluable_pair_count']}/{paired['pair_count']}",
        "- 生成方式：复用基线 100 份冻结研究快照，每个 topic/length 只执行一次内容生成；"
        "仅 JSON 格式错误允许无上限、内容保持不变的格式修复。",
        "- 来源研究目标字数为 450；生成目标字数为 280/450/700。",
        "- 任务 / Hy3 / 来源搜索 / 新搜索 / Judge 并发：300 / 512 / 8 / 0 / 512。",
        "- 本次没有新的查询规划或 Tavily 调用；快照中的查询和搜索结果只作为冻结输入。",
        "",
        "## 质量对比",
        "",
        "| 指标 | 三候选主编基线 | 端到端直接生成 | 差值 |",
        "| --- | ---: | ---: | ---: |",
        f"| 平均最终分 | {baseline_mean:.6f} | {candidate_mean:.6f} | {paired['mean_delta']:+.6f} |",
        f"| 门控失败 | {baseline_quality['gate_failed']} | {candidate_quality['gate_failed']} | "
        f"{candidate_quality['gate_failed'] - baseline_quality['gate_failed']:+d} |",
        "",
        f"成对结果：单次生成胜 {paired['wins']}、平 {paired['ties']}、负 {paired['losses']}；"
        f"中位差值 {paired['median_delta']:+.6f}。",
        "",
        "### 分维度",
        "",
        "| 维度 | 基线均分 | 单次生成均分 | 平均差值 | 改善/不变/下降 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dimension in _DIMENSIONS:
        stats = paired["dimensions"][dimension]
        lines.append(
            f"| {_DIMENSION_LABELS[dimension]} | "
            f"{baseline_quality['dimensions'][dimension]:.4f} | "
            f"{candidate_quality['dimensions'][dimension]:.4f} | "
            f"{stats['mean_delta']:+.4f} | "
            f"{stats['improved']}/{stats['unchanged']}/{stats['declined']} |"
        )
    attempts = candidate_resources["attempts"]
    lines.extend(
        [
            "",
            "## 可靠性与资源",
            "",
            f"- 首次尝试成功：{attempts['first_attempt_completed']}/300；"
            f"首次失败：{attempts['first_attempt_failed']}。",
            f"- 生成共 {attempts['attempt_batch_count']} 个批次、"
            f"{attempts['attempt_record_count']} 条 attempt 记录；其中失败记录 "
            f"{attempts['failed_attempt_records']} 条，只补失败项。",
            f"- 选中 trace 的 Hy3 token：{candidate_resources['selected_hy3_tokens']}；"
            f"其中本次检索规划 {candidate_resources['selected_research_hy3_tokens']}，"
            f"单次生成与格式修复 {candidate_resources['selected_script_hy3_tokens']}。",
            f"- 内容生成调用：{candidate_resources['selected_content_generation_calls']}；"
            f"格式修复调用：{candidate_resources['selected_format_repair_calls']}。",
            f"- 本次新增搜索请求：{candidate_resources['selected_search_calls']}；"
            f"累计响应时间 {candidate_resources['selected_search_latency_seconds']:.2f} 秒。",
            f"- 首轮 Judge：{candidate_resources['first_judge']['reported_calls']} 次请求，"
            f"{candidate_resources['first_judge']['total_tokens']} token。",
            f"- 复评 Judge：{candidate_resources['repeat_judge']['reported_calls']} 次请求，"
            f"{candidate_resources['repeat_judge']['total_tokens']} token。",
        ]
    )
    for label, usage in (
        ("首轮 Judge", candidate_resources["first_judge"]),
        ("复评 Judge", candidate_resources["repeat_judge"]),
    ):
        resume = usage["resume"]
        if resume["resume_detected"]:
            lines.append(
                f"- {label} 首批完成 {resume['completed_before_last_resume']}/300，"
                f"剩余 {resume['remaining_before_last_resume']} 项在续跑中全部完成。"
            )
    lines.extend(
        [
            "",
            "两组都使用同一批基线冻结研究。下表中的基线搜索量是当时实际发生并按三档均摊的"
            "历史成本；单次生成侧显示的是本次 replay 的新增成本，因此搜索列不能解释为流程本身"
            "天然节省了检索。",
        ]
    )
    _append_group_table(
        lines,
        title="按目标长度",
        baseline=by_length["baseline"],
        candidate=by_length["candidate"],
    )
    _append_group_table(
        lines,
        title="按领域",
        baseline=by_domain["baseline"],
        candidate=by_domain["candidate"],
        labels=_DOMAIN_LABELS,
    )
    _append_group_table(
        lines,
        title="按难例标签（标签可重叠）",
        baseline=by_challenge["baseline"],
        candidate=by_challenge["candidate"],
        labels=_CHALLENGE_LABELS,
    )
    lines.extend(
        [
            "",
            "## Judge 内部一致性",
            "",
        ]
    )
    for label, stability in (
        ("三候选主编基线", baseline_stability),
        ("端到端直接生成", candidate_stability),
    ):
        overall = stability["overall"]
        lines.append(
            f"- {label}：逐维完全一致率 "
            f"{overall['dimension_exact_agreement_rate']:.4f}，"
            f"全维一致率 {overall['all_dimensions_exact_rate']:.4f}，"
            f"总分 MAE {overall['normalized_score_mae']:.6f}，"
            f"Spearman {overall['normalized_score_spearman']}。"
        )
    lines.extend(
        [
            "",
            "| Judge 维度 | 基线一致率 | 单次生成一致率 | 基线二次加权 Kappa | "
            "单次生成二次加权 Kappa | 基线 MAE | 单次生成 MAE |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for dimension in _DIMENSIONS:
        if dimension == "length_compliance":
            continue
        first = baseline_stability["dimensions"][dimension]
        second = candidate_stability["dimensions"][dimension]
        lines.append(
            f"| {_DIMENSION_LABELS[dimension]} | "
            f"{first['exact_agreement_rate']:.4f} | "
            f"{second['exact_agreement_rate']:.4f} | "
            f"{first['quadratic_weighted_kappa']:.4f} | "
            f"{second['quadratic_weighted_kappa']:.4f} | "
            f"{first['mae']:.4f} | {second['mae']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "两组使用相同冻结研究、Rubric 和 Judge 指纹。单次生成调用现有 background-generation"
            " prompt，而基线使用 editorial pipeline；生成随机性仍然存在。因此差异描述的是"
            "单次生成与三候选主编流程的整体结果，不是实时搜索可靠性测试。",
            "",
        ]
    )
    atomic_write_text(report_dir / "comparison.md", "\n".join(lines))
    return summary


__all__ = [
    "DEFAULT_HY3_CONCURRENCY",
    "DEFAULT_JUDGE_CONCURRENCY",
    "DEFAULT_SEARCH_CONCURRENCY",
    "DEFAULT_TASK_CONCURRENCY",
    "EXPECTED_TRACE_COUNT",
    "export_e2e_report",
    "generate_e2e_experiment",
    "prepare_e2e_experiment",
    "repeat_e2e_judge",
    "score_e2e_experiment",
    "select_e2e_traces",
]
