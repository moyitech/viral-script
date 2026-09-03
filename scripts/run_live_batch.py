"""Run a bounded live-research and script-generation batch from JSON inputs."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

from hyscript.agent import (
    ResearchAgent,
    ResearchGenerationError,
    ResearchOutcome,
    ScriptAgent,
    ScriptArtifact,
    ScriptGenerationError,
    ScriptTask,
)
from hyscript.artifacts import build_generation_trace
from hyscript.config import settings
from hyscript.llm import AsyncHy3Client, summarize_token_usage
from hyscript.llm.prompts import (
    BACKGROUND_SCRIPT_CANDIDATE_PROMPT_VERSION,
    BACKGROUND_SCRIPT_CANDIDATE_SYSTEM_PROMPT,
    BACKGROUND_SCRIPT_EDITOR_PROMPT_VERSION,
    BACKGROUND_SCRIPT_EDITOR_SYSTEM_PROMPT,
    BACKGROUND_SCRIPT_GENERATION_PROMPT_VERSION,
    BACKGROUND_SCRIPT_GENERATION_SYSTEM_PROMPT,
    BACKGROUND_SCRIPT_PIPELINE_VERSION,
    BACKGROUND_SELECTION_VERSION,
    RESEARCH_QUERY_PLAN_PROMPT_VERSION,
    RESEARCH_QUERY_PLAN_SYSTEM_PROMPT,
)
from hyscript.search import AsyncTavilySearchProvider


_TASK_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_TASK_SPEC_FIELDS = {"task_id", "dataset_index", "target_length"}
_MAX_TASK_CONCURRENCY = 512
_MAX_HY3_CONCURRENCY = 512
_MAX_SEARCH_CONCURRENCY = 64


def _print_console(message: str) -> None:
    """Print Unicode safely even when Windows exposes a legacy code page."""

    encoding = getattr(sys.stdout, "encoding", None)
    if encoding:
        message = message.encode(encoding, errors="backslashreplace").decode(encoding)
    print(message, flush=True)


@dataclass(frozen=True, slots=True)
class BatchTask:
    """A validated dataset reference plus its request-scoped script task."""

    task_id: str
    dataset_index: int
    task: ScriptTask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run live search followed by background-informed script generation. "
            "Tasks use bounded concurrency and evaluation is not performed."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="UTF-8 JSON file containing a list of topic strings.",
    )
    parser.add_argument(
        "--task-spec",
        type=Path,
        required=True,
        help=(
            "UTF-8 JSON file containing a list of task_id, dataset_index, and "
            "target_length objects."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument(
        "--task-concurrency",
        type=int,
        choices=range(1, _MAX_TASK_CONCURRENCY + 1),
        default=None,
        metavar=f"1-{_MAX_TASK_CONCURRENCY}",
        help="Maximum number of complete task pipelines in flight (default: 1).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        choices=range(1, _MAX_TASK_CONCURRENCY + 1),
        default=1,
        metavar=f"1-{_MAX_TASK_CONCURRENCY}",
        help="Deprecated alias for --task-concurrency.",
    )
    parser.add_argument(
        "--hy3-concurrency",
        type=int,
        choices=range(1, _MAX_HY3_CONCURRENCY + 1),
        default=None,
        metavar=f"1-{_MAX_HY3_CONCURRENCY}",
        help="Maximum Hy3 requests in flight across research and generation.",
    )
    parser.add_argument(
        "--request-concurrency",
        type=int,
        choices=range(1, _MAX_HY3_CONCURRENCY + 1),
        default=16,
        metavar=f"1-{_MAX_HY3_CONCURRENCY}",
        help="Deprecated alias for --hy3-concurrency.",
    )
    parser.add_argument(
        "--search-concurrency",
        type=int,
        choices=range(1, _MAX_SEARCH_CONCURRENCY + 1),
        default=8,
        metavar=f"1-{_MAX_SEARCH_CONCURRENCY}",
        help="Maximum Tavily requests in flight across all tasks (default: 8).",
    )
    parser.add_argument(
        "--generation-mode",
        choices=("single", "editorial_candidates"),
        default=None,
        help="Override the configured background-script generation mode.",
    )
    parser.add_argument(
        "--research-only",
        action="store_true",
        help="Freeze live background snapshots without generating scripts.",
    )
    return parser


def _read_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load {label}: {path}") from exc


def _load_dataset(path: Path) -> tuple[str, ...]:
    payload = _read_json(path, label="dataset")
    if not isinstance(payload, list) or any(not isinstance(item, str) for item in payload):
        raise ValueError("Dataset must be a JSON list of strings.")
    if not payload:
        raise ValueError("Dataset must not be empty.")
    return tuple(payload)


def _load_batch_tasks(path: Path, dataset: tuple[str, ...]) -> tuple[BatchTask, ...]:
    payload = _read_json(path, label="task spec")
    if not isinstance(payload, list) or not payload:
        raise ValueError("Task spec must be a non-empty JSON list of objects.")

    tasks: list[BatchTask] = []
    seen_ids: set[str] = set()
    for position, raw_task in enumerate(payload):
        context = f"Task spec item {position}"
        if not isinstance(raw_task, dict):
            raise ValueError(f"{context} must be an object.")
        if set(raw_task) != _TASK_SPEC_FIELDS:
            raise ValueError(
                f"{context} must contain exactly task_id, dataset_index, and "
                "target_length."
            )

        task_id = raw_task["task_id"]
        if not isinstance(task_id, str) or _TASK_ID_PATTERN.fullmatch(task_id) is None:
            raise ValueError(
                f"{context} task_id must use 1-64 ASCII letters, digits, '_' or '-'."
            )
        if task_id in seen_ids:
            raise ValueError(f"Task spec repeats task_id: {task_id}")
        seen_ids.add(task_id)

        dataset_index = raw_task["dataset_index"]
        if (
            isinstance(dataset_index, bool)
            or not isinstance(dataset_index, int)
            or not 0 <= dataset_index < len(dataset)
        ):
            raise ValueError(f"{context} dataset_index is outside the dataset.")

        target_length = raw_task["target_length"]
        if isinstance(target_length, bool) or not isinstance(target_length, int):
            raise ValueError(f"{context} target_length must be an integer.")
        try:
            script_task = ScriptTask(
                topic=dataset[dataset_index],
                target_length=target_length,
            )
        except ValueError as exc:
            raise ValueError(f"{context} is invalid: {exc}") from exc
        tasks.append(
            BatchTask(
                task_id=task_id,
                dataset_index=dataset_index,
                task=script_task,
            )
        )
    return tuple(tasks)


def _validate_label(value: str, *, name: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > 120 or any(
        character in "\r\n\x00" for character in normalized
    ):
        raise ValueError(
            f"{name} must be a non-empty single-line value of at most 120 characters."
        )
    return normalized


def _validate_task_concurrency(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= _MAX_TASK_CONCURRENCY
    ):
        raise ValueError(
            f"task-concurrency must be between 1 and {_MAX_TASK_CONCURRENCY}."
        )
    return value


def _resolve_concurrency(
    preferred: Any,
    legacy: Any,
    *,
    preferred_name: str,
    legacy_name: str,
    default: int,
) -> int:
    if preferred is not None:
        return preferred
    if legacy is not None and legacy != default:
        _print_console(f"warning: --{legacy_name} is deprecated; use --{preferred_name}")
    value = preferred if preferred is not None else legacy
    return default if value is None else value


def _prepare_output_dir(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"Output path is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError(
            f"Output directory is not empty; refusing to overwrite or resume: {path}"
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(
    path: Path,
    payload: Any,
    *,
    replace: bool,
) -> None:
    """Atomically create or replace one private JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.chmod(temporary_path, 0o600)
        if replace:
            os.replace(temporary_path, path)
        else:
            os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _usage_payload(
    research: ResearchOutcome | ResearchGenerationError,
    script: ScriptArtifact | None,
    script_error: ScriptGenerationError | None = None,
) -> dict[str, int]:
    if script is not None and script_error is not None:
        raise ValueError("script and script_error are mutually exclusive.")
    script_usages = (
        script.llm_usages
        if script is not None
        else script_error.llm_usages
        if script_error is not None
        else ()
    )
    usages = (*research.llm_usages, *script_usages)
    summary = summarize_token_usage(usages)
    generation_attempts = (
        script.generation_attempt_count
        if script is not None
        else script_error.generation_attempt_count
        if script_error is not None
        else 0
    )
    review_attempts = (
        script.grounding_review_attempt_count
        if script is not None
        else script_error.grounding_review_attempt_count
        if script_error is not None
        else 0
    )
    successful_search_count = (
        len(research.search_responses)
        if isinstance(research, ResearchOutcome)
        else research.successful_search_count
    )
    return {
        "hy3_attempted_calls": (
            research.llm_request_count + generation_attempts + review_attempts
        ),
        "research_hy3_attempted_calls": research.llm_request_count,
        "script_generation_attempted_calls": generation_attempts,
        "script_grounding_review_attempted_calls": review_attempts,
        "hy3_reported_calls": summary.reported_call_count,
        "hy3_input_tokens": summary.input_tokens,
        "hy3_output_tokens": summary.output_tokens,
        "hy3_total_tokens": summary.total_tokens,
        "hy3_reasoning_tokens": summary.reasoning_tokens,
        "hy3_cached_input_tokens": summary.cached_input_tokens,
        "tavily_attempted_calls": research.search_request_count,
        "tavily_succeeded_calls": successful_search_count,
    }


def _safe_error_message(exc: Exception) -> str:
    if isinstance(exc, (ResearchGenerationError, ScriptGenerationError)):
        return " ".join(str(exc).split())[:300]
    return "Unexpected task failure. See local console logs for the exception type."


def _safe_research_errors(research: ResearchOutcome) -> list[str]:
    return [" ".join(message.split())[:300] for message in research.errors]


def _counts(
    records: list[dict[str, Any]],
    *,
    input_count: int,
) -> dict[str, int]:
    pending = sum(item["status"] == "pending" for item in records)
    return {
        "input": input_count,
        "accounted": len(records) - pending,
        "pending": pending,
        "completed": sum(item["status"] == "completed" for item in records),
        "insufficient_evidence": sum(
            item["status"] == "insufficient_evidence" for item in records
        ),
        "failed": sum(item["status"] == "failed" for item in records),
        "running": sum(item["status"] == "running" for item in records),
    }


async def _run(args: argparse.Namespace) -> int:
    task_concurrency = _validate_task_concurrency(
        _resolve_concurrency(
            getattr(args, "task_concurrency", None),
            getattr(args, "concurrency", None),
            preferred_name="task-concurrency",
            legacy_name="concurrency",
            default=1,
        )
    )
    request_concurrency = _resolve_concurrency(
        getattr(args, "hy3_concurrency", None),
        getattr(args, "request_concurrency", None),
        preferred_name="hy3-concurrency",
        legacy_name="request-concurrency",
        default=16,
    )
    search_concurrency = getattr(args, "search_concurrency", 8)
    if not 1 <= request_concurrency <= _MAX_HY3_CONCURRENCY:
        raise ValueError(
            f"hy3-concurrency must be between 1 and {_MAX_HY3_CONCURRENCY}."
        )
    if not 1 <= search_concurrency <= _MAX_SEARCH_CONCURRENCY:
        raise ValueError(
            f"search-concurrency must be between 1 and {_MAX_SEARCH_CONCURRENCY}."
        )
    research_only = bool(getattr(args, "research_only", False))
    dataset_path = args.dataset.resolve()
    task_spec_path = args.task_spec.resolve()
    dataset = _load_dataset(dataset_path)
    tasks = _load_batch_tasks(task_spec_path, dataset)
    experiment_id = _validate_label(args.experiment_id, name="experiment-id")
    phase = _validate_label(args.phase, name="phase")
    script_config = replace(
        settings.script_generation,
        grounding_review_enabled=False,
        generation_mode=(
            getattr(args, "generation_mode", None)
            or settings.script_generation.generation_mode
        ),
    )
    editorial_mode = script_config.generation_mode == "editorial_candidates"
    script_prompt_version = (
        None
        if research_only
        else BACKGROUND_SCRIPT_PIPELINE_VERSION
        if editorial_mode
        else BACKGROUND_SCRIPT_GENERATION_PROMPT_VERSION
    )

    output_dir = args.output_dir.resolve()
    _prepare_output_dir(output_dir)
    manifest_path = output_dir / "manifest.json"
    dataset_sha256 = _sha256(dataset_path)
    task_spec_sha256 = _sha256(task_spec_path)
    task_records: list[dict[str, Any]] = [
        {
            "task_id": batch_task.task_id,
            "dataset_index": batch_task.dataset_index,
            "topic": batch_task.task.topic,
            "target_length": batch_task.task.target_length,
            "status": "pending",
            "research_status": None,
            "research_snapshot": None,
            "trace": None,
            "error_type": None,
            "error": None,
        }
        for batch_task in tasks
    ]
    manifest: dict[str, Any] = {
        "experiment_id": experiment_id,
        "phase": phase,
        "mode": (
            "live_search_background_collection"
            if research_only
            else "live_search_background_script_generation"
        ),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "task_spec": str(task_spec_path),
        "task_spec_sha256": task_spec_sha256,
        "execution": {
            "task_concurrency": task_concurrency,
            "hy3_concurrency": request_concurrency,
            "search_concurrency": search_concurrency,
            "grounding_review_enabled": False,
        },
        "request_concurrency": request_concurrency,
        "research_only": research_only,
        "prompt_versions": {
            "research_query_plan": RESEARCH_QUERY_PLAN_PROMPT_VERSION,
            "background_selection": BACKGROUND_SELECTION_VERSION,
            "script_generation": script_prompt_version,
            "script_candidate": (
                BACKGROUND_SCRIPT_CANDIDATE_PROMPT_VERSION
                if editorial_mode and not research_only
                else None
            ),
            "script_editor": (
                BACKGROUND_SCRIPT_EDITOR_PROMPT_VERSION
                if editorial_mode and not research_only
                else None
            ),
            "script_grounding_review": None,
        },
        "system_prompt_sha256": {
            "research_query_plan": _text_sha256(RESEARCH_QUERY_PLAN_SYSTEM_PROMPT),
            "background_selection": None,
            "script_generation": (
                None
                if research_only
                else _text_sha256(
                    BACKGROUND_SCRIPT_CANDIDATE_SYSTEM_PROMPT
                    + "\n"
                    + BACKGROUND_SCRIPT_EDITOR_SYSTEM_PROMPT
                    if editorial_mode
                    else BACKGROUND_SCRIPT_GENERATION_SYSTEM_PROMPT
                )
            ),
            "script_candidate": (
                _text_sha256(BACKGROUND_SCRIPT_CANDIDATE_SYSTEM_PROMPT)
                if editorial_mode and not research_only
                else None
            ),
            "script_editor": (
                _text_sha256(BACKGROUND_SCRIPT_EDITOR_SYSTEM_PROMPT)
                if editorial_mode and not research_only
                else None
            ),
            "script_grounding_review": None,
        },
        "hy3": {
            "model": settings.hy3.model,
            "temperature": settings.hy3.temperature,
            "top_p": settings.hy3.top_p,
        },
        "research_config": asdict(settings.research),
        "script_generation_config": asdict(script_config),
        "tasks": task_records,
        "counts": _counts(task_records, input_count=len(tasks)),
    }
    _write_json(manifest_path, manifest, replace=False)
    manifest_lock = asyncio.Lock()
    task_semaphore = asyncio.Semaphore(task_concurrency)
    request_semaphore = asyncio.Semaphore(request_concurrency)
    search_semaphore = asyncio.Semaphore(search_concurrency)

    async def checkpoint(position: int, record: dict[str, Any]) -> None:
        """Replace one ordered task record and atomically persist the manifest."""

        async with manifest_lock:
            manifest["tasks"][position] = dict(record)
            manifest["counts"] = _counts(
                manifest["tasks"],
                input_count=len(tasks),
            )
            _write_json(manifest_path, manifest, replace=True)

    async with (
        AsyncHy3Client(settings.hy3) as llm,
        AsyncTavilySearchProvider(settings.tavily) as search,
    ):
        research_agent = ResearchAgent(llm, search, config=settings.research)
        if hasattr(research_agent, "_request_semaphore"):
            research_agent._request_semaphore = request_semaphore
        if hasattr(research_agent, "_search_semaphore"):
            research_agent._search_semaphore = search_semaphore
        script_agent = ScriptAgent(llm, config=script_config)
        if hasattr(script_agent, "_request_semaphore"):
            script_agent._request_semaphore = request_semaphore

        async def run_task(position: int, batch_task: BatchTask) -> None:
            task = batch_task.task
            record = dict(task_records[position])
            async with task_semaphore:
                record["status"] = "running"
                await checkpoint(position, record)
                display_position = position + 1
                _print_console(
                    f"[{display_position}/{len(tasks)}] START {batch_task.task_id} "
                    f"length={task.target_length} {task.topic}"
                )

                research: ResearchOutcome | None = None
                script: ScriptArtifact | None = None
                try:
                    collect_background = getattr(
                        research_agent,
                        "collect_background",
                        None,
                    ) or research_agent.research
                    research = await collect_background(task)
                    research_path = (
                        output_dir / "research" / f"{batch_task.task_id}.json"
                    )
                    _write_json(research_path, asdict(research), replace=False)
                    record.update(
                        {
                            "research_status": research.status,
                            "research_snapshot": str(research_path),
                            "research_errors": _safe_research_errors(research),
                            "usage": _usage_payload(research, None),
                        }
                    )
                    await checkpoint(position, record)
                    if research.status != "ready":
                        record["status"] = "insufficient_evidence"
                        _print_console(
                            f"[{display_position}/{len(tasks)}] STOP "
                            f"{batch_task.task_id} insufficient_evidence"
                        )
                        return

                    if research_only:
                        record["status"] = "completed"
                        _print_console(
                            f"[{display_position}/{len(tasks)}] DONE "
                            f"{batch_task.task_id} background-only"
                        )
                        return

                    script = await script_agent.generate(task, research)
                    trace = build_generation_trace(
                        task,
                        research,
                        script,
                        config={
                            "research": asdict(settings.research),
                            "script_generation": asdict(script_config),
                            "experiment": {
                                "experiment_id": experiment_id,
                                "phase": phase,
                                "mode": "live_search_background_script_generation",
                                "task_id": batch_task.task_id,
                                "dataset_index": batch_task.dataset_index,
                                "dataset_sha256": dataset_sha256,
                                "task_spec_sha256": task_spec_sha256,
                                "system_prompt_sha256": manifest[
                                    "system_prompt_sha256"
                                ],
                                "hy3_model": settings.hy3.model,
                                "hy3_temperature": settings.hy3.temperature,
                                "hy3_top_p": settings.hy3.top_p,
                            },
                        },
                    )
                    trace_path = (
                        output_dir
                        / "traces"
                        / f"{batch_task.task_id}-{trace.run_id}.json"
                    )
                    trace.write_json(trace_path)
                    record.update(
                        {
                            "status": "completed",
                            "trace": str(trace_path),
                            "run_id": trace.run_id,
                            "character_count": script.character_count,
                            "reference_ids": list(script.reference_ids),
                            "generation_attempt_count": (
                                script.generation_attempt_count
                            ),
                            "generation_mode": script.generation_mode,
                            "candidate_count": len(script.generation_candidates),
                            "editor_attempt_count": script.editor_attempt_count,
                            "length_within_tolerance": (
                                script.length_within_tolerance
                            ),
                            "length_repair_attempted": (
                                script.length_repair_attempted
                            ),
                            "usage": _usage_payload(research, script),
                        }
                    )
                    _print_console(
                        f"[{display_position}/{len(tasks)}] DONE "
                        f"{batch_task.task_id} chars={script.character_count} "
                        f"attempts={script.generation_attempt_count}"
                    )
                except Exception as exc:
                    safe_message = _safe_error_message(exc)
                    record.update(
                        {
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error": safe_message,
                        }
                    )
                    if research is None and isinstance(exc, ResearchGenerationError):
                        record["usage"] = _usage_payload(exc, None)
                    elif research is not None:
                        script_error = (
                            exc if isinstance(exc, ScriptGenerationError) else None
                        )
                        record["usage"] = _usage_payload(
                            research,
                            script,
                            script_error,
                        )
                        if script_error is not None:
                            record.update(
                                {
                                    "generation_attempt_count": (
                                        script_error.generation_attempt_count
                                    ),
                                    "grounding_review_attempt_count": (
                                        script_error.grounding_review_attempt_count
                                    ),
                                }
                            )
                    _print_console(
                        f"[{display_position}/{len(tasks)}] FAIL "
                        f"{batch_task.task_id} {type(exc).__name__}: "
                        f"{safe_message}"
                    )
                finally:
                    await checkpoint(position, record)

        await asyncio.gather(
            *(run_task(position, task) for position, task in enumerate(tasks))
        )

    async with manifest_lock:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["counts"] = _counts(
            manifest["tasks"],
            input_count=len(tasks),
        )
        _write_json(manifest_path, manifest, replace=True)
    _print_console(
        json.dumps(
            {"manifest": str(manifest_path), "counts": manifest["counts"]},
            ensure_ascii=False,
        )
    )
    return 0 if all(item["status"] == "completed" for item in manifest["tasks"]) else 1


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(_run(args))
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    raise SystemExit(main())
