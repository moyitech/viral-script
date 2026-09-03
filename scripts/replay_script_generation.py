"""Regenerate scripts from frozen research snapshots without new search calls."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from hyscript.agent import (
    ScriptAgent,
    ScriptCandidate,
    ScriptGenerationError,
    ScriptTask,
)
from hyscript.artifacts import build_generation_trace, load_research_outcome
from hyscript.config import PROJECT_ROOT, settings
from hyscript.llm import AsyncHy3Client, summarize_token_usage
from hyscript.llm.prompts import (
    BACKGROUND_SCRIPT_CANDIDATE_PROMPT_VERSION,
    BACKGROUND_SCRIPT_CANDIDATE_SYSTEM_PROMPT,
    BACKGROUND_SCRIPT_EDITOR_PROMPT_VERSION,
    BACKGROUND_SCRIPT_EDITOR_SYSTEM_PROMPT,
    BACKGROUND_SCRIPT_GENERATION_PROMPT_VERSION,
    BACKGROUND_SCRIPT_GENERATION_SYSTEM_PROMPT,
    BACKGROUND_SCRIPT_FORMAT_REPAIR_PROMPT_VERSION,
    BACKGROUND_SCRIPT_FORMAT_REPAIR_SYSTEM_PROMPT,
    BACKGROUND_SCRIPT_PIPELINE_VERSION,
    SCRIPT_GROUNDING_REVIEW_PROMPT_VERSION,
    SCRIPT_GROUNDING_REVIEW_SYSTEM_PROMPT,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay script generation from frozen research snapshots. "
            "This command never performs search."
        )
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Experiment id written to replay manifests and frozen traces.",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Replay only this task id; repeat for multiple tasks.",
    )
    parser.add_argument(
        "--output-task-id",
        action="append",
        default=[],
        help=(
            "After target-length expansion, generate only this exact output task id; "
            "repeat for resume-safe partial attempts."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        choices=range(1, 513),
        default=1,
        metavar="1-512",
    )
    parser.add_argument(
        "--request-concurrency",
        type=int,
        choices=range(1, 513),
        default=16,
        metavar="1-512",
        help="Maximum Hy3 script requests in flight across all tasks.",
    )
    parser.add_argument(
        "--generation-mode",
        choices=("single", "single_shot", "editorial_candidates"),
        default=None,
        help="Override the configured background-script generation mode.",
    )
    parser.add_argument(
        "--reuse-candidates-from-manifest",
        type=Path,
        default=None,
        help=(
            "Re-run only the chief-editor stage using candidates frozen in this "
            "editorial replay manifest."
        ),
    )
    parser.add_argument(
        "--target-length",
        action="append",
        type=int,
        default=[],
        help=(
            "Override and expand every selected topic to this target length; "
            "repeat the option to build a length matrix."
        ),
    )
    parser.add_argument(
        "--grounding-review",
        action="store_true",
        help="Run one evidence-boundary review after each valid draft.",
    )
    parser.add_argument(
        "--require-grounding-review-accepted",
        action="store_true",
        help=(
            "Formal-run gate: enable grounding review and fail any task whose "
            "review outcome is not accepted. The trace is still retained."
        ),
    )
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load source manifest: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise ValueError("Source manifest must contain a tasks list.")
    return payload


def _source_tasks(
    manifest: dict[str, Any],
    requested_ids: Sequence[str],
) -> list[dict[str, Any]]:
    requested = set(requested_ids)
    tasks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_task in manifest["tasks"]:
        if not isinstance(raw_task, dict):
            raise ValueError("Source manifest contains an invalid task.")
        task_id = raw_task.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("Source manifest task_id must be a non-empty string.")
        if task_id in seen_ids:
            raise ValueError(f"Source manifest repeats task_id: {task_id}")
        seen_ids.add(task_id)
        if requested and task_id not in requested:
            continue
        has_replayable_research = (
            raw_task.get("status") == "completed"
            or raw_task.get("research_status") == "ready"
        )
        if not has_replayable_research:
            continue
        if not isinstance(raw_task.get("research_snapshot"), str):
            raise ValueError(f"Replayable task {task_id} has no research snapshot.")
        tasks.append(raw_task)
    missing = sorted(requested - seen_ids)
    if missing:
        raise ValueError(f"Unknown task ids: {', '.join(missing)}")
    if not tasks:
        raise ValueError("No source tasks with ready research were selected for replay.")
    return tasks


def _expand_target_lengths(
    tasks: Sequence[dict[str, Any]],
    target_lengths: Sequence[int],
) -> list[dict[str, Any]]:
    """Expand replay inputs while keeping arbitrary product lengths supported."""

    if not target_lengths:
        return [dict(task) for task in tasks]
    lengths = tuple(target_lengths)
    if len(set(lengths)) != len(lengths):
        raise ValueError("target-length values must not repeat.")
    for length in lengths:
        try:
            ScriptTask(topic="测试选题", target_length=length)
        except ValueError as exc:
            raise ValueError(f"Invalid target-length {length}: {exc}") from exc
    expanded: list[dict[str, Any]] = []
    for raw_task in tasks:
        for length in lengths:
            expanded.append(
                {
                    **raw_task,
                    "source_task_id": raw_task["task_id"],
                    "task_id": f"{raw_task['task_id']}-L{length}",
                    "target_length": length,
                }
            )
    return expanded


def _candidate_trace_map(manifest_path: Path) -> dict[str, Path]:
    manifest = _load_manifest(manifest_path)
    traces: dict[str, Path] = {}
    for raw_task in manifest["tasks"]:
        if not isinstance(raw_task, dict) or raw_task.get("status") != "completed":
            continue
        task_id = raw_task.get("task_id")
        trace = raw_task.get("trace")
        if not isinstance(task_id, str) or not isinstance(trace, str):
            raise ValueError("Candidate source manifest contains an invalid task.")
        if task_id in traces:
            raise ValueError(f"Candidate source repeats task id: {task_id}")
        traces[task_id] = Path(trace).resolve()
    if not traces:
        raise ValueError("Candidate source manifest contains no completed traces.")
    return traces


def _load_frozen_candidates(path: Path) -> tuple[ScriptCandidate, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load candidate source trace: {path}") from exc
    artifact = payload.get("script_artifact") if isinstance(payload, dict) else None
    raw_candidates = (
        artifact.get("generation_candidates")
        if isinstance(artifact, dict)
        else None
    )
    if not isinstance(raw_candidates, list) or len(raw_candidates) != 3:
        raise ValueError(f"Candidate source trace does not contain three drafts: {path}")
    candidates: list[ScriptCandidate] = []
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            raise ValueError(f"Candidate source trace contains an invalid draft: {path}")
        try:
            candidate = ScriptCandidate(
                candidate_id=raw_candidate["candidate_id"],
                strategy=raw_candidate["strategy"],
                outline=tuple(raw_candidate["outline"]),
                script_text=raw_candidate["script_text"],
                reference_ids=tuple(raw_candidate["reference_ids"]),
                character_count=raw_candidate["character_count"],
                prompt_version=raw_candidate["prompt_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Candidate source trace contains an invalid draft: {path}"
            ) from exc
        if candidate.prompt_version != BACKGROUND_SCRIPT_CANDIDATE_PROMPT_VERSION:
            raise ValueError(
                "Frozen candidate prompt version differs from the current candidate "
                "prompt; editor-only attribution would be invalid."
            )
        candidates.append(candidate)
    return tuple(candidates)


def _script_usage(script: Any) -> dict[str, int]:
    summary = summarize_token_usage(script.llm_usages)
    generation_attempts = script.generation_attempt_count
    content_generation_attempts = getattr(
        script,
        "content_generation_attempt_count",
        generation_attempts,
    )
    format_repair_attempts = getattr(script, "format_repair_attempt_count", 0)
    review_attempts = getattr(script, "grounding_review_attempt_count", 0)
    final_rewrite_attempts = getattr(script, "final_rewrite_attempt_count", 0)
    return {
        "attempted_calls": (
            generation_attempts + review_attempts + final_rewrite_attempts
        ),
        "generation_attempted_calls": generation_attempts,
        "content_generation_attempted_calls": content_generation_attempts,
        "format_repair_attempted_calls": format_repair_attempts,
        "grounding_review_attempted_calls": review_attempts,
        "final_rewrite_attempted_calls": final_rewrite_attempts,
        "reported_calls": summary.reported_call_count,
        "input_tokens": summary.input_tokens,
        "output_tokens": summary.output_tokens,
        "total_tokens": summary.total_tokens,
        "reasoning_tokens": summary.reasoning_tokens,
        "cached_input_tokens": summary.cached_input_tokens,
    }


def _mark_frozen_research_replay(trace: Any, research: Any, script: Any) -> None:
    """Keep frozen research in the trace while counting only calls made by replay."""

    request_counts = trace.config.get("request_counts", {})
    source_counts = {
        "research_llm": research.llm_request_count,
        "tavily_attempted": research.search_request_count,
        "tavily_succeeded": len(research.search_responses),
        "tavily_failed": research.search_request_count
        - len(research.search_responses),
    }
    trace.config["source_research_request_counts"] = source_counts
    script_calls = (
        script.generation_attempt_count
        + script.grounding_review_attempt_count
        + script.final_rewrite_attempt_count
    )
    request_counts.update(
        {
            "research_llm": 0,
            "search": 0,
            "hy3_total": script_calls,
            "tavily_attempted": 0,
            "tavily_succeeded": 0,
            "tavily_failed": 0,
        }
    )
    trace.config["request_counts"] = request_counts

    source_usage = summarize_token_usage(research.llm_usages)
    trace.config["source_research_token_usage"] = {
        "hy3_reported_call_count": source_usage.reported_call_count,
        "hy3_input_tokens": source_usage.input_tokens,
        "hy3_output_tokens": source_usage.output_tokens,
        "hy3_total_tokens": source_usage.total_tokens,
        "hy3_reasoning_tokens": source_usage.reasoning_tokens,
        "hy3_cached_input_tokens": source_usage.cached_input_tokens,
    }
    script_usage = summarize_token_usage(script.llm_usages)
    trace.token_usage = {
        "hy3_reported_call_count": script_usage.reported_call_count,
        "hy3_input_tokens": script_usage.input_tokens,
        "hy3_output_tokens": script_usage.output_tokens,
        "hy3_total_tokens": script_usage.total_tokens,
        "hy3_reasoning_tokens": script_usage.reasoning_tokens,
        "hy3_cached_input_tokens": script_usage.cached_input_tokens,
    }
    source_latency = trace.latency.get("search_response_time_sum", 0)
    trace.latency["source_search_response_time_sum"] = source_latency
    trace.latency["search_response_time_sum"] = 0
    all_calls = trace.lineage.get("llm_calls", [])
    trace.lineage["source_research_llm_calls"] = [
        item
        for item in all_calls
        if isinstance(item, dict)
        and str(item.get("stage", "")).startswith("research.")
    ]
    trace.lineage["llm_calls"] = [
        item
        for item in all_calls
        if not (
            isinstance(item, dict)
            and str(item.get("stage", "")).startswith("research.")
        )
    ]


async def _run(args: argparse.Namespace) -> int:
    if not 1 <= args.concurrency <= 512:
        raise ValueError("concurrency must be between 1 and 512.")
    request_concurrency = getattr(args, "request_concurrency", 16)
    if not 1 <= request_concurrency <= 512:
        raise ValueError("request-concurrency must be between 1 and 512.")
    source_manifest_path = args.source_manifest.resolve()
    source_manifest = _load_manifest(source_manifest_path)
    source_tasks = _source_tasks(source_manifest, args.task_id)
    tasks = _expand_target_lengths(
        source_tasks,
        getattr(args, "target_length", ()),
    )
    requested_output_ids = set(getattr(args, "output_task_id", ()))
    if requested_output_ids:
        known_output_ids = {task["task_id"] for task in tasks}
        unknown_output_ids = sorted(requested_output_ids - known_output_ids)
        if unknown_output_ids:
            raise ValueError(
                "Unknown output task ids: " + ", ".join(unknown_output_ids)
            )
        tasks = [task for task in tasks if task["task_id"] in requested_output_ids]
    candidate_source_manifest_arg = getattr(
        args,
        "reuse_candidates_from_manifest",
        None,
    )
    candidate_source_manifest_path = (
        candidate_source_manifest_arg.resolve()
        if candidate_source_manifest_arg is not None
        else None
    )
    candidate_trace_by_task = (
        _candidate_trace_map(candidate_source_manifest_path)
        if candidate_source_manifest_path is not None
        else {}
    )
    if candidate_trace_by_task:
        expected_task_ids = {task["task_id"] for task in tasks}
        if set(candidate_trace_by_task) != expected_task_ids:
            raise ValueError(
                "Candidate source task ids must exactly match the replay task matrix."
            )
    output_dir = args.output_dir.resolve()
    require_review_accepted = bool(
        getattr(args, "require_grounding_review_accepted", False)
    )
    generation_mode = (
        getattr(args, "generation_mode", None)
        or settings.script_generation.generation_mode
    )
    single_shot_mode = generation_mode == "single_shot"
    script_config = replace(
        settings.script_generation,
        grounding_review_enabled=(
            False
            if single_shot_mode
            else (
                settings.script_generation.grounding_review_enabled
                or args.grounding_review
                or require_review_accepted
            )
        ),
        generation_mode=generation_mode,
        final_rewrite_enabled=(
            False
            if single_shot_mode
            else settings.script_generation.final_rewrite_enabled
        ),
    )
    if candidate_trace_by_task and script_config.generation_mode != "editorial_candidates":
        raise ValueError(
            "Frozen candidates can only be reused in editorial_candidates mode."
        )
    editorial_mode = script_config.generation_mode == "editorial_candidates"
    script_prompt_version = (
        BACKGROUND_SCRIPT_PIPELINE_VERSION
        if editorial_mode
        else BACKGROUND_SCRIPT_GENERATION_PROMPT_VERSION
    )
    script_prompt_text = (
        BACKGROUND_SCRIPT_CANDIDATE_SYSTEM_PROMPT
        + "\n"
        + BACKGROUND_SCRIPT_EDITOR_SYSTEM_PROMPT
        if editorial_mode
        else BACKGROUND_SCRIPT_GENERATION_SYSTEM_PROMPT
    )
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise ValueError(f"Output manifest already exists: {manifest_path}")

    experiment_id = getattr(args, "experiment_id", None) or source_manifest.get(
        "experiment_id"
    )
    manifest: dict[str, Any] = {
        "experiment_id": experiment_id,
        "phase": args.phase,
        "mode": (
            "frozen_research_candidate_editor_replay"
            if candidate_trace_by_task
            else (
                "frozen_research_single_shot_replay"
                if single_shot_mode
                else "frozen_research_script_replay"
            )
        ),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "candidate_source_manifest": (
            str(candidate_source_manifest_path)
            if candidate_source_manifest_path is not None
            else None
        ),
        "candidate_source_manifest_sha256": (
            _sha256(candidate_source_manifest_path)
            if candidate_source_manifest_path is not None
            else None
        ),
        "search_request_count": 0,
        "new_tavily_request_count": 0,
        "source_tavily_concurrency": 8,
        "require_grounding_review_accepted": require_review_accepted,
        "test_target_lengths": list(getattr(args, "target_length", ())),
        "script_prompt_version": script_prompt_version,
        "script_system_prompt_sha256": hashlib.sha256(
            script_prompt_text.encode("utf-8")
        ).hexdigest(),
        "script_format_repair_prompt_version": (
            BACKGROUND_SCRIPT_FORMAT_REPAIR_PROMPT_VERSION
            if single_shot_mode
            else None
        ),
        "script_format_repair_prompt_sha256": (
            hashlib.sha256(
                BACKGROUND_SCRIPT_FORMAT_REPAIR_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest()
            if single_shot_mode
            else None
        ),
        "script_candidate_prompt_version": (
            BACKGROUND_SCRIPT_CANDIDATE_PROMPT_VERSION if editorial_mode else None
        ),
        "script_candidate_prompt_sha256": (
            hashlib.sha256(
                BACKGROUND_SCRIPT_CANDIDATE_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest()
            if editorial_mode
            else None
        ),
        "script_editor_prompt_version": (
            BACKGROUND_SCRIPT_EDITOR_PROMPT_VERSION if editorial_mode else None
        ),
        "script_editor_prompt_sha256": (
            hashlib.sha256(
                BACKGROUND_SCRIPT_EDITOR_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest()
            if editorial_mode
            else None
        ),
        "script_grounding_review_prompt_version": (
            SCRIPT_GROUNDING_REVIEW_PROMPT_VERSION
            if script_config.grounding_review_enabled
            else None
        ),
        "script_grounding_review_prompt_sha256": (
            hashlib.sha256(
                SCRIPT_GROUNDING_REVIEW_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest()
            if script_config.grounding_review_enabled
            else None
        ),
        "hy3": {
            "model": settings.hy3.model,
            "temperature": settings.hy3.temperature,
            "top_p": settings.hy3.top_p,
        },
        "script_generation_config": asdict(script_config),
        "execution": {
            "task_concurrency": args.concurrency,
            "hy3_concurrency": request_concurrency,
        },
        "request_concurrency": request_concurrency,
        "tasks": [],
    }
    _write_json(manifest_path, manifest)
    semaphore = asyncio.Semaphore(args.concurrency)
    request_semaphore = asyncio.Semaphore(request_concurrency)

    async with AsyncHy3Client(settings.hy3) as llm:
        async def replay(raw_task: dict[str, Any]) -> dict[str, Any]:
            task_id = raw_task["task_id"]
            result: dict[str, Any] = {
                "task_id": task_id,
                "source_task_id": raw_task.get("source_task_id", task_id),
                "dataset_index": raw_task.get("dataset_index"),
                "topic": raw_task.get("topic"),
                "target_length": raw_task.get("target_length"),
                "status": "running",
                "source_research_snapshot": raw_task["research_snapshot"],
                "trace": None,
                "error_type": None,
                "error": None,
            }
            print(f"START {task_id} {result['topic']}", flush=True)
            try:
                topic = result["topic"]
                target_length = result["target_length"]
                if not isinstance(topic, str) or not isinstance(target_length, int):
                    raise ValueError(f"Task {task_id} has invalid topic or target length.")
                research_path = Path(raw_task["research_snapshot"])
                if not research_path.is_absolute():
                    research_path = source_manifest_path.parent / research_path
                research_path = research_path.resolve()
                research = load_research_outcome(research_path)
                if research.status != "ready":
                    raise ValueError(f"Task {task_id} research snapshot is not ready.")
                task = ScriptTask(topic=topic, target_length=target_length)
                async with semaphore:
                    script_agent = ScriptAgent(llm, config=script_config)
                    if hasattr(script_agent, "_request_semaphore"):
                        script_agent._request_semaphore = request_semaphore
                    if candidate_trace_by_task:
                        frozen_candidates = _load_frozen_candidates(
                            candidate_trace_by_task[task_id]
                        )
                        script = await script_agent.edit_background_candidates(
                            task,
                            research,
                            frozen_candidates,
                        )
                    else:
                        script = await script_agent.generate(task, research)
                trace = build_generation_trace(
                    task,
                    research,
                    script,
                    config={
                        "research": source_manifest.get("research_config", {}),
                        "script_generation": asdict(script_config),
                        "experiment": {
                            "experiment_id": experiment_id,
                            "phase": args.phase,
                            "mode": manifest["mode"],
                            "task_id": task_id,
                            "source_manifest_sha256": manifest[
                                "source_manifest_sha256"
                            ],
                            "source_research_snapshot": str(research_path),
                            "source_research_snapshot_sha256": _sha256(research_path),
                            "source_candidate_trace": (
                                str(candidate_trace_by_task[task_id])
                                if candidate_trace_by_task
                                else None
                            ),
                            "source_candidate_trace_sha256": (
                                _sha256(candidate_trace_by_task[task_id])
                                if candidate_trace_by_task
                                else None
                            ),
                            "search_requests_in_replay": 0,
                            "new_tavily_request_count": 0,
                            "source_tavily_concurrency": 8,
                            "script_system_prompt_sha256": manifest[
                                "script_system_prompt_sha256"
                            ],
                            "script_candidate_prompt_sha256": manifest[
                                "script_candidate_prompt_sha256"
                            ],
                            "script_editor_prompt_sha256": manifest[
                                "script_editor_prompt_sha256"
                            ],
                            "script_grounding_review_prompt_sha256": manifest[
                                "script_grounding_review_prompt_sha256"
                            ],
                            "script_format_repair_prompt_sha256": manifest[
                                "script_format_repair_prompt_sha256"
                            ],
                            "hy3_model": settings.hy3.model,
                            "hy3_temperature": settings.hy3.temperature,
                            "hy3_top_p": settings.hy3.top_p,
                        },
                    },
                )
                _mark_frozen_research_replay(trace, research, script)
                trace_path = output_dir / "traces" / f"{task_id}-{trace.run_id}.json"
                trace.write_json(trace_path)
                review_gate_failed = (
                    require_review_accepted
                    and script.grounding_review_status != "accepted"
                )
                result.update(
                    {
                        "status": "failed" if review_gate_failed else "completed",
                        "trace": str(trace_path),
                        "run_id": trace.run_id,
                        "character_count": script.character_count,
                        "generation_attempt_count": script.generation_attempt_count,
                        "content_generation_attempt_count": (
                            script.content_generation_attempt_count
                        ),
                        "format_repair_attempt_count": (
                            script.format_repair_attempt_count
                        ),
                        "generation_mode": script.generation_mode,
                        "candidate_count": len(script.generation_candidates),
                        "editor_attempt_count": script.editor_attempt_count,
                        "length_within_tolerance": script.length_within_tolerance,
                        "length_repair_attempted": script.length_repair_attempted,
                        "grounding_review_attempt_count": (
                            script.grounding_review_attempt_count
                        ),
                        "grounding_review_status": script.grounding_review_status,
                        "grounding_review_issues": list(
                            script.grounding_review_issues
                        ),
                        "grounding_review_failure_reason": (
                            script.grounding_review_failure_reason
                        ),
                        "script_usage": _script_usage(script),
                    }
                )
                if review_gate_failed:
                    result.update(
                        {
                            "error_type": "GroundingReviewNotAccepted",
                            "error": (
                                "Formal result requires grounding review status "
                                f"accepted; got {script.grounding_review_status}."
                            ),
                        }
                    )
                    print(
                        f"FAIL {task_id} grounding_review="
                        f"{script.grounding_review_status}",
                        flush=True,
                    )
                    return result
                print(
                    f"DONE {task_id} chars={script.character_count} "
                    f"attempts={script.generation_attempt_count}",
                    flush=True,
                )
            except Exception as exc:
                result.update(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                if isinstance(exc, ScriptGenerationError):
                    result.update(
                        {
                            "generation_attempt_count": (
                                exc.generation_attempt_count
                            ),
                            "content_generation_attempt_count": (
                                exc.content_generation_attempt_count
                            ),
                            "format_repair_attempt_count": (
                                exc.format_repair_attempt_count
                            ),
                            "grounding_review_attempt_count": (
                                exc.grounding_review_attempt_count
                            ),
                            "script_usage": _script_usage(exc),
                        }
                    )
                print(f"FAIL {task_id} {type(exc).__name__}: {exc}", flush=True)
            return result

        results = await asyncio.gather(*(replay(task) for task in tasks))

    manifest["tasks"] = list(results)
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["counts"] = {
        "input": len(results),
        "completed": sum(item["status"] == "completed" for item in results),
        "failed": sum(item["status"] == "failed" for item in results),
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "counts": manifest["counts"]}))
    return 1 if manifest["counts"]["failed"] else 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(_run(args))
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    raise SystemExit(main())
