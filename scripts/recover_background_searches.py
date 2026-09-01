"""Recover provider-failed searches without regenerating frozen query plans."""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from hyscript.agent import ResearchAgent
from hyscript.artifacts import load_research_outcome
from hyscript.config import settings
from hyscript.search import AsyncTavilySearchProvider


class _UnusedLLM:
    async def complete(self, *_: object, **__: object) -> object:
        raise AssertionError("Background search recovery must not call Hy3.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retry only provider-failed searches from frozen background snapshots. "
            "Successful queries and Hy3 query plans are never rerun."
        )
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--task-id", action="append", required=True)
    return parser


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load JSON: {path}") from exc


def _write_json(path: Path, payload: Any) -> None:
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


def _prepare_output_dir(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _counts(tasks: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "input": len(tasks),
        "accounted": len(tasks),
        "pending": 0,
        "completed": sum(task.get("status") == "completed" for task in tasks),
        "insufficient_evidence": sum(
            task.get("status") == "insufficient_evidence" for task in tasks
        ),
        "failed": sum(task.get("status") == "failed" for task in tasks),
        "running": 0,
    }


async def _run(args: argparse.Namespace) -> int:
    source_manifest_path = args.source_manifest.resolve()
    source_manifest = _read_json(source_manifest_path)
    if not isinstance(source_manifest, dict) or not isinstance(
        source_manifest.get("tasks"), list
    ):
        raise ValueError("Source manifest must contain a tasks list.")
    requested = set(args.task_id)
    if not requested or len(requested) != len(args.task_id):
        raise ValueError("task-id values must be unique and non-empty.")
    tasks = deepcopy(source_manifest["tasks"])
    task_ids = {
        task.get("task_id")
        for task in tasks
        if isinstance(task, dict) and isinstance(task.get("task_id"), str)
    }
    missing = sorted(requested - task_ids)
    if missing:
        raise ValueError(f"Unknown task ids: {', '.join(missing)}")

    output_dir = args.output_dir.resolve()
    _prepare_output_dir(output_dir)
    started_at = datetime.now(timezone.utc).isoformat()
    manifest = deepcopy(source_manifest)
    manifest.update(
        {
            "phase": args.phase,
            "mode": "live_search_background_collection_recovery",
            "started_at": started_at,
            "completed_at": None,
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": _sha256(source_manifest_path),
            "recovery_task_ids": sorted(requested),
            "tasks": tasks,
        }
    )
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    async with AsyncTavilySearchProvider(settings.tavily) as search:
        agent = ResearchAgent(_UnusedLLM(), search, config=settings.research)  # type: ignore[arg-type]
        for task in tasks:
            if not isinstance(task, dict):
                raise ValueError("Source manifest contains an invalid task.")
            task_id = task.get("task_id")
            snapshot_value = task.get("research_snapshot")
            if not isinstance(task_id, str) or not isinstance(snapshot_value, str):
                raise ValueError("Source task lacks task_id/research_snapshot.")
            source_snapshot = Path(snapshot_value).resolve()
            output_snapshot = output_dir / "research" / f"{task_id}.json"
            if task_id not in requested:
                output_snapshot.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_snapshot, output_snapshot)
                os.chmod(output_snapshot, 0o600)
                task["research_snapshot"] = str(output_snapshot)
                continue

            frozen = load_research_outcome(source_snapshot)
            recovered = await agent.retry_failed_background_searches(frozen)
            _write_json(output_snapshot, asdict(recovered))
            usage = dict(task.get("usage") or {})
            usage.update(
                {
                    "research_hy3_attempted_calls": recovered.llm_request_count,
                    "tavily_attempted_calls": recovered.search_request_count,
                    "tavily_succeeded_calls": len(recovered.search_responses),
                }
            )
            task.update(
                {
                    "status": (
                        "completed"
                        if recovered.status == "ready"
                        else "insufficient_evidence"
                    ),
                    "research_status": recovered.status,
                    "research_snapshot": str(output_snapshot),
                    "research_errors": list(recovered.errors),
                    "usage": usage,
                }
            )
            print(
                f"RECOVER {task_id} status={recovered.status} "
                f"search_attempts={recovered.search_request_count}",
                flush=True,
            )

    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["counts"] = _counts(tasks)
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {"manifest": str(manifest_path), "counts": manifest["counts"]},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if manifest["counts"]["completed"] == len(tasks) else 1


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(_run(args))
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    raise SystemExit(main())
