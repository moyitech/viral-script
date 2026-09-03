"""Score frozen generation traces with rules and an optional Hy3 Judge."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

from hyscript.config import PROJECT_ROOT, SettingsError, get_settings
from hyscript.evaluation import (
    BatchEvaluationConfig,
    BatchEvaluationRunner,
    Hy3JudgeEvaluator,
    JudgeConfig,
    load_rubric,
)
from hyscript.llm import AsyncHy3Client


MAX_JUDGE_CONCURRENCY = 512


def _evaluator_names(value: str) -> tuple[str, ...]:
    names = tuple(name.strip().lower() for name in value.split(",") if name.strip())
    if not names or any(name not in {"rules", "judge"} for name in names):
        raise argparse.ArgumentTypeError(
            "evaluators must be a comma-separated subset of rules,judge"
        )
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("evaluator names must not repeat")
    return names


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "eval/results/runs" / f"{timestamp}-{uuid4().hex[:8]}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate already-frozen generation traces. The generation phase is "
            "not available until the existing-topic Agent workflow is implemented."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score", help="Score one trace or a trace directory.")
    source = score.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace", type=Path, help="One frozen trace JSON file.")
    source.add_argument(
        "--trace-dir",
        type=Path,
        action="append",
        help=(
            "Directory recursively containing frozen trace JSON files; repeat "
            "to combine a primary batch with retry traces."
        ),
    )
    source.add_argument(
        "--trace-manifest",
        type=Path,
        help=(
            "JSON manifest whose completed tasks name exact trace paths. "
            "Use this for immutable formal-experiment selections."
        ),
    )
    score.add_argument(
        "--rubric",
        type=Path,
        default=PROJECT_ROOT / "eval/rubrics/script_quality_v1.json",
    )
    score.add_argument(
        "--evaluators",
        type=_evaluator_names,
        default=("rules",),
        help="Comma-separated evaluators. Default: rules. Judge calls consume API quota.",
    )
    score.add_argument("--output-dir", type=Path, default=None)
    score.add_argument(
        "--concurrency",
        type=int,
        choices=range(1, MAX_JUDGE_CONCURRENCY + 1),
        default=2,
        metavar=f"1-{MAX_JUDGE_CONCURRENCY}",
    )
    score.add_argument("--overwrite", action="store_true")
    score.add_argument(
        "--reasoning-effort",
        choices=("no_think", "low", "high"),
        default="high",
        help="Hy3 Judge reasoning effort. Default: high.",
    )
    return parser


def _is_trace_candidate(path: Path) -> bool:
    """Keep traces and malformed trace candidates, but skip result/config JSON."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return True
    return (
        isinstance(payload, dict)
        and "run_id" in payload
        and not ("evaluation_id" in payload and "evaluator" in payload)
    )


def _trace_paths(args: argparse.Namespace, *, output_dir: Path) -> list[Path]:
    if args.trace is not None:
        return [args.trace]
    trace_manifest = getattr(args, "trace_manifest", None)
    if trace_manifest is not None:
        try:
            payload = json.loads(trace_manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not load trace manifest: {trace_manifest}") from exc
        tasks = payload.get("tasks") if isinstance(payload, dict) else None
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("Trace manifest must contain a non-empty tasks list.")
        paths: list[Path] = []
        for index, task in enumerate(tasks):
            if not isinstance(task, dict) or task.get("status") != "completed":
                raise ValueError(
                    f"Trace manifest task {index} must be a completed object."
                )
            raw_path = task.get("trace")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f"Trace manifest task {index} has no trace path.")
            path = Path(raw_path)
            if not path.is_absolute():
                path = trace_manifest.parent / path
            if not path.is_file():
                raise ValueError(f"Trace manifest path does not exist: {path}")
            paths.append(path.resolve())
        if len(set(paths)) != len(paths):
            raise ValueError("Trace manifest contains duplicate trace paths.")
        return paths
    raw_directories = args.trace_dir
    directories = (
        [raw_directories]
        if isinstance(raw_directories, Path)
        else list(raw_directories or [])
    )
    if not directories:
        raise ValueError("At least one trace directory is required.")
    for directory in directories:
        if not directory.is_dir():
            raise ValueError(f"Trace directory does not exist: {directory}")
    output_root = output_dir.resolve()
    paths = sorted(
        path
        for directory in directories
        for path in directory.rglob("*.json")
        if path.is_file()
        and not path.resolve().is_relative_to(output_root)
        and _is_trace_candidate(path)
    )
    if not paths:
        raise ValueError("Trace directories contain no JSON files.")
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Trace directories contain duplicate trace paths.")
    return paths


async def _run(args: argparse.Namespace) -> int:
    rubric = load_rubric(args.rubric)
    output_dir = (args.output_dir or _default_output_dir()).resolve()
    paths = _trace_paths(args, output_dir=output_dir)
    config = BatchEvaluationConfig(
        output_dir=output_dir,
        evaluators=args.evaluators,
        concurrency=args.concurrency,
        overwrite=args.overwrite,
    )

    if "judge" not in args.evaluators:
        result = await BatchEvaluationRunner(rubric, config).run(paths)
    else:
        # Judge sampling is isolated from the generation model defaults.
        hy3 = replace(get_settings().hy3, temperature=0.0, top_p=1.0)
        async with AsyncHy3Client(hy3) as client:
            judge = Hy3JudgeEvaluator(
                client,
                model_name=hy3.model,
                config=JudgeConfig(
                    reasoning_effort=args.reasoning_effort,
                ),
                sampling_parameters={
                    "temperature": hy3.temperature,
                    "top_p": hy3.top_p,
                },
            )
            result = await BatchEvaluationRunner(
                rubric,
                config,
                judge_evaluator=judge,
            ).run(paths)

    completed = sum(outcome.status == "completed" for outcome in result.outcomes)
    skipped = sum(outcome.status == "skipped" for outcome in result.outcomes)
    print(
        f"evaluation_id={result.evaluation_id} completed={completed} "
        f"skipped={skipped} failed={result.failed_count} output={result.output_dir}"
    )
    for outcome in result.outcomes:
        if outcome.status == "failed":
            print(
                f"{outcome.error_code}: {outcome.trace_path}: {outcome.message}",
                file=sys.stderr,
            )
    return 1 if result.failed_count else 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(_run(args))
    except (ValueError, OSError, RuntimeError, SettingsError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    raise SystemExit(main())
