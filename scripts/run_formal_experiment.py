"""Prepare and explicitly run the versioned 100-topic formal experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from hyscript.config import PROJECT_ROOT
from hyscript.evaluation.formal import (
    DEFAULT_HY3_CONCURRENCY,
    DEFAULT_JUDGE_CONCURRENCY,
    DEFAULT_SEARCH_CONCURRENCY,
    DEFAULT_TASK_CONCURRENCY,
    export_report,
    generate_experiment,
    prepare_experiment,
    score_experiment,
)


DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "eval/experiments/formal-100-v1"
DEFAULT_DATASET = PROJECT_ROOT / "eval/datasets/eval_topics_synthetic_v1.json"
DEFAULT_RUBRIC = PROJECT_ROOT / "eval/rubrics/script_quality_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Orchestrate the immutable formal experiment. Only generate, score, "
            "and run make external API calls."
        )
    )
    parser.add_argument("command", choices=("prepare", "generate", "score", "report", "run"))
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    parser.add_argument("--task-concurrency", type=int, default=DEFAULT_TASK_CONCURRENCY)
    parser.add_argument("--hy3-concurrency", type=int, default=DEFAULT_HY3_CONCURRENCY)
    parser.add_argument("--search-concurrency", type=int, default=DEFAULT_SEARCH_CONCURRENCY)
    parser.add_argument("--judge-concurrency", type=int, default=DEFAULT_JUDGE_CONCURRENCY)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    experiment_dir = args.experiment_dir.resolve()
    if args.command in {"prepare", "run"}:
        prepare_experiment(
            experiment_dir,
            dataset_path=args.dataset,
            rubric_path=args.rubric,
        )
        print(f"prepared={experiment_dir}")
    if args.command in {"generate", "run"}:
        manifest = generate_experiment(
            experiment_dir,
            task_concurrency=args.task_concurrency,
            hy3_concurrency=args.hy3_concurrency,
            search_concurrency=args.search_concurrency,
        )
        print(f"generated={manifest['selected_count']}/{manifest['expected_count']}")
    if args.command in {"score", "run"}:
        score_experiment(experiment_dir, judge_concurrency=args.judge_concurrency)
        print(f"scored={experiment_dir / 'results'}")
    if args.command in {"report", "run"}:
        summary = export_report(experiment_dir)
        print(
            f"reported={experiment_dir / 'report/report.md'} "
            f"records={summary['scored_records']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
