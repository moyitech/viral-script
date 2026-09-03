"""Run the paired 300-trace frozen-research single-shot experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from hyscript.config import PROJECT_ROOT
from hyscript.evaluation.end_to_end_formal import (
    DEFAULT_HY3_CONCURRENCY,
    DEFAULT_JUDGE_CONCURRENCY,
    DEFAULT_TASK_CONCURRENCY,
    export_e2e_report,
    generate_e2e_experiment,
    prepare_e2e_experiment,
    repeat_e2e_judge,
    score_e2e_experiment,
)


DEFAULT_EXPERIMENT_DIR = (
    PROJECT_ROOT / "eval/experiments/formal-100-e2e-single-shot-v1"
)
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "eval/experiments/formal-100-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "generate", "score", "repeat", "report", "run"),
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument(
        "--task-concurrency",
        type=int,
        choices=range(1, 513),
        default=DEFAULT_TASK_CONCURRENCY,
        metavar="1-512",
    )
    parser.add_argument(
        "--hy3-concurrency",
        type=int,
        choices=range(1, 513),
        default=DEFAULT_HY3_CONCURRENCY,
        metavar="1-512",
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        choices=range(1, 513),
        default=DEFAULT_JUDGE_CONCURRENCY,
        metavar="1-512",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    experiment_dir = args.experiment_dir.resolve()
    baseline_dir = args.baseline_dir.resolve()
    if args.command in {"prepare", "run"}:
        prepare_e2e_experiment(experiment_dir, baseline_dir=baseline_dir)
        print(f"prepared={experiment_dir}")
    if args.command in {"generate", "run"}:
        manifest = generate_e2e_experiment(
            experiment_dir,
            baseline_dir=baseline_dir,
            task_concurrency=args.task_concurrency,
            hy3_concurrency=args.hy3_concurrency,
        )
        print(f"generated={manifest['selected_count']}/{manifest['expected_count']}")
    if args.command in {"score", "run"}:
        score_e2e_experiment(
            experiment_dir,
            baseline_dir=baseline_dir,
            judge_concurrency=args.judge_concurrency,
        )
        print(f"scored={experiment_dir / 'results'}")
    if args.command in {"repeat", "run"}:
        stability = repeat_e2e_judge(
            experiment_dir,
            baseline_dir=baseline_dir,
            judge_concurrency=args.judge_concurrency,
        )
        print(f"repeat={stability['record_count']}")
    if args.command in {"report", "run"}:
        summary = export_e2e_report(experiment_dir, baseline_dir=baseline_dir)
        print(
            f"reported={experiment_dir / 'report/comparison.md'} "
            f"pairs={summary['paired']['pair_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
