"""Run the versioned Hy3-versus-GPT-5.6-Luna Judge comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from hyscript.config import PROJECT_ROOT
from hyscript.evaluation.judge_comparison import (
    JUDGE_CONCURRENCY,
    export_comparison_report,
    prepare_comparison,
    repeat_comparison,
    score_comparison,
)


DEFAULT_EXPERIMENT_DIR = (
    PROJECT_ROOT / "eval/experiments/formal-100-judge-comparison-v1"
)
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "eval/experiments/formal-100-v1"
DEFAULT_SINGLE_SHOT_DIR = (
    PROJECT_ROOT / "eval/experiments/formal-100-e2e-single-shot-v1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "score", "repeat", "report", "run")
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument(
        "--single-shot-dir", type=Path, default=DEFAULT_SINGLE_SHOT_DIR
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    experiment_dir = args.experiment_dir.resolve()
    if args.command in {"prepare", "run"}:
        prepare_comparison(
            experiment_dir,
            baseline_dir=args.baseline_dir.resolve(),
            single_shot_dir=args.single_shot_dir.resolve(),
        )
        print(f"prepared={experiment_dir} judge_concurrency={JUDGE_CONCURRENCY}")
    if args.command in {"score", "run"}:
        score_comparison(experiment_dir)
        print(f"scored={experiment_dir / 'results'}")
    if args.command in {"repeat", "run"}:
        repeat_comparison(experiment_dir)
        print(f"repeated={experiment_dir / 'results'}")
    if args.command in {"report", "run"}:
        summary = export_comparison_report(experiment_dir)
        print(
            f"reported={experiment_dir / 'report/comparison.md'} "
            f"luna_records={summary['expected_luna_record_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
