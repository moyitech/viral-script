"""Export internal-consistency metrics for two immutable Hy3 Judge passes."""

from __future__ import annotations

import argparse
from pathlib import Path

from hyscript.config import PROJECT_ROOT
from hyscript.evaluation.stability import export_judge_stability


def build_parser() -> argparse.ArgumentParser:
    experiment = PROJECT_ROOT / "eval/experiments/formal-100-v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=experiment / "results")
    parser.add_argument(
        "--repeat-dir",
        type=Path,
        default=experiment / "validation/stability/repeat-001/results",
    )
    parser.add_argument(
        "--trace-manifest",
        type=Path,
        default=experiment / "generation/trace_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=experiment / "validation/stability/repeat-001",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = export_judge_stability(
        args.baseline_dir.resolve(),
        args.repeat_dir.resolve(),
        args.output_dir.resolve(),
        trace_manifest=args.trace_manifest.resolve(),
    )
    print(
        f"compared={summary['record_count']} "
        f"dimension_agreement={summary['overall']['dimension_exact_agreement_rate']:.4f} "
        f"output={args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
