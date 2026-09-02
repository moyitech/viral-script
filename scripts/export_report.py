"""Export the formal experiment's complete CSV, summary, and report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hyscript.config import PROJECT_ROOT
from hyscript.evaluation.formal import export_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=PROJECT_ROOT / "eval/experiments/formal-100-v1",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = export_report(args.experiment_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
