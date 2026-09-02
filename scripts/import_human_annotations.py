"""Import two blinded human-review CSVs and optional disagreement arbitration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hyscript.config import PROJECT_ROOT
from hyscript.evaluation.human import import_human_annotations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=PROJECT_ROOT / "eval/experiments/formal-100-v1",
    )
    parser.add_argument("--reviewer", type=Path, action="append", required=True)
    parser.add_argument("--arbitration", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = import_human_annotations(
        args.experiment_dir.resolve(),
        reviewer_files=args.reviewer,
        arbitration_file=args.arbitration,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not result["pending_arbitration"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
