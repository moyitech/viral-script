"""Recompute archived human preference results without network access."""

import argparse
import json
from pathlib import Path

from hyscript.preference_evaluation import recompute


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=root / "eval/experiments/formal-100-e2e-single-shot-v1/validation/human_preference")
    parser.add_argument("--paired-results", type=Path, default=root / "eval/experiments/formal-100-e2e-single-shot-v1/report/paired_results.csv")
    parser.add_argument("--check", action="store_true", help="Compare computed results with archived summary")
    args = parser.parse_args()
    result = recompute(args.archive, args.paired_results)
    if args.check:
        expected = json.loads((args.archive / "summary.json").read_text())
        if result != expected:
            raise SystemExit("Computed results differ from summary.json")
        print("Archived preference results reproduced successfully.")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
