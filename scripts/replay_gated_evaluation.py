"""Apply both gates to frozen formal results without live API calls."""

import argparse
import asyncio
import json
from pathlib import Path

from hyscript.config import PROJECT_ROOT
from hyscript.evaluation.gates import GATED_RESULTS_DIRECTORY
from hyscript.evaluation.gated_replay import replay_gated_evaluation, export_gated_comparison


async def run(root: Path):
    rubric = PROJECT_ROOT / "eval/rubrics/script_quality_v1.json"
    names = ("formal-100-v1", "formal-100-e2e-single-shot-v1")
    for name in names:
        directory = root / name
        summary = await replay_gated_evaluation(
            trace_manifest=directory / "generation/trace_manifest.json",
            source_results=directory / "results",
            reward_cache=directory / "validation/incremental_attack/reward_hacking",
            citation_cache=directory / "validation/incremental_attack/citation",
            output=directory / GATED_RESULTS_DIRECTORY, rubric_path=rubric,
        )
        print(name, json.dumps(summary, ensure_ascii=False))
    baseline = root / names[0]
    attack_ids = {
        item["run_id"] for item in json.loads(
            (baseline / "validation/reward_hacking/manifest.json").read_text()
        )["items"]
    }
    attacks = await replay_gated_evaluation(
        trace_manifest=baseline / "validation/discrimination/trace_manifest.json",
        source_results=baseline / "validation/discrimination/results",
        reward_cache=baseline / "validation/reward_hacking",
        citation_cache=baseline / "validation/citation_verification",
        output=baseline / "validation/attacks-gated-v1", rubric_path=rubric,
        selected_run_ids=attack_ids,
    )
    print("attacks", json.dumps(attacks, ensure_ascii=False))
    print("paired", export_gated_comparison(
        baseline / GATED_RESULTS_DIRECTORY,
        root / names[1] / GATED_RESULTS_DIRECTORY,
        root / names[1] / GATED_RESULTS_DIRECTORY,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, default=PROJECT_ROOT / "eval/experiments")
    args = parser.parse_args()
    asyncio.run(run(args.experiments_root))


if __name__ == "__main__":
    main()
