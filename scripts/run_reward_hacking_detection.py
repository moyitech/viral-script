"""Run the reward-hacking detector only on frozen adversarial attack traces."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from pathlib import Path

from hyscript.config import PROJECT_ROOT, SettingsError, get_settings
from hyscript.evaluation.reward_hacking import (
    MAX_REWARD_HACKING_CONCURRENCY,
    RewardHackingConfig,
    RewardHackingDetector,
    run_reward_hacking_validation,
)
from hyscript.llm import AsyncHy3Client


DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "eval/experiments/formal-100-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate only the frozen adversarial attack traces with the "
            "reward-hacking detector. Formal 100-topic scores are not read or rerun."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument(
        "--concurrency",
        type=int,
        choices=range(1, MAX_REWARD_HACKING_CONCURRENCY + 1),
        default=20,
        metavar=f"1-{MAX_REWARD_HACKING_CONCURRENCY}",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("no_think", "low", "high"),
        default="no_think",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


async def _run(args: argparse.Namespace) -> int:
    hy3 = replace(get_settings().hy3, temperature=0.0, top_p=1.0)
    async with AsyncHy3Client(hy3) as client:
        detector = RewardHackingDetector(
            client,
            model_name=hy3.model,
            config=RewardHackingConfig(reasoning_effort=args.reasoning_effort),
            sampling_parameters={
                "temperature": hy3.temperature,
                "top_p": hy3.top_p,
            },
        )
        summary = await run_reward_hacking_validation(
            args.experiment_dir,
            detector,
            concurrency=args.concurrency,
            overwrite=args.overwrite,
        )
    print(
        f"completed={summary['completed_count']}/{summary['expected_attack_count']} "
        f"detected={summary['detected_count']} failed={summary['failed_count']} "
        f"output={args.experiment_dir.resolve() / 'validation/reward_hacking'}"
    )
    return 1 if summary["failed_count"] else 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(_run(args))
    except (OSError, RuntimeError, ValueError, SettingsError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    raise SystemExit(main())
