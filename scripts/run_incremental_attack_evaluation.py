"""Incrementally evaluate attack detectors on frozen formal outputs only."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import hashlib
from pathlib import Path

from hyscript.config import PROJECT_ROOT, SettingsError, get_settings
from hyscript.evaluation.citation_verification import (
    MAX_CITATION_SEARCH_CONCURRENCY,
    CitationVerificationConfig,
    CitationVerifier,
    run_citation_verification_formal_validation,
)
from hyscript.evaluation.reward_hacking import (
    MAX_REWARD_HACKING_CONCURRENCY,
    RewardHackingConfig,
    RewardHackingDetector,
    run_reward_hacking_formal_validation,
)
from hyscript.llm import AsyncHy3Client
from hyscript.search import AsyncTavilySearchProvider


DEFAULT_EXPERIMENT_DIRS = (
    PROJECT_ROOT / "eval/experiments/formal-100-v1",
    PROJECT_ROOT / "eval/experiments/formal-100-e2e-single-shot-v1",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run only reward-hacking and Tavily-backed citation checks on frozen "
            "formal outputs. Existing eight-dimension scores are never rerun."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        action="append",
        dest="experiment_dirs",
        help="Formal experiment directory; repeat to evaluate multiple experiments.",
    )
    parser.add_argument(
        "--hy3-concurrency",
        type=int,
        choices=range(1, MAX_REWARD_HACKING_CONCURRENCY + 1),
        default=MAX_REWARD_HACKING_CONCURRENCY,
        metavar=f"1-{MAX_REWARD_HACKING_CONCURRENCY}",
    )
    parser.add_argument(
        "--tavily-concurrency",
        type=int,
        choices=range(1, MAX_CITATION_SEARCH_CONCURRENCY + 1),
        default=8,
        metavar=f"1-{MAX_CITATION_SEARCH_CONCURRENCY}",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("no_think", "low", "high", "xhigh"),
        default="no_think",
    )
    parser.add_argument(
        "--checks",
        choices=("all", "reward-hacking", "citation"),
        default="all",
        help="Select one incremental check or run both. Default: all.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    hy3 = replace(settings.hy3, temperature=0.0, top_p=1.0)
    tavily = settings.tavily
    experiment_dirs = tuple(args.experiment_dirs or DEFAULT_EXPERIMENT_DIRS)
    search_parameters = {
        "provider": "tavily",
        "search_depth": tavily.search_depth,
        "topic": tavily.topic,
        "max_results": tavily.max_results,
        "base_url_sha256": hashlib.sha256(
            tavily.sdk_base_url.encode("utf-8")
        ).hexdigest(),
    }
    failed = False
    async with (
        AsyncHy3Client(hy3) as client,
        AsyncTavilySearchProvider(tavily) as search_provider,
    ):
        reward_detector = RewardHackingDetector(
            client,
            model_name=hy3.model,
            config=RewardHackingConfig(reasoning_effort=args.reasoning_effort),
            sampling_parameters={
                "temperature": hy3.temperature,
                "top_p": hy3.top_p,
            },
        )
        citation_verifier = CitationVerifier(
            client,
            search_provider,
            model_name=hy3.model,
            config=CitationVerificationConfig(
                reasoning_effort=args.reasoning_effort
            ),
            sampling_parameters={
                "temperature": hy3.temperature,
                "top_p": hy3.top_p,
            },
            search_parameters=search_parameters,
            search_concurrency=args.tavily_concurrency,
        )
        for experiment_dir in experiment_dirs:
            report = [f"experiment={experiment_dir.resolve()}"]
            experiment_failures = 0
            if args.checks in {"all", "reward-hacking"}:
                reward = await run_reward_hacking_formal_validation(
                    experiment_dir,
                    reward_detector,
                    concurrency=args.hy3_concurrency,
                    overwrite=args.overwrite,
                )
                experiment_failures += int(reward["failed_count"])
                report.append(
                    f"reward_hacking={reward['flagged_count']}/"
                    f"{reward['completed_count']}"
                )
            if args.checks in {"all", "citation"}:
                citation = await run_citation_verification_formal_validation(
                    experiment_dir,
                    citation_verifier,
                    concurrency=args.hy3_concurrency,
                    overwrite=args.overwrite,
                )
                experiment_failures += int(citation["failed_count"])
                report.extend(
                    (
                        f"citation={citation['flagged_count']}/"
                        f"{citation['completed_count']}",
                        f"citation_searches={citation['search_request_count']}",
                    )
                )
            failed = failed or bool(experiment_failures)
            report.append(f"failures={experiment_failures}")
            print(" ".join(report))
    return 1 if failed else 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(_run(args))
    except (OSError, RuntimeError, ValueError, SettingsError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    raise SystemExit(main())
