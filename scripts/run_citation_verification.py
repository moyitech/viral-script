"""Verify explicit citations in the frozen adversarial attack traces."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import hashlib
from pathlib import Path

from hyscript.config import PROJECT_ROOT, SettingsError, get_settings
from hyscript.evaluation.citation_verification import (
    MAX_CITATION_CONCURRENCY,
    MAX_CITATION_SEARCH_CONCURRENCY,
    CitationVerificationConfig,
    CitationVerifier,
    run_citation_verification_validation,
)
from hyscript.llm import AsyncHy3Client
from hyscript.search import AsyncTavilySearchProvider


DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "eval/experiments/formal-100-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Tavily-backed explicit-citation verification only on frozen "
            "adversarial attack traces. Formal 100-topic scores are not rerun."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument(
        "--concurrency",
        type=int,
        choices=range(1, MAX_CITATION_CONCURRENCY + 1),
        default=20,
        metavar=f"1-{MAX_CITATION_CONCURRENCY}",
    )
    parser.add_argument(
        "--search-concurrency",
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
    parser.add_argument("--overwrite", action="store_true")
    return parser


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    hy3 = replace(settings.hy3, temperature=0.0, top_p=1.0)
    tavily = settings.tavily
    search_parameters = {
        "provider": "tavily",
        "search_depth": tavily.search_depth,
        "topic": tavily.topic,
        "max_results": tavily.max_results,
        "base_url_sha256": hashlib.sha256(
            tavily.sdk_base_url.encode("utf-8")
        ).hexdigest(),
    }
    async with (
        AsyncHy3Client(hy3) as client,
        AsyncTavilySearchProvider(tavily) as search_provider,
    ):
        verifier = CitationVerifier(
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
            search_concurrency=args.search_concurrency,
        )
        summary = await run_citation_verification_validation(
            args.experiment_dir,
            verifier,
            concurrency=args.concurrency,
            overwrite=args.overwrite,
        )
    print(
        f"completed={summary['completed_count']}/{summary['expected_case_count']} "
        f"fabricated_recall={summary['attack_recall']} "
        f"control_false_positives={summary['false_positive_count']} "
        f"failed={summary['failed_count']} "
        f"output={args.experiment_dir.resolve() / 'validation/citation_verification'}"
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
