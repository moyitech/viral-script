"""Research one selected topic, generate its oral script, and freeze a trace."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import logging

from hyscript.agent import (
    ResearchAgent,
    ResearchOutcome,
    ScriptAgent,
    ScriptArtifact,
    ScriptTask,
)
from hyscript.artifacts import build_generation_trace
from hyscript.config import settings
from hyscript.llm import AsyncHy3Client, summarize_token_usage
from hyscript.search import AsyncTavilySearchProvider


logger = logging.getLogger("hyscript.example.end_to_end")


def configure_progress_logging() -> None:
    """Show HyScript progress without enabling noisy third-party SDK logs."""

    package_logger = logging.getLogger("hyscript")
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    package_logger.handlers.clear()
    package_logger.addHandler(handler)
    package_logger.setLevel(settings.runtime.log_level)
    package_logger.propagate = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research one selected topic and generate an evidence-backed oral script.",
    )
    parser.add_argument(
        "topic",
        help="Selected topic to research; a hot-list title is not evidence.",
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=450,
        help="Target non-whitespace character count (default: 450).",
    )
    parser.add_argument("--angle", default="", help="Optional creator-selected angle.")
    return parser.parse_args()


def usage_statistics(
    research: ResearchOutcome,
    script: ScriptArtifact | None = None,
) -> dict[str, dict[str, int]]:
    """Build provider-call and provider-reported token totals for this run."""

    script_attempts = script.generation_attempt_count if script is not None else 0
    llm_usages = (
        (*research.llm_usages, *script.llm_usages)
        if script is not None
        else research.llm_usages
    )
    token_summary = summarize_token_usage(llm_usages)
    tavily_succeeded = len(research.search_responses)
    return {
        "tavily": {
            "attempted_calls": research.search_request_count,
            "succeeded_calls": tavily_succeeded,
            "failed_calls": research.search_request_count - tavily_succeeded,
        },
        "hy3": {
            "attempted_calls": research.llm_request_count + script_attempts,
            "reported_usage_calls": token_summary.reported_call_count,
            "input_tokens": token_summary.input_tokens,
            "output_tokens": token_summary.output_tokens,
            "total_tokens": token_summary.total_tokens,
            "reasoning_tokens": token_summary.reasoning_tokens,
            "cached_input_tokens": token_summary.cached_input_tokens,
        },
    }


def log_usage_statistics(usage: dict[str, dict[str, int]]) -> None:
    tavily = usage["tavily"]
    hy3 = usage["hy3"]
    logger.info(
        "调用统计：Tavily 尝试=%d，成功=%d，失败=%d；"
        "Hy3 尝试=%d，input=%d，output=%d，total=%d，reasoning=%d，cached_input=%d",
        tavily["attempted_calls"],
        tavily["succeeded_calls"],
        tavily["failed_calls"],
        hy3["attempted_calls"],
        hy3["input_tokens"],
        hy3["output_tokens"],
        hy3["total_tokens"],
        hy3["reasoning_tokens"],
        hy3["cached_input_tokens"],
    )
    if hy3["reported_usage_calls"] < hy3["attempted_calls"]:
        logger.warning(
            "有 %d 次 Hy3 请求没有返回 token usage，token 合计只包含服务端已报告的调用",
            hy3["attempted_calls"] - hy3["reported_usage_calls"],
        )


async def run(args: argparse.Namespace) -> None:
    task = ScriptTask(
        topic=args.topic,
        target_length=args.target_length,
        angle=args.angle,
    )
    async with (
        AsyncHy3Client(settings.hy3) as llm,
        AsyncTavilySearchProvider(settings.tavily) as search,
    ):
        research = await ResearchAgent(
            llm,
            search,
            config=settings.research,
        ).research(task)
        if research.status != "ready":
            usage = usage_statistics(research)
            log_usage_statistics(usage)
            print(
                json.dumps(
                    {
                        "status": research.status,
                        "errors": research.errors,
                        "query_plan": asdict(research.query_plan),
                        "search_request_count": research.search_request_count,
                        "usage": usage,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        script = await ScriptAgent(
            llm,
            config=settings.script_generation,
        ).generate(task, research)

    usage = usage_statistics(research, script)
    log_usage_statistics(usage)

    trace = build_generation_trace(
        task,
        research,
        script,
        config={
            "research": asdict(settings.research),
            "script_generation": asdict(settings.script_generation),
        },
    )
    trace_path = settings.runtime.runs_dir / f"{trace.run_id}.json"
    logger.info("[5/5] 正在保存生成 trace")
    trace.write_json(trace_path)
    logger.info("trace 已保存：%s", trace_path)
    print(script.script_text)
    print(f"\ntrace: {trace_path}")
    print("usage:")
    print(json.dumps(usage, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    configure_progress_logging()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
