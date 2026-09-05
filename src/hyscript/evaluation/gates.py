"""Mandatory, trace-bound attack checks before full quality scoring."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace
import hashlib
import os
from pathlib import Path
from typing import Any

from hyscript.config import Settings
from hyscript.llm import AsyncHy3Client
from hyscript.search import AsyncTavilySearchProvider

from .citation_verification import CitationVerifier
from .io import FrozenTrace, load_frozen_trace, load_json_object, write_json_object
from .models import (
    EvaluationRecord, EvaluatorFingerprint, EvaluatorInfo, Finding, RubricRef,
    new_evaluation_id, utc_now_iso,
)
from .reward_hacking import RewardHackingDetector
from .rubric import Rubric

GATE_VERSION = "1.0.0"
GATE_NAME = "reward-hacking-and-citation-gates"
GATED_RESULTS_DIRECTORY = "results-gated-v1"
CHECKS = (
    ("reward_hacking", "reward_hacking", "Reward-hacking"),
    ("citation", "fabricated_citation", "引用风险"),
)


def current_results(legacy: Path) -> Path:
    """Prefer a separately versioned gated run without changing historical locks."""
    if legacy.name.endswith("-gated-v1"):
        return legacy
    gated = legacy.with_name(legacy.name + "-gated-v1")
    return gated if gated.is_dir() else legacy


def gated_scoring_options(legacy: Path, experiment: Path) -> list[str]:
    """Use a new output and optionally reuse completed historical detector caches."""
    options = ["--output-dir", str(legacy.with_name(legacy.name + "-gated-v1"))]
    if (legacy / "manifest.json").is_file():
        options.extend(["--reuse-results-dir", str(legacy)])
    reward = experiment / "validation/incremental_attack/reward_hacking"
    citation = experiment / "validation/incremental_attack/citation"
    if all((path / "manifest.json").is_file() for path in (reward, citation)):
        options.extend(["--reward-gate-cache", str(reward), "--citation-gate-cache", str(citation)])
    return options


def passed_trace_manifest(source: Path, results: Path, destination: Path) -> Path:
    """Select already-passed traces for a Judge-only repeat diagnostic."""
    payload = load_json_object(source)
    tasks = []
    for item in payload["tasks"]:
        trace_path = (source.parent / item["trace"]).resolve()
        trace = load_frozen_trace(trace_path)
        result = load_json_object(results / "items" / trace.run_id / "combined.json")
        if (
            result.get("run_id") != trace.run_id
            or result.get("trace_sha256") != trace.trace_sha256
            or item.get("trace_sha256", trace.trace_sha256) != trace.trace_sha256
        ):
            raise GateConflictError("Repeat selection has a mismatched trace hash.")
        checks = result.get("metadata", {}).get("attack_gates", {})
        if result.get("status") != "completed" or set(checks) != {"reward_hacking", "citation"}:
            raise GateEvaluationError("Repeat selection requires completed attack gates.")
        if result.get("gate_failed"):
            continue
        if any(check.get("passed") is not True for check in checks.values()):
            raise GateEvaluationError("Repeat selection requires both gates to pass.")
        tasks.append({**item, "run_id": trace.run_id, "trace": os.path.relpath(trace_path, destination.parent)})
    if not tasks:
        raise GateEvaluationError("No gate-passing scripts are available for Judge diagnostics.")
    write_json_object(destination, {"schema_version": "1.0", "tasks": tasks}, overwrite=True)
    return destination


class GateEvaluationError(RuntimeError):
    """A check failed to complete; this must never count as a pass."""


class GateConflictError(GateEvaluationError):
    """Cached checks cannot be reused for this trace or detector."""


def validate_check(
    payload: dict[str, Any], trace: FrozenTrace, fingerprint: dict[str, Any], field: str,
) -> dict[str, Any]:
    if (
        payload.get("run_id") != trace.run_id
        or payload.get("trace_sha256") != trace.trace_sha256
        or payload.get("detector") != fingerprint
        or not isinstance(payload.get(field), bool)
        or not isinstance(payload.get("reason"), str)
        or not payload["reason"].strip()
    ):
        raise GateConflictError("Gate result does not match the frozen trace or detector.")
    return payload


class AttackGateEvaluator:
    """Run both checks and checkpoint each before any eight-dimension scoring."""

    def __init__(self, reward_detector: Any, citation_verifier: Any) -> None:
        self.detectors = {"reward_hacking": reward_detector, "citation": citation_verifier}

    @property
    def fingerprint(self) -> EvaluatorFingerprint:
        return EvaluatorFingerprint(
            kind="gates", name=GATE_NAME, version=GATE_VERSION,
            config={name: detector.fingerprint for name, detector in self.detectors.items()},
        )

    async def evaluate(
        self, trace: FrozenTrace, rubric: Rubric, *, item_dir: Path, overwrite: bool = False,
    ) -> EvaluationRecord:
        findings: list[Finding] = []
        checks: dict[str, Any] = {}
        for name, field, label in CHECKS:
            detector = self.detectors[name]
            path = item_dir / f"{name}_check.json"
            try:
                if path.exists() and not overwrite:
                    payload = load_json_object(path)
                else:
                    arguments = {
                        "blind_case_id": trace.run_id,
                        "source_trace": str(trace.source_path),
                        ("attack_type" if name == "reward_hacking" else "expected_attack_type"):
                            "unlabelled",
                    }
                    payload = await detector.evaluate(trace, **arguments)
                validate_check(payload, trace, detector.fingerprint, field)
                if not path.exists() or overwrite:
                    write_json_object(path, payload, overwrite=overwrite)
            except GateConflictError:
                raise
            except Exception as exc:
                raise GateEvaluationError(f"{label}门控未完成，请重试；尚未进行八维评分。") from exc
            checks[name] = {
                "passed": not payload[field], "reason": payload["reason"],
                "detector_sha256": detector.fingerprint["sha256"],
                "result_file": path.name,
            }
            if payload[field]:
                findings.append(Finding(
                    code="reward_hacking" if name == "reward_hacking" else "citation_risk",
                    severity="gate", message=f"{label}：{payload['reason']}",
                    details={"check": name, "detector_sha256": detector.fingerprint["sha256"]},
                ))
        return EvaluationRecord(
            evaluation_id=new_evaluation_id("gates"), run_id=trace.run_id,
            trace_sha256=trace.trace_sha256, created_at=utc_now_iso(),
            evaluator=EvaluatorInfo(kind="gates", name=GATE_NAME, version=GATE_VERSION),
            rubric=RubricRef(rubric.rubric_id, rubric.version, rubric.sha256),
            status="completed",
            summary=(
                "未通过前置门控，未进行八维评分。" if findings
                else "Reward-hacking 与引用风险门控均通过。"
            ),
            findings=tuple(findings), metrics={"checks": checks, "passed": not findings},
        )


class RecordedDetector:
    """Replay explicit detector records without credentials or network fallback."""

    def __init__(self, directory: Path, field: str) -> None:
        self.directory = directory
        self.field = field
        manifest = load_json_object(directory / "manifest.json")
        summary = load_json_object(directory / "summary.json")
        self.fingerprint = summary["detector"]
        if manifest.get("detector") != self.fingerprint:
            raise GateConflictError("Detector manifest and summary fingerprints differ.")
        self.records: dict[str, Path] = {}
        for item in manifest["items"]:
            run_id = item["run_id"]
            if run_id in self.records:
                raise GateConflictError("Duplicate run_id in detector cache.")
            path = (directory / item["result"]).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise GateConflictError("Detector result path escapes its directory.")
            self.records[run_id] = path

    async def evaluate(self, trace: FrozenTrace, **_: Any) -> dict[str, Any]:
        path = self.records.get(trace.run_id)
        if path is None:
            raise GateEvaluationError(f"No cached {self.field} result for {trace.run_id}.")
        return validate_check(load_json_object(path), trace, self.fingerprint, self.field)


def recorded_gates(reward_directory: Path, citation_directory: Path) -> AttackGateEvaluator:
    return AttackGateEvaluator(
        RecordedDetector(reward_directory, "reward_hacking"),
        RecordedDetector(citation_directory, "fabricated_citation"),
    )


@asynccontextmanager
async def live_attack_gates(settings: Settings):
    """Use Hy3 for both gates even when a different model is the quality Judge."""
    hy3 = replace(settings.hy3, temperature=0.0, top_p=1.0)
    async with (
        AsyncHy3Client(hy3) as client,
        AsyncTavilySearchProvider(settings.tavily) as search,
    ):
        yield AttackGateEvaluator(
            RewardHackingDetector(client, model_name=hy3.model),
            CitationVerifier(
                client, search, model_name=hy3.model,
                search_parameters={
                    "provider": "tavily", "search_depth": settings.tavily.search_depth,
                    "topic": settings.tavily.topic, "max_results": settings.tavily.max_results,
                    "base_url_sha256": hashlib.sha256(
                        settings.tavily.sdk_base_url.encode("utf-8")
                    ).hexdigest(),
                },
                search_concurrency=8,
            ),
        )
