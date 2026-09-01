"""Batch scoring of frozen traces with failure isolation and resume support."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, Sequence
from uuid import uuid4

from .aggregate import (
    AGGREGATOR_NAME,
    AGGREGATOR_VERSION,
    combine_evaluations,
    summarize_batch,
)
from .io import (
    FrozenTrace,
    load_frozen_trace,
    load_json_object,
    write_evaluation_record,
    write_json_object,
)
from .judge import (
    JUDGE_EVALUATOR_NAME,
    JUDGE_EVALUATOR_VERSION,
    JUDGE_PROMPT_VERSION,
    Hy3JudgeEvaluator,
    JudgeEvaluationError,
)
from .models import (
    EvaluationFingerprint,
    EvaluationRecord,
    EvaluatorFingerprint,
    evaluation_record_from_dict,
    utc_now_iso,
)
from .rubric import Rubric
from .rules import RULE_EVALUATOR_NAME, RULE_EVALUATOR_VERSION, RuleEvaluator

EvaluatorName = Literal["rules", "judge"]


class EvaluationConflictError(RuntimeError):
    """Raised when resume would mix incompatible traces or evaluator versions."""


@dataclass(frozen=True, slots=True)
class BatchEvaluationConfig:
    """Explicit behavior for one scoring invocation."""

    output_dir: Path
    evaluators: tuple[EvaluatorName, ...] = ("rules",)
    concurrency: int = 2
    overwrite: bool = False

    def __post_init__(self) -> None:
        if not self.evaluators:
            raise ValueError("At least one evaluator is required.")
        if len(set(self.evaluators)) != len(self.evaluators):
            raise ValueError("Evaluator names must be unique.")
        if any(name not in {"rules", "judge"} for name in self.evaluators):
            raise ValueError("Evaluators must be rules and/or judge.")
        if self.concurrency < 1:
            raise ValueError("concurrency must be greater than zero.")


@dataclass(frozen=True, slots=True)
class TraceOutcome:
    trace_path: str
    run_id: str | None
    status: Literal["completed", "skipped", "failed"]
    error_code: str | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class BatchEvaluationResult:
    evaluation_id: str
    output_dir: Path
    outcomes: tuple[TraceOutcome, ...]

    @property
    def failed_count(self) -> int:
        return sum(outcome.status == "failed" for outcome in self.outcomes)


def _resume_record(
    path: Path,
    *,
    trace: FrozenTrace,
    rubric: Rubric,
    expected: EvaluatorFingerprint,
) -> EvaluationRecord:
    try:
        record = evaluation_record_from_dict(load_json_object(path))
    except (ValueError, RuntimeError):
        raise EvaluationConflictError(
            f"Stored {expected.kind} result is invalid; use --overwrite."
        ) from None
    if record.run_id != trace.run_id or record.trace_sha256 != trace.trace_sha256:
        raise EvaluationConflictError(
            f"Stored {expected.kind} result does not match the frozen trace."
        )
    if record.rubric.sha256 != rubric.sha256:
        raise EvaluationConflictError(
            f"Stored {expected.kind} result uses a different rubric."
        )
    if record.evaluator.kind != expected.kind:
        raise EvaluationConflictError(
            f"Stored result is not a {expected.kind} evaluation."
        )
    if (
        record.evaluator.name != expected.name
        or record.evaluator.version != expected.version
    ):
        raise EvaluationConflictError(
            f"Stored {expected.kind} result uses a different evaluator implementation."
        )
    if record.metadata.get("evaluator_fingerprint") != expected.to_dict():
        raise EvaluationConflictError(
            f"Stored {expected.kind} result uses different evaluator settings."
        )
    if record.status != "completed":
        raise EvaluationConflictError(
            f"Stored {expected.kind} result is incomplete; use --overwrite."
        )
    return record


def _with_fingerprint(
    record: EvaluationRecord,
    fingerprint: EvaluatorFingerprint,
) -> EvaluationRecord:
    """Attach the effective evaluator config to the immutable stored record."""

    return replace(
        record,
        metadata={
            **record.metadata,
            "evaluator_fingerprint": fingerprint.to_dict(),
        },
    )


def _source_ids(record: EvaluationRecord) -> list[tuple[str, str]]:
    sources = record.metadata.get("source_evaluations")
    if not isinstance(sources, list):
        return []
    result: list[tuple[str, str]] = []
    for source in sources:
        if not isinstance(source, dict):
            return []
        evaluation_id = source.get("evaluation_id")
        kind = source.get("kind")
        if not isinstance(evaluation_id, str) or not isinstance(kind, str):
            return []
        result.append((kind, evaluation_id))
    return result


class BatchEvaluationRunner:
    """Score traces while keeping generation inputs byte-for-byte unchanged."""

    def __init__(
        self,
        rubric: Rubric,
        config: BatchEvaluationConfig,
        *,
        rule_evaluator: RuleEvaluator | None = None,
        judge_evaluator: Hy3JudgeEvaluator | None = None,
    ) -> None:
        if "judge" in config.evaluators and judge_evaluator is None:
            raise ValueError("judge_evaluator is required when judge is selected.")
        self.rubric = rubric
        self.config = config
        self.rule_evaluator = rule_evaluator or RuleEvaluator()
        self.judge_evaluator = judge_evaluator
        self._judge_semaphore = asyncio.Semaphore(config.concurrency)
        evaluator_fingerprints: list[EvaluatorFingerprint] = []
        if "rules" in config.evaluators:
            evaluator_fingerprints.append(
                EvaluatorFingerprint(
                    kind="rules",
                    name=RULE_EVALUATOR_NAME,
                    version=RULE_EVALUATOR_VERSION,
                    config=asdict(self.rule_evaluator.config),
                )
            )
        if "judge" in config.evaluators:
            if self.judge_evaluator is None:  # pragma: no cover - guarded above
                raise RuntimeError("Judge evaluator is unavailable.")
            evaluator_fingerprints.append(
                EvaluatorFingerprint(
                    kind="judge",
                    name=JUDGE_EVALUATOR_NAME,
                    version=JUDGE_EVALUATOR_VERSION,
                    model=self.judge_evaluator.model_name,
                    prompt_version=JUDGE_PROMPT_VERSION,
                    config={
                        "request": asdict(self.judge_evaluator.config),
                        "sampling_parameters": self.judge_evaluator.sampling_parameters,
                    },
                )
            )
        self._fingerprints = {
            fingerprint.kind: fingerprint for fingerprint in evaluator_fingerprints
        }
        self._aggregate_fingerprint = EvaluatorFingerprint(
            kind="aggregate",
            name=AGGREGATOR_NAME,
            version=AGGREGATOR_VERSION,
        )
        self.fingerprint = EvaluationFingerprint(
            rubric_sha256=rubric.sha256,
            evaluators=tuple(evaluator_fingerprints),
            aggregator=self._aggregate_fingerprint,
        )

    async def _judge(self, trace: FrozenTrace) -> EvaluationRecord:
        if self.judge_evaluator is None:  # pragma: no cover - constructor guards it
            raise RuntimeError("Judge evaluator is unavailable.")
        return await self.judge_evaluator.evaluate(
            trace,
            self.rubric,
            request_semaphore=self._judge_semaphore,
        )

    async def _score_trace(
        self,
        trace: FrozenTrace,
    ) -> tuple[EvaluationRecord, bool]:
        item_dir = self.config.output_dir / "items" / trace.run_id
        records: list[EvaluationRecord] = []
        all_resumed = True

        if "rules" in self.config.evaluators:
            rules_fingerprint = self._fingerprints["rules"]
            rules_path = item_dir / "rules.json"
            if rules_path.exists() and not self.config.overwrite:
                rule_record = _resume_record(
                    rules_path,
                    trace=trace,
                    rubric=self.rubric,
                    expected=rules_fingerprint,
                )
            else:
                rule_record = _with_fingerprint(
                    self.rule_evaluator.evaluate(trace, self.rubric),
                    rules_fingerprint,
                )
                write_evaluation_record(
                    rules_path,
                    rule_record,
                    overwrite=self.config.overwrite,
                )
                all_resumed = False
            records.append(rule_record)

        if "judge" in self.config.evaluators:
            judge_fingerprint = self._fingerprints["judge"]
            judge_path = item_dir / "hy3_judge.json"
            if judge_path.exists() and not self.config.overwrite:
                judge_record = _resume_record(
                    judge_path,
                    trace=trace,
                    rubric=self.rubric,
                    expected=judge_fingerprint,
                )
            else:
                judge_record = _with_fingerprint(
                    await self._judge(trace),
                    judge_fingerprint,
                )
                write_evaluation_record(
                    judge_path,
                    judge_record,
                    overwrite=self.config.overwrite,
                )
                all_resumed = False
            records.append(judge_record)

        combined_path = item_dir / "combined.json"
        expected_sources = [
            (record.evaluator.kind, record.evaluation_id) for record in records
        ]
        if all_resumed and combined_path.exists() and not self.config.overwrite:
            try:
                combined = _resume_record(
                    combined_path,
                    trace=trace,
                    rubric=self.rubric,
                    expected=self._aggregate_fingerprint,
                )
                if _source_ids(combined) != expected_sources:
                    raise EvaluationConflictError(
                        "Stored aggregate has different source evaluations."
                    )
            except EvaluationConflictError:
                combined = _with_fingerprint(
                    combine_evaluations(trace, self.rubric, records),
                    self._aggregate_fingerprint,
                )
                write_evaluation_record(combined_path, combined, overwrite=True)
        else:
            combined = _with_fingerprint(
                combine_evaluations(trace, self.rubric, records),
                self._aggregate_fingerprint,
            )
            # Combined output is derived from independently persisted sources.
            # Replacing a stale derived record is safe during partial resume.
            write_evaluation_record(combined_path, combined, overwrite=True)
        return combined, all_resumed

    def _prepare_manifest(
        self,
        trace_paths: Sequence[Path],
        manifest_inputs: list[dict[str, object]],
    ) -> tuple[str, dict[str, object]]:
        """Validate invocation compatibility before touching stored results."""

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.config.output_dir / "manifest.json"
        now = utc_now_iso()
        evaluation_id = (
            f"eval-{now.replace(':', '').replace('-', '')}-{uuid4().hex[:8]}"
        )
        started_at = now
        existing_manifest: dict[str, object] | None = None
        if manifest_path.exists():
            try:
                existing_manifest = load_json_object(manifest_path)
            except RuntimeError as exc:
                if not self.config.overwrite:
                    raise EvaluationConflictError(
                        "Existing manifest is invalid; use --overwrite or a new output directory."
                    ) from exc
            if existing_manifest is not None:
                stored_id = existing_manifest.get("evaluation_id")
                stored_started_at = existing_manifest.get("started_at")
                if isinstance(stored_id, str) and stored_id:
                    evaluation_id = stored_id
                if isinstance(stored_started_at, str) and stored_started_at:
                    started_at = stored_started_at
                if (
                    not self.config.overwrite
                    and existing_manifest.get("fingerprint")
                    != self.fingerprint.to_dict()
                ):
                    raise EvaluationConflictError(
                        "Evaluation settings differ from the existing output; "
                        "use --overwrite or a new output directory."
                    )
                if not self.config.overwrite:
                    stored_trace_files = existing_manifest.get("trace_files")
                    if stored_trace_files != [path.name for path in trace_paths]:
                        raise EvaluationConflictError(
                            "Input trace set differs from the existing output; "
                            "use --overwrite or a new output directory."
                        )
                    stored_inputs = existing_manifest.get("inputs")
                    if stored_inputs is not None and stored_inputs != manifest_inputs:
                        raise EvaluationConflictError(
                            "Input trace hashes differ from the existing output; "
                            "use --overwrite or a new output directory."
                        )
        elif any(self.config.output_dir.iterdir()) and not self.config.overwrite:
            raise EvaluationConflictError(
                "Output directory is not empty and has no compatible manifest; "
                "use --overwrite or a new output directory."
            )

        manifest_base: dict[str, object] = {
            "schema_version": "1.0",
            "evaluation_id": evaluation_id,
            "started_at": started_at,
            "last_started_at": now,
            "rubric": {
                "rubric_id": self.rubric.rubric_id,
                "version": self.rubric.version,
                "sha256": self.rubric.sha256,
            },
            "evaluators": list(self.config.evaluators),
            "fingerprint": self.fingerprint.to_dict(),
            "concurrency": self.config.concurrency,
            # Store only display names here; run_id and hashes are added after
            # strict loading. Absolute workstation paths are unnecessary.
            "trace_files": [path.name for path in trace_paths],
        }
        write_json_object(
            manifest_path,
            {**manifest_base, "inputs": manifest_inputs, "status": "running"},
            overwrite=True,
        )
        return evaluation_id, manifest_base

    async def run(self, trace_paths: Sequence[Path]) -> BatchEvaluationResult:
        """Evaluate all paths with per-trace error isolation."""

        if not trace_paths:
            raise ValueError("At least one trace path is required.")
        loaded: list[FrozenTrace] = []
        outcomes: list[TraceOutcome] = []
        seen_run_ids: set[str] = set()
        manifest_inputs: list[dict[str, object]] = []
        for path in trace_paths:
            try:
                trace = load_frozen_trace(path)
                if trace.run_id in seen_run_ids:
                    raise EvaluationConflictError(
                        f"Duplicate run_id in input: {trace.run_id}"
                    )
                seen_run_ids.add(trace.run_id)
                loaded.append(trace)
                manifest_inputs.append(
                    {
                        "source": path.name,
                        "run_id": trace.run_id,
                        "trace_sha256": trace.trace_sha256,
                    }
                )
            except Exception as exc:
                manifest_inputs.append({"source": path.name, "status": "invalid_trace"})
                outcomes.append(
                    TraceOutcome(
                        trace_path=str(path),
                        run_id=None,
                        status="failed",
                        error_code="invalid_trace",
                        message=str(exc),
                    )
                )

        try:
            evaluation_id, manifest_base = self._prepare_manifest(
                trace_paths,
                manifest_inputs,
            )
        except EvaluationConflictError as exc:
            return BatchEvaluationResult(
                evaluation_id="resume-conflict",
                output_dir=self.config.output_dir,
                outcomes=tuple(
                    TraceOutcome(
                        trace_path=str(path),
                        run_id=None,
                        status="failed",
                        error_code="resume_conflict",
                        message=str(exc),
                    )
                    for path in trace_paths
                ),
            )

        async def score_one(
            trace: FrozenTrace,
        ) -> tuple[TraceOutcome, EvaluationRecord | None]:
            try:
                combined, resumed = await self._score_trace(trace)
                return (
                    TraceOutcome(
                        trace_path=str(trace.source_path),
                        run_id=trace.run_id,
                        status="skipped" if resumed else "completed",
                    ),
                    combined,
                )
            except EvaluationConflictError as exc:
                return (
                    TraceOutcome(
                        trace_path=str(trace.source_path),
                        run_id=trace.run_id,
                        status="failed",
                        error_code="resume_conflict",
                        message=str(exc),
                    ),
                    None,
                )
            except JudgeEvaluationError as exc:
                return (
                    TraceOutcome(
                        trace_path=str(trace.source_path),
                        run_id=trace.run_id,
                        status="failed",
                        error_code="judge_failed",
                        message=str(exc),
                    ),
                    None,
                )
            except Exception:
                return (
                    TraceOutcome(
                        trace_path=str(trace.source_path),
                        run_id=trace.run_id,
                        status="failed",
                        error_code="evaluation_failed",
                        message="Evaluation failed unexpectedly.",
                    ),
                    None,
                )

        scored = await asyncio.gather(*(score_one(trace) for trace in loaded))
        combined_records: list[EvaluationRecord] = []
        for outcome, combined in scored:
            outcomes.append(outcome)
            if combined is not None:
                combined_records.append(combined)

        summary = {
            "schema_version": "1.0",
            "evaluation_id": evaluation_id,
            "completed_at": utc_now_iso(),
            "counts_scope": "current_invocation",
            "counts": {
                "input": len(trace_paths),
                "completed": sum(outcome.status == "completed" for outcome in outcomes),
                "skipped": sum(outcome.status == "skipped" for outcome in outcomes),
                "failed": sum(outcome.status == "failed" for outcome in outcomes),
            },
            "record_coverage": {
                "input_trace_count": len(trace_paths),
                "validated_trace_count": len(loaded),
                "combined_record_count": len(combined_records),
                "unavailable_record_count": len(trace_paths)
                - len(combined_records),
                "complete": len(combined_records) == len(trace_paths),
            },
            "aggregate": summarize_batch(combined_records),
        }
        write_json_object(
            self.config.output_dir / "summary.json",
            summary,
            overwrite=True,
        )
        failures = [
            {
                "trace_path": outcome.trace_path,
                "run_id": outcome.run_id,
                "error_code": outcome.error_code,
                "message": outcome.message,
            }
            for outcome in outcomes
            if outcome.status == "failed"
        ]
        write_json_object(
            self.config.output_dir / "failures.json",
            {"schema_version": "1.0", "items": failures},
            overwrite=True,
        )
        write_json_object(
            self.config.output_dir / "manifest.json",
            {
                **manifest_base,
                "completed_at": summary["completed_at"],
                "inputs": manifest_inputs,
                "status": "completed_with_failures" if failures else "completed",
            },
            overwrite=True,
        )
        return BatchEvaluationResult(
            evaluation_id=evaluation_id,
            output_dir=self.config.output_dir,
            outcomes=tuple(outcomes),
        )


__all__ = [
    "BatchEvaluationConfig",
    "BatchEvaluationResult",
    "BatchEvaluationRunner",
    "EvaluationConflictError",
    "EvaluatorName",
    "TraceOutcome",
]
