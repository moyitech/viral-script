"""Serializable contracts for trace-linked offline evaluation records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Literal
from uuid import uuid4

EvaluationKind = Literal["rules", "judge", "human", "aggregate"]
EvaluationStatus = Literal["completed", "failed"]
FindingSeverity = Literal["info", "warning", "gate"]


def _canonical_json_sha256(payload: dict[str, Any]) -> str:
    """Hash one JSON object using a stable, locale-independent encoding."""

    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("fingerprint config must be JSON serializable.") from exc
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class EvaluatorFingerprint:
    """Stable identity and effective parameters for one evaluator."""

    kind: EvaluationKind
    name: str
    version: str
    config: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    prompt_version: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"rules", "judge", "human", "aggregate"}:
            raise ValueError("fingerprint evaluator kind is unsupported.")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("fingerprint evaluator name must not be empty.")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("fingerprint evaluator version must not be empty.")
        if self.model is not None and (
            not isinstance(self.model, str) or not self.model.strip()
        ):
            raise ValueError("fingerprint model must be null or non-empty.")
        if self.prompt_version is not None and (
            not isinstance(self.prompt_version, str) or not self.prompt_version.strip()
        ):
            raise ValueError("fingerprint prompt_version must be null or non-empty.")
        if not isinstance(self.config, dict):
            raise ValueError("fingerprint config must be an object.")
        _canonical_json_sha256(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "version": self.version,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "config": self.config,
        }

    @property
    def sha256(self) -> str:
        return _canonical_json_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class EvaluationFingerprint:
    """Complete scoring configuration used to decide whether resume is safe."""

    rubric_sha256: str
    evaluators: tuple[EvaluatorFingerprint, ...]
    aggregator: EvaluatorFingerprint
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("fingerprint schema_version must be '1.0'.")
        if (
            not isinstance(self.rubric_sha256, str)
            or len(self.rubric_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.rubric_sha256
            )
        ):
            raise ValueError(
                "fingerprint rubric_sha256 must be a lowercase hexadecimal digest."
            )
        if not self.evaluators:
            raise ValueError("fingerprint must contain at least one evaluator.")
        kinds = [fingerprint.kind for fingerprint in self.evaluators]
        if len(set(kinds)) != len(kinds):
            raise ValueError("fingerprint evaluator kinds must be unique.")
        if self.aggregator.kind != "aggregate":
            raise ValueError("fingerprint aggregator must use kind 'aggregate'.")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rubric_sha256": self.rubric_sha256,
            "evaluators": [
                fingerprint.to_dict()
                for fingerprint in sorted(self.evaluators, key=lambda item: item.kind)
            ],
            "aggregator": self.aggregator.to_dict(),
        }

    @property
    def sha256(self) -> str:
        return _canonical_json_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "sha256": self.sha256}


def new_evaluation_id(prefix: str) -> str:
    """Create a readable, collision-resistant evaluator invocation id."""

    return f"{prefix}-{uuid4().hex}"


def utc_now_iso() -> str:
    """Return an RFC 3339 UTC timestamp without platform-dependent formatting."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def evaluation_record_from_dict(payload: Any) -> "EvaluationRecord":
    """Parse a stored record for safe resume and aggregation."""

    if not isinstance(payload, dict):
        raise ValueError("evaluation record must be an object.")
    try:
        allowed_root = {
            "evaluation_id",
            "run_id",
            "trace_sha256",
            "created_at",
            "evaluator",
            "rubric",
            "status",
            "summary",
            "dimension_scores",
            "metrics",
            "findings",
            "metadata",
            "errors",
            "schema_version",
            "gate_failed",
        }
        if set(payload) != allowed_root:
            raise TypeError
        evaluator_payload = payload["evaluator"]
        rubric_payload = payload["rubric"]
        scores_payload = payload["dimension_scores"]
        findings_payload = payload["findings"]
        if not isinstance(evaluator_payload, dict) or not isinstance(
            rubric_payload, dict
        ):
            raise TypeError
        if set(evaluator_payload) != {"kind", "name", "version", "model"}:
            raise TypeError
        if set(rubric_payload) != {"rubric_id", "version", "sha256"}:
            raise TypeError
        if not isinstance(scores_payload, (list, tuple)) or not isinstance(
            findings_payload, (list, tuple)
        ):
            raise TypeError
        evaluator = EvaluatorInfo(**evaluator_payload)
        rubric = RubricRef(**rubric_payload)
        scores = tuple(
            DimensionScore(
                dimension_id=item["dimension_id"],
                name=item["name"],
                score=item["score"],
                reason=item["reason"],
                script_spans=tuple(item.get("script_spans", [])),
                evidence_refs=tuple(item.get("evidence_refs", [])),
            )
            for item in scores_payload
            if isinstance(item, dict)
            and set(item)
            == {
                "dimension_id",
                "name",
                "score",
                "reason",
                "script_spans",
                "evidence_refs",
            }
            and isinstance(item["script_spans"], (list, tuple))
            and isinstance(item["evidence_refs"], (list, tuple))
        )
        if len(scores) != len(scores_payload):
            raise TypeError
        findings = tuple(
            Finding(
                code=item["code"],
                severity=item["severity"],
                message=item["message"],
                details=dict(item.get("details", {})),
            )
            for item in findings_payload
            if isinstance(item, dict)
            and set(item) == {"code", "severity", "message", "details"}
            and isinstance(item["details"], dict)
        )
        if len(findings) != len(findings_payload):
            raise TypeError
        metrics = payload["metrics"]
        metadata = payload["metadata"]
        errors = payload["errors"]
        if (
            not isinstance(metrics, dict)
            or not isinstance(metadata, dict)
            or not isinstance(errors, (list, tuple))
            or any(not isinstance(error, str) or not error for error in errors)
            or not isinstance(payload["gate_failed"], bool)
        ):
            raise TypeError
        record = EvaluationRecord(
            evaluation_id=payload["evaluation_id"],
            run_id=payload["run_id"],
            trace_sha256=payload["trace_sha256"],
            created_at=payload["created_at"],
            evaluator=evaluator,
            rubric=rubric,
            status=payload["status"],
            summary=payload["summary"],
            dimension_scores=scores,
            metrics=dict(metrics),
            findings=findings,
            metadata=dict(metadata),
            errors=tuple(errors),
            schema_version=payload["schema_version"],
        )
        if payload["gate_failed"] != record.gate_failed:
            raise TypeError
        return record
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ValueError("stored evaluation record is invalid.") from None


@dataclass(frozen=True, slots=True)
class EvaluatorInfo:
    """Identity and version of one evaluator invocation."""

    kind: EvaluationKind
    name: str
    version: str
    model: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"rules", "judge", "human", "aggregate"}:
            raise ValueError("evaluator kind is unsupported.")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("evaluator name must not be empty.")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("evaluator version must not be empty.")
        if self.model is not None and not isinstance(self.model, str):
            raise ValueError("evaluator model must be a string or null.")


@dataclass(frozen=True, slots=True)
class RubricRef:
    """Version and content hash of the rubric used for a score."""

    rubric_id: str
    version: str
    sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.rubric_id, str)
            or not self.rubric_id.strip()
            or not isinstance(self.version, str)
            or not self.version.strip()
        ):
            raise ValueError("rubric id and version must not be empty.")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("rubric sha256 must be a lowercase hexadecimal digest.")


@dataclass(frozen=True, slots=True)
class DimensionScore:
    """One validated 0-4 score with an auditable explanation."""

    dimension_id: str
    name: str
    score: int | None
    reason: str
    script_spans: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.dimension_id, str) or not self.dimension_id.strip():
            raise ValueError("dimension_id must not be empty.")
        if self.score is not None:
            if isinstance(self.score, bool) or not isinstance(self.score, int):
                raise TypeError("dimension score must be an integer or null.")
            if not 0 <= self.score <= 4:
                raise ValueError("dimension score must be between 0 and 4.")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("dimension score reason must not be empty.")
        if any(not isinstance(span, str) or not span for span in self.script_spans):
            raise ValueError("script spans must be non-empty strings.")
        if any(not isinstance(ref, str) or not ref for ref in self.evidence_refs):
            raise ValueError("evidence refs must be non-empty strings.")


@dataclass(frozen=True, slots=True)
class Finding:
    """A deterministic or model-based issue found during evaluation."""

    code: str
    severity: FindingSeverity
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in {"info", "warning", "gate"}:
            raise ValueError("finding severity is unsupported.")
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("finding code must not be empty.")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("finding message must not be empty.")


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """One immutable result linked to a frozen generation trace."""

    evaluation_id: str
    run_id: str
    trace_sha256: str
    created_at: str
    evaluator: EvaluatorInfo
    rubric: RubricRef
    status: EvaluationStatus
    summary: str = ""
    dimension_scores: tuple[DimensionScore, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    findings: tuple[Finding, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation_id, str) or not self.evaluation_id.strip():
            raise ValueError("evaluation_id must not be empty.")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must not be empty.")
        if (
            not isinstance(self.trace_sha256, str)
            or len(self.trace_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.trace_sha256
            )
        ):
            raise ValueError("trace_sha256 must be a lowercase hexadecimal digest.")
        if self.status not in {"completed", "failed"}:
            raise ValueError("evaluation status is unsupported.")
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("created_at must not be empty.")
        if not isinstance(self.summary, str):
            raise ValueError("summary must be a string.")
        if not isinstance(self.metrics, dict) or not isinstance(self.metadata, dict):
            raise ValueError("metrics and metadata must be objects.")
        if any(not isinstance(error, str) or not error for error in self.errors):
            raise ValueError("errors must contain non-empty strings.")
        if self.schema_version != "1.0":
            raise ValueError("evaluation schema_version must be '1.0'.")
        if self.status == "completed" and self.errors:
            raise ValueError("completed evaluation records cannot contain errors.")
        if self.status == "failed" and not self.errors:
            raise ValueError("failed evaluation records must contain an error.")

    @property
    def gate_failed(self) -> bool:
        """Return whether any evaluator raised a non-compensable gate."""

        return any(finding.severity == "gate" for finding in self.findings)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""

        payload = asdict(self)
        payload["gate_failed"] = self.gate_failed
        return payload


__all__ = [
    "DimensionScore",
    "EvaluationFingerprint",
    "EvaluationKind",
    "EvaluationRecord",
    "EvaluationStatus",
    "EvaluatorFingerprint",
    "EvaluatorInfo",
    "Finding",
    "FindingSeverity",
    "RubricRef",
    "evaluation_record_from_dict",
    "new_evaluation_id",
    "utc_now_iso",
]
