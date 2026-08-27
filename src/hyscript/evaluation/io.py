"""Strict trace loading and independent evaluation-result storage."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .models import EvaluationRecord

SUPPORTED_TRACE_SCHEMA_VERSION = "1.0"
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SUPPORT_STATUSES = {"supported", "unsupported", "contradicted", "uncertain"}


class TraceInputError(ValueError):
    """Raised when a frozen trace cannot be evaluated safely."""


class ResultWriteError(RuntimeError):
    """Raised when an evaluation result cannot be stored safely."""


@dataclass(frozen=True, slots=True)
class FrozenTrace:
    """The stable subset of a generation trace consumed by evaluators."""

    run_id: str
    task: dict[str, Any]
    script_text: str
    selected_evidence: tuple[dict[str, Any], ...]
    claims: tuple[dict[str, Any], ...]
    queries: tuple[str, ...]
    search_result_count: int
    trace_sha256: str
    source_path: Path
    schema_version: str = SUPPORTED_TRACE_SCHEMA_VERSION


def _object_list(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TraceInputError(f"trace.{key} must be a list of objects.")
    return value


def _required_text(payload: dict[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TraceInputError(f"{context}.{key} must be a non-empty string.")
    return value.strip()


def _validate_task(payload: dict[str, Any]) -> dict[str, Any]:
    task = dict(payload)
    task["topic"] = _required_text(task, "topic", "trace.task")

    if "target_length" in task:
        target_length = task["target_length"]
        if (
            isinstance(target_length, bool)
            or not isinstance(target_length, int)
            or target_length <= 0
        ):
            raise TraceInputError(
                "trace.task.target_length must be a positive integer."
            )

    if "forbidden_phrases" in task:
        phrases = task["forbidden_phrases"]
        if not isinstance(phrases, list):
            raise TraceInputError(
                "trace.task.forbidden_phrases must be a list of strings."
            )
        normalized_phrases: list[str] = []
        for index, phrase in enumerate(phrases):
            if not isinstance(phrase, str) or not phrase.strip():
                raise TraceInputError(
                    f"trace.task.forbidden_phrases[{index}] must be a non-empty string."
                )
            normalized_phrases.append(phrase.strip())
        if len(set(normalized_phrases)) != len(normalized_phrases):
            raise TraceInputError("trace.task.forbidden_phrases must be unique.")
        task["forbidden_phrases"] = normalized_phrases
    return task


def _validate_evidence(payload: dict[str, Any], index: int) -> dict[str, Any]:
    context = f"trace.selected_evidence[{index}]"
    evidence = dict(payload)
    evidence["evidence_id"] = _required_text(evidence, "evidence_id", context)
    if not _SAFE_ARTIFACT_ID.fullmatch(evidence["evidence_id"]):
        raise TraceInputError(
            f"{context}.evidence_id contains unsupported characters or is too long."
        )
    for key in ("url", "title", "snippet", "raw_content", "content"):
        if key in evidence and (
            not isinstance(evidence[key], str) or not evidence[key].strip()
        ):
            raise TraceInputError(
                f"{context}.{key} must be a non-empty string when present."
            )
    return evidence


def _validate_claim(payload: dict[str, Any], index: int) -> dict[str, Any]:
    context = f"trace.claims[{index}]"
    claim = dict(payload)
    claim["claim_id"] = _required_text(claim, "claim_id", context)
    if not _SAFE_ARTIFACT_ID.fullmatch(claim["claim_id"]):
        raise TraceInputError(
            f"{context}.claim_id contains unsupported characters or is too long."
        )
    if "text" in claim:
        claim["text"] = _required_text(claim, "text", context)

    if not isinstance(claim.get("is_core"), bool):
        raise TraceInputError(f"{context}.is_core must be a boolean.")

    references = claim.get("evidence_ids")
    if not isinstance(references, list):
        raise TraceInputError(f"{context}.evidence_ids must be a list of strings.")
    normalized_references: list[str] = []
    for reference_index, reference in enumerate(references):
        if not isinstance(reference, str) or not reference.strip():
            raise TraceInputError(
                f"{context}.evidence_ids[{reference_index}] must be a non-empty string."
            )
        normalized_reference = reference.strip()
        if not _SAFE_ARTIFACT_ID.fullmatch(normalized_reference):
            raise TraceInputError(
                f"{context}.evidence_ids[{reference_index}] contains unsupported "
                "characters or is too long."
            )
        normalized_references.append(normalized_reference)
    if len(set(normalized_references)) != len(normalized_references):
        raise TraceInputError(f"{context}.evidence_ids must be unique.")
    claim["evidence_ids"] = normalized_references

    if "support_status" in claim:
        support_status = claim["support_status"]
        if (
            not isinstance(support_status, str)
            or support_status not in _SUPPORT_STATUSES
        ):
            raise TraceInputError(
                f"{context}.support_status must be one of: "
                f"{', '.join(sorted(_SUPPORT_STATUSES))}."
            )
    return claim


def _validate_unique_ids(
    items: list[dict[str, Any]],
    *,
    key: str,
    context: str,
) -> None:
    seen: set[str] = set()
    for item in items:
        identifier = item[key]
        if identifier in seen:
            raise TraceInputError(
                f"{context}.{key} values must be unique: {identifier!r}."
            )
        seen.add(identifier)


def frozen_trace_from_payload(
    payload: Any,
    *,
    trace_sha256: str,
    source_path: Path,
) -> FrozenTrace:
    """Validate a decoded generation trace and extract evaluator inputs."""

    if not isinstance(payload, dict):
        raise TraceInputError("trace root must be an object.")
    if not isinstance(trace_sha256, str) or not _HEX_DIGEST.fullmatch(trace_sha256):
        raise TraceInputError("trace_sha256 must be a lowercase hexadecimal digest.")
    if payload.get("schema_version") != SUPPORTED_TRACE_SCHEMA_VERSION:
        raise TraceInputError(
            f"trace.schema_version must be {SUPPORTED_TRACE_SCHEMA_VERSION!r}."
        )

    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not _SAFE_RUN_ID.fullmatch(run_id):
        raise TraceInputError(
            "trace.run_id must use 1-128 letters, digits, dots, underscores, or hyphens."
        )

    task = payload.get("task")
    if not isinstance(task, dict):
        raise TraceInputError("trace.task must be an object.")
    validated_task = _validate_task(task)

    artifact = payload.get("script_artifact")
    if not isinstance(artifact, dict):
        raise TraceInputError("trace.script_artifact must be an object.")
    script_text = artifact.get("script_text")
    if not isinstance(script_text, str):
        raise TraceInputError("trace.script_artifact.script_text must be a string.")

    queries_payload = payload.get("queries", [])
    if not isinstance(queries_payload, list):
        raise TraceInputError("trace.queries must be a list of strings.")
    for index, query in enumerate(queries_payload):
        if not isinstance(query, str) or not query.strip():
            raise TraceInputError(f"trace.queries[{index}] must be a non-empty string.")
    search_results = payload.get("search_results", [])
    if not isinstance(search_results, list) or any(
        not isinstance(result, dict) for result in search_results
    ):
        raise TraceInputError("trace.search_results must be a list of objects.")

    evidence = [
        _validate_evidence(item, index)
        for index, item in enumerate(_object_list(payload, "selected_evidence"))
    ]
    claims = [
        _validate_claim(item, index)
        for index, item in enumerate(_object_list(payload, "claims"))
    ]
    _validate_unique_ids(
        evidence,
        key="evidence_id",
        context="trace.selected_evidence",
    )
    _validate_unique_ids(claims, key="claim_id", context="trace.claims")
    if claims and not any(claim["is_core"] for claim in claims):
        raise TraceInputError(
            "trace.claims must mark at least one claim as core when claims are present."
        )

    return FrozenTrace(
        run_id=run_id,
        task=validated_task,
        script_text=script_text,
        selected_evidence=tuple(evidence),
        claims=tuple(claims),
        queries=tuple(queries_payload),
        search_result_count=len(search_results),
        trace_sha256=trace_sha256,
        source_path=source_path,
    )


def load_frozen_trace(path: Path) -> FrozenTrace:
    """Load one immutable trace and hash the exact bytes that were scored."""

    try:
        content = path.read_bytes()
    except OSError as exc:
        raise TraceInputError(f"Could not read trace: {path}") from exc
    digest = hashlib.sha256(content).hexdigest()
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TraceInputError(f"Trace is not valid UTF-8 JSON: {path}") from exc
    return frozen_trace_from_payload(
        payload,
        trace_sha256=digest,
        source_path=path.resolve(),
    )


def write_json_object(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    """Atomically write UTF-8 JSON without silently replacing a result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise ResultWriteError("Result is not valid finite JSON data.") from exc
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.chmod(temporary_path, 0o600)
        if overwrite:
            os.replace(temporary_path, path)
        else:
            # A hard link provides atomic create-if-absent semantics without
            # exposing a partially-written result at the final path.
            os.link(temporary_path, path)
    except FileExistsError:
        raise ResultWriteError(f"Result already exists: {path}") from None
    except OSError as exc:
        raise ResultWriteError(f"Could not write result: {path}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_evaluation_record(
    path: Path,
    record: EvaluationRecord,
    *,
    overwrite: bool = False,
) -> None:
    """Store an evaluation record separately from its frozen trace."""

    write_json_object(path, record.to_dict(), overwrite=overwrite)


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a stored JSON object for resume checks or aggregation."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResultWriteError(f"Could not read stored result: {path}") from exc
    if not isinstance(payload, dict):
        raise ResultWriteError(f"Stored result must be a JSON object: {path}")
    return payload


__all__ = [
    "FrozenTrace",
    "ResultWriteError",
    "SUPPORTED_TRACE_SCHEMA_VERSION",
    "TraceInputError",
    "frozen_trace_from_payload",
    "load_frozen_trace",
    "load_json_object",
    "write_evaluation_record",
    "write_json_object",
]
