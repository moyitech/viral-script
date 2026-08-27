"""Serializable generation trace frozen before offline evaluation starts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any


@dataclass(slots=True)
class TraceSearchResult:
    """Normalized live-search result retained for replay and audit."""

    rank: int
    title: str
    url: str
    snippet: str
    score: float | None = None
    published_at: str | None = None
    content_hash: str | None = None


@dataclass(slots=True)
class RunTrace:
    """Inputs and intermediate artifacts from one generation run.

    Offline scores deliberately do not belong in this schema. An evaluator
    writes a separate result that references ``run_id``. Evaluation-ready
    traces use ``task.topic``, optional ``task.target_length`` and
    ``task.forbidden_phrases``, ``script_artifact.script_text``,
    ``selected_evidence[*].evidence_id``, and
    ``claims[*].evidence_ids`` as their stable field contract.
    """

    run_id: str
    created_at: str
    task: dict[str, Any]
    schema_version: str = "1.0"
    config: dict[str, Any] = field(default_factory=dict)
    query_plan: dict[str, Any] = field(default_factory=dict)
    queries: list[str] = field(default_factory=list)
    search_results: list[TraceSearchResult] = field(default_factory=list)
    selected_evidence: list[dict[str, Any]] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    script_artifact: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    latency: dict[str, float] = field(default_factory=dict)
    token_usage: dict[str, int] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)

    def write_json(self, path: Path, *, overwrite: bool = False) -> None:
        """Atomically freeze the trace; replacement must be explicit."""

        path.parent.mkdir(parents=True, exist_ok=True)
        data = (
            json.dumps(asdict(self), ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )
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
                os.link(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
