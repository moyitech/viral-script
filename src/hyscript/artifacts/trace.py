"""Serializable generation trace frozen before offline evaluation starts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
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
    writes a separate result that references ``run_id``.
    """

    run_id: str
    created_at: str
    task: dict[str, Any]
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

    def write_json(self, path: Path) -> None:
        """Write the generation artifact as UTF-8 JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
