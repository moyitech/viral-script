"""Serializable, score-free artifacts produced by the generation workflow."""

from .generation import build_generation_trace
from .research_snapshot import (
    ResearchSnapshotError,
    load_research_outcome,
    research_outcome_from_dict,
)
from .trace import RunTrace, TraceSearchResult

__all__ = [
    "ResearchSnapshotError",
    "RunTrace",
    "TraceSearchResult",
    "build_generation_trace",
    "load_research_outcome",
    "research_outcome_from_dict",
]
