"""Serializable, score-free artifacts produced by the generation workflow."""

from .generation import build_generation_trace
from .trace import RunTrace, TraceSearchResult

__all__ = ["RunTrace", "TraceSearchResult", "build_generation_trace"]
