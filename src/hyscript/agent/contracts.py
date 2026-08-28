"""Provider-independent contracts for research-backed script generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from hyscript.llm import LLMCallUsage
from hyscript.search import SearchResponse

ResearchStatus = Literal["ready", "insufficient_evidence"]
ClaimSupportStatus = Literal["supported", "conflicting", "unsupported"]


def _normalized_text(value: str, name: str, *, max_length: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty.")
    normalized = value.strip()
    if len(normalized) > max_length:
        raise ValueError(f"{name} is too long.")
    return normalized


@dataclass(frozen=True, slots=True)
class ScriptTask:
    """One selected topic plus request-scoped oral-script constraints."""

    topic: str
    target_length: int = 450
    angle: str = ""
    audience: str = "普通中文短视频观众"
    platform: str = "短视频"
    style: str = "自然、克制、有信息量"
    forbidden_phrases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "topic",
            _normalized_text(self.topic, "topic", max_length=200),
        )
        if isinstance(self.target_length, bool) or not 50 <= self.target_length <= 5000:
            raise ValueError("target_length must be between 50 and 5000.")
        angle = self.angle.strip()
        if len(angle) > 300:
            raise ValueError("angle is too long.")
        object.__setattr__(self, "angle", angle)
        for name in ("audience", "platform", "style"):
            object.__setattr__(
                self,
                name,
                _normalized_text(getattr(self, name), name, max_length=120),
            )
        normalized_phrases = tuple(
            dict.fromkeys(
                phrase.strip()
                for phrase in self.forbidden_phrases
                if isinstance(phrase, str) and phrase.strip()
            )
        )
        if len(normalized_phrases) > 30:
            raise ValueError("forbidden_phrases must contain at most 30 items.")
        if any(len(phrase) > 80 for phrase in normalized_phrases):
            raise ValueError("forbidden_phrases contains an overlong item.")
        object.__setattr__(self, "forbidden_phrases", normalized_phrases)


@dataclass(frozen=True, slots=True)
class PlannedQuery:
    """One live-search query and the information need it serves."""

    query: str
    purpose: str


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """Hy3-authored research goal and bounded initial queries."""

    goal: str
    must_verify: tuple[str, ...]
    queries: tuple[PlannedQuery, ...]
    current_date: str | None = None


@dataclass(frozen=True, slots=True)
class Evidence:
    """An exact source excerpt selected to support one or more claims."""

    evidence_id: str
    result_ref: str
    title: str
    url: str
    excerpt: str
    source_query: str
    published_at: str | None = None
    content_hash: str | None = None
    score: float | None = None

    def __post_init__(self) -> None:
        parsed = urlparse(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Evidence URL must be HTTP(S).")


@dataclass(frozen=True, slots=True)
class Claim:
    """A candidate factual claim linked to selected evidence."""

    claim_id: str
    text: str
    evidence_ids: tuple[str, ...]
    is_core: bool
    support_status: ClaimSupportStatus = "supported"


@dataclass(frozen=True, slots=True)
class ResearchOutcome:
    """Research result returned before any script generation or scoring."""

    status: ResearchStatus
    query_plan: QueryPlan
    search_responses: tuple[SearchResponse, ...]
    evidence: tuple[Evidence, ...]
    claims: tuple[Claim, ...]
    errors: tuple[str, ...]
    query_plan_prompt_version: str
    evidence_prompt_version: str
    llm_request_count: int
    search_request_count: int
    executed_queries: tuple[PlannedQuery, ...] = ()
    llm_usages: tuple[LLMCallUsage, ...] = ()


@dataclass(frozen=True, slots=True)
class ClaimUsage:
    """An exact script span showing where one candidate claim was used."""

    claim_id: str
    script_quote: str


@dataclass(frozen=True, slots=True)
class ScriptArtifact:
    """Clean oral script plus separately retained claim lineage."""

    outline: tuple[str, ...]
    script_text: str
    claim_usages: tuple[ClaimUsage, ...]
    character_count: int
    prompt_version: str
    generation_attempt_count: int
    llm_usages: tuple[LLMCallUsage, ...] = ()
