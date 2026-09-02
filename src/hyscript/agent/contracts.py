"""Provider-independent contracts for research-backed script generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from hyscript.llm import LLMCallUsage
from hyscript.search import SearchResponse

ResearchStatus = Literal["ready", "insufficient_evidence"]
TitleChainComponent = Literal[
    "subject_scope",
    "stated_context",
    "question_predicate",
]
TitleChainStatus = Literal["covered", "missing"]
ClaimSupportStatus = Literal["supported", "conflicting", "unsupported"]
EvidenceSourceType = Literal[
    "official_primary",
    "direct_terms",
    "primary_research",
    "authoritative_dataset",
    "reputable_reporting",
    "independent_secondary",
    "vendor_or_advocacy",
    "encyclopedia_social_personal",
    "unclassified",
]
ClaimKind = Literal[
    "rule_or_terms",
    "quantitative_state",
    "causal_effect",
    "case_event",
    "expert_opinion",
    "descriptive_context",
    "uncertainty_boundary",
]
ScriptGenerationMode = Literal["single", "editorial_candidates"]
CORE_SOURCE_TYPES_BY_CLAIM_KIND: dict[
    ClaimKind,
    frozenset[EvidenceSourceType],
] = {
    "rule_or_terms": frozenset({"official_primary", "direct_terms"}),
    "quantitative_state": frozenset(
        {"official_primary", "primary_research", "authoritative_dataset"}
    ),
    "causal_effect": frozenset(
        {"official_primary", "primary_research", "authoritative_dataset"}
    ),
    "case_event": frozenset({"official_primary", "reputable_reporting"}),
    "expert_opinion": frozenset(
        {
            "official_primary",
            "primary_research",
            "reputable_reporting",
        }
    ),
    "descriptive_context": frozenset(
        {
            "official_primary",
            "direct_terms",
            "primary_research",
            "authoritative_dataset",
            "reputable_reporting",
        }
    ),
    "uncertainty_boundary": frozenset(
        {
            "official_primary",
            "primary_research",
            "authoritative_dataset",
            "reputable_reporting",
            "independent_secondary",
        }
    ),
}
_EVIDENCE_SOURCE_TYPES = {
    "official_primary",
    "direct_terms",
    "primary_research",
    "authoritative_dataset",
    "reputable_reporting",
    "independent_secondary",
    "vendor_or_advocacy",
    "encyclopedia_social_personal",
    "unclassified",
}
_CLAIM_KINDS = {
    "rule_or_terms",
    "quantitative_state",
    "causal_effect",
    "case_event",
    "expert_opinion",
    "descriptive_context",
    "uncertainty_boundary",
}


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
        if isinstance(self.target_length, bool) or not isinstance(
            self.target_length, int
        ):
            raise ValueError("target_length must be an integer.")
        if not 50 <= self.target_length <= 5000:
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
class TitleChainPart:
    """One persisted audit decision for a topic-title component."""

    component: TitleChainComponent
    status: TitleChainStatus
    claim_ids: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if self.component not in {
            "subject_scope",
            "stated_context",
            "question_predicate",
        }:
            raise ValueError("TitleChainPart component is invalid.")
        if self.status not in {"covered", "missing"}:
            raise ValueError("TitleChainPart status is invalid.")
        if (
            not isinstance(self.claim_ids, tuple)
            or any(
                not isinstance(claim_id, str) or not claim_id.strip()
                for claim_id in self.claim_ids
            )
            or len(set(self.claim_ids)) != len(self.claim_ids)
        ):
            raise ValueError("TitleChainPart claim_ids are invalid.")
        if self.status == "covered" and not self.claim_ids:
            raise ValueError("Covered TitleChainPart requires claim_ids.")
        if self.status == "missing" and self.claim_ids:
            raise ValueError("Missing TitleChainPart cannot reference claims.")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("TitleChainPart reason must not be empty.")
        if len(self.reason) > 300:
            raise ValueError("TitleChainPart reason is too long.")


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
    source_type: EvidenceSourceType = "unclassified"
    source_scope: str = ""
    time_basis: str = ""

    def __post_init__(self) -> None:
        parsed = urlparse(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Evidence URL must be HTTP(S).")
        if self.source_type not in _EVIDENCE_SOURCE_TYPES:
            raise ValueError("Evidence source_type is invalid.")
        if not isinstance(self.source_scope, str) or len(self.source_scope) > 240:
            raise ValueError("Evidence source_scope is invalid.")
        if not isinstance(self.time_basis, str) or len(self.time_basis) > 160:
            raise ValueError("Evidence time_basis is invalid.")


@dataclass(frozen=True, slots=True)
class Claim:
    """A candidate factual claim linked to selected evidence."""

    claim_id: str
    text: str
    evidence_ids: tuple[str, ...]
    is_core: bool
    support_status: ClaimSupportStatus = "supported"
    claim_kind: ClaimKind = "descriptive_context"

    def __post_init__(self) -> None:
        if self.claim_kind not in _CLAIM_KINDS:
            raise ValueError("Claim claim_kind is invalid.")


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
    title_chain: tuple[TitleChainPart, ...] = ()


@dataclass(frozen=True, slots=True)
class ClaimUsage:
    """An exact script span showing where one candidate claim was used."""

    claim_id: str
    script_quote: str


@dataclass(frozen=True, slots=True)
class ScriptCandidate:
    """One background-informed draft retained before chief-editor synthesis."""

    candidate_id: str
    strategy: str
    outline: tuple[str, ...]
    script_text: str
    reference_ids: tuple[str, ...]
    character_count: int
    prompt_version: str


@dataclass(frozen=True, slots=True)
class ScriptArtifact:
    """Clean oral script plus separately retained scoring metadata.

    ``reference_ids`` records which background sources informed the draft.  The
    references are intentionally kept outside ``script_text`` so the spoken
    body stays clean while an offline evaluator can still inspect citations.
    ``claim_usages`` and grounding-review fields remain for loading historical
    evidence-grounded runs; the background-first generation path leaves them
    empty/disabled.
    """

    outline: tuple[str, ...]
    script_text: str
    claim_usages: tuple[ClaimUsage, ...]
    character_count: int
    prompt_version: str
    generation_attempt_count: int
    llm_usages: tuple[LLMCallUsage, ...] = ()
    grounding_review_attempt_count: int = 0
    grounding_review_status: Literal[
        "disabled",
        "accepted",
        "rejected",
        "fallback",
    ] = "disabled"
    grounding_review_prompt_version: str | None = None
    grounding_review_draft_text: str | None = None
    grounding_review_draft_character_count: int | None = None
    grounding_review_issues: tuple[str, ...] = ()
    grounding_review_failure_reason: str | None = None
    reference_ids: tuple[str, ...] = ()
    generation_mode: ScriptGenerationMode = "single"
    generation_candidates: tuple[ScriptCandidate, ...] = ()
    selected_candidate_ids: tuple[str, ...] = ()
    editor_prompt_version: str | None = None
    editor_attempt_count: int = 0
    length_within_tolerance: bool = True
    length_repair_attempted: bool = False
    final_rewrite_attempt_count: int = 0
    final_rewrite_prompt_version: str | None = None
    final_rewrite_draft_text: str | None = None
    final_rewrite_draft_character_count: int | None = None
