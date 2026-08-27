"""Deterministic checks that run before any paid Judge request."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any

from .io import FrozenTrace
from .models import (
    EvaluationRecord,
    EvaluatorInfo,
    Finding,
    RubricRef,
    new_evaluation_id,
    utc_now_iso,
)
from .rubric import Rubric

RULE_EVALUATOR_VERSION = "1.1.0"
RULE_EVALUATOR_NAME = "deterministic-script-rules"


@dataclass(frozen=True, slots=True)
class RuleConfig:
    """Thresholds kept explicit so Pilot changes can be versioned."""

    length_tolerance_ratio: float = 0.10
    repeated_character_threshold: int = 10
    repeated_sentence_threshold: int = 3

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.length_tolerance_ratio)
            or not 0 <= self.length_tolerance_ratio < 1
        ):
            raise ValueError("length_tolerance_ratio must be in [0, 1).")
        if self.repeated_character_threshold < 2:
            raise ValueError("repeated_character_threshold must be at least 2.")
        if self.repeated_sentence_threshold < 2:
            raise ValueError("repeated_sentence_threshold must be at least 2.")


_META_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("reasoning_leakage", re.compile(r"</?think(?:\s[^>]*)?>", re.IGNORECASE)),
    (
        "non_script_analysis",
        re.compile(
            r"(?:^|[\r\n]|</think>)\s*(?:#{1,6}\s*)?"
            r"(?:开场钩子|关键点剖析|结构分析|写作思路|修改说明|"
            r"选题匹配度|口播流畅度|自我评价)\s*[:：]",
            re.IGNORECASE,
        ),
    ),
    (
        "meta_evaluation",
        re.compile(
            r"(?:这|本)(?:篇|段)?(?:文案|口播).{0,18}(?:符合|满足|达到).{0,12}(?:要求|标准|高分|满分)"
        ),
    ),
)
_BODY_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_INLINE_CITATION = re.compile(r"(?:\[\d+\]|【\d+】)")
_MARKDOWN_HEADING = re.compile(r"(?m)^\s{0,3}#{1,6}\s+\S+")
_SENTENCE_SPLIT = re.compile(r"[。！？!?；;\n]+")


def count_script_characters(text: str) -> int:
    """Count non-whitespace Unicode characters in the exact delivered body."""

    return sum(1 for character in text if not character.isspace())


def _finding_for_match(code: str, message: str, match: re.Match[str]) -> Finding:
    return Finding(
        code=code,
        severity="gate",
        message=message,
        details={
            "start": match.start(),
            "end": match.end(),
            "text": match.group(0)[:80],
        },
    )


def _target_length(task: dict[str, Any]) -> int | None:
    value = task.get("target_length")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _forbidden_phrases(task: dict[str, Any]) -> tuple[str, ...]:
    value = task.get("forbidden_phrases", [])
    if not isinstance(value, list):
        return ()
    return tuple(
        phrase.strip() for phrase in value if isinstance(phrase, str) and phrase.strip()
    )


def _claim_evidence_metrics(
    trace: FrozenTrace,
) -> tuple[dict[str, Any], list[Finding]]:
    findings: list[Finding] = []
    evidence_ids: list[str] = []
    for evidence in trace.selected_evidence:
        evidence_id = evidence.get("evidence_id")
        if isinstance(evidence_id, str) and evidence_id.strip():
            evidence_ids.append(evidence_id.strip())
    duplicate_evidence_ids = sorted(
        evidence_id for evidence_id, count in Counter(evidence_ids).items() if count > 1
    )
    if duplicate_evidence_ids:
        findings.append(
            Finding(
                code="duplicate_evidence_id",
                severity="warning",
                message="Frozen evidence contains duplicate evidence ids.",
                details={"evidence_ids": duplicate_evidence_ids},
            )
        )
    known_evidence_ids = set(evidence_ids)

    core_claims = 0
    supported_core_claims = 0
    cited_claims = 0
    invalid_references: dict[str, list[str]] = {}
    unsupported_core_ids: list[str] = []
    for index, claim in enumerate(trace.claims):
        claim_id_value = claim.get("claim_id")
        claim_id = (
            claim_id_value.strip()
            if isinstance(claim_id_value, str) and claim_id_value.strip()
            else f"claim-index-{index}"
        )
        is_core = claim.get("is_core") is True
        if is_core:
            core_claims += 1

        references = claim.get("evidence_ids", [])
        if not isinstance(references, list):
            references = []
            findings.append(
                Finding(
                    code="invalid_claim_evidence_ids",
                    severity="warning",
                    message="A claim has a non-list evidence_ids field.",
                    details={"claim_id": claim_id},
                )
            )
        normalized_references = [
            value.strip()
            for value in references
            if isinstance(value, str) and value.strip()
        ]
        if normalized_references:
            cited_claims += 1
        missing = sorted(set(normalized_references) - known_evidence_ids)
        if missing:
            invalid_references[claim_id] = missing
        valid_references = set(normalized_references) & known_evidence_ids
        support_status = claim.get("support_status")
        explicitly_unsupported = isinstance(support_status, str) and support_status in {
            "unsupported",
            "contradicted",
        }
        if is_core and valid_references and not explicitly_unsupported:
            supported_core_claims += 1
        elif is_core:
            unsupported_core_ids.append(claim_id)

    if invalid_references:
        findings.append(
            Finding(
                code="fabricated_citation",
                severity="gate",
                message="One or more claims reference evidence ids absent from the frozen trace.",
                details={"claims": invalid_references},
            )
        )
    if unsupported_core_ids:
        findings.append(
            Finding(
                code="unsupported_core_claim",
                severity="gate",
                message="One or more core claims lack valid supporting evidence.",
                details={"claim_ids": unsupported_core_ids},
            )
        )
    if trace.claims and core_claims == 0:
        findings.append(
            Finding(
                code="core_claim_mapping_missing",
                severity="gate",
                message="Claims are present but none is marked as a core claim.",
            )
        )
    if not trace.claims:
        findings.append(
            Finding(
                code="claim_mapping_missing",
                severity="warning",
                message="No claim-to-evidence mapping is available for deterministic checks.",
            )
        )

    claim_count = len(trace.claims)
    metrics: dict[str, Any] = {
        "evidence_count": len(trace.selected_evidence),
        "claim_count": claim_count,
        "core_claim_count": core_claims,
        "cited_claim_count": cited_claims,
        "claim_citation_coverage": cited_claims / claim_count if claim_count else None,
        "supported_core_claim_count": supported_core_claims,
        "core_claim_support_rate": (
            supported_core_claims / core_claims if core_claims else None
        ),
    }
    return metrics, findings


class RuleEvaluator:
    """Run deterministic, explainable checks against one frozen trace."""

    def __init__(self, config: RuleConfig | None = None) -> None:
        self.config = config or RuleConfig()

    def evaluate(self, trace: FrozenTrace, rubric: Rubric) -> EvaluationRecord:
        text = trace.script_text
        findings: list[Finding] = []
        char_count = count_script_characters(text)
        target_length = _target_length(trace.task)
        length_deviation_ratio: float | None = None
        length_within_tolerance: bool | None = None

        if not text.strip():
            findings.append(
                Finding(
                    code="empty_script",
                    severity="gate",
                    message="The generated script body is empty.",
                )
            )
        if target_length is None:
            findings.append(
                Finding(
                    code="target_length_missing",
                    severity="warning",
                    message="task.target_length is absent or invalid; length accuracy is not evaluable.",
                )
            )
        else:
            length_deviation_ratio = abs(char_count - target_length) / target_length
            length_within_tolerance = (
                length_deviation_ratio <= self.config.length_tolerance_ratio
            )
            if not length_within_tolerance:
                findings.append(
                    Finding(
                        code="length_out_of_range",
                        severity="warning",
                        message="Script length is outside the configured target range.",
                        details={
                            "actual": char_count,
                            "target": target_length,
                            "tolerance_ratio": self.config.length_tolerance_ratio,
                            "deviation_ratio": length_deviation_ratio,
                        },
                    )
                )

        for code, pattern in _META_PATTERNS:
            match = pattern.search(text)
            if match:
                findings.append(
                    _finding_for_match(
                        code,
                        "The delivered body contains model reasoning, self-evaluation, or non-script analysis.",
                        match,
                    )
                )

        repeated_pattern = re.compile(
            rf"(.)\1{{{self.config.repeated_character_threshold - 1},}}",
            re.DOTALL,
        )
        repeated_match = repeated_pattern.search(text)
        if repeated_match:
            findings.append(
                _finding_for_match(
                    "repetition_padding",
                    "The script contains an excessive run of one repeated character.",
                    repeated_match,
                )
            )

        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_SPLIT.split(text)
            if len(sentence.strip()) >= 8
        ]
        repeated_sentences = {
            sentence: count
            for sentence, count in Counter(sentences).items()
            if count >= self.config.repeated_sentence_threshold
        }
        if repeated_sentences:
            findings.append(
                Finding(
                    code="repetition_padding",
                    severity="gate",
                    message="The script repeats the same substantive sentence multiple times.",
                    details={
                        "sentences": [
                            {"text": sentence[:120], "count": count}
                            for sentence, count in sorted(repeated_sentences.items())
                        ]
                    },
                )
            )

        url_matches = _BODY_URL.findall(text)
        citation_matches = _INLINE_CITATION.findall(text)
        heading_matches = _MARKDOWN_HEADING.findall(text)
        if url_matches or citation_matches or heading_matches:
            findings.append(
                Finding(
                    code="non_body_formatting",
                    severity="warning",
                    message="The spoken body contains URLs, citation markers, or Markdown headings.",
                    details={
                        "url_count": len(url_matches),
                        "citation_marker_count": len(citation_matches),
                        "heading_count": len(heading_matches),
                    },
                )
            )

        forbidden_hits = [
            phrase for phrase in _forbidden_phrases(trace.task) if phrase in text
        ]
        if forbidden_hits:
            findings.append(
                Finding(
                    code="forbidden_phrase",
                    severity="gate",
                    message="The script contains task-level forbidden phrases.",
                    details={"phrases": forbidden_hits},
                )
            )

        claim_metrics, claim_findings = _claim_evidence_metrics(trace)
        findings.extend(claim_findings)
        metrics: dict[str, Any] = {
            "script_character_count": char_count,
            "target_length": target_length,
            "length_deviation_ratio": length_deviation_ratio,
            "length_within_tolerance": length_within_tolerance,
            "query_count": len(trace.queries),
            "search_result_count": trace.search_result_count,
            **claim_metrics,
        }
        return EvaluationRecord(
            evaluation_id=new_evaluation_id("rules"),
            run_id=trace.run_id,
            trace_sha256=trace.trace_sha256,
            created_at=utc_now_iso(),
            evaluator=EvaluatorInfo(
                kind="rules",
                name=RULE_EVALUATOR_NAME,
                version=RULE_EVALUATOR_VERSION,
            ),
            rubric=RubricRef(
                rubric_id=rubric.rubric_id,
                version=rubric.version,
                sha256=rubric.sha256,
            ),
            status="completed",
            summary=(
                "Deterministic gates triggered."
                if any(finding.severity == "gate" for finding in findings)
                else "Deterministic checks completed."
            ),
            metrics=metrics,
            findings=tuple(findings),
            metadata={
                "config": {
                    "length_tolerance_ratio": self.config.length_tolerance_ratio,
                    "repeated_character_threshold": self.config.repeated_character_threshold,
                    "repeated_sentence_threshold": self.config.repeated_sentence_threshold,
                }
            },
        )


__all__ = [
    "RULE_EVALUATOR_VERSION",
    "RULE_EVALUATOR_NAME",
    "RuleConfig",
    "RuleEvaluator",
    "count_script_characters",
]
