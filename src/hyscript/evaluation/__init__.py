"""Offline-only scoring, Judge, and aggregation components."""

from .aggregate import combine_evaluations, summarize_batch
from .citation_verification import (
    CitationVerificationConfig,
    CitationVerificationConflictError,
    CitationVerificationError,
    CitationVerifier,
    run_citation_verification_formal_validation,
    run_citation_verification_validation,
)
from .io import FrozenTrace, TraceInputError, load_frozen_trace
from .judge import (
    Hy3JudgeEvaluator,
    JudgeConfig,
    JudgeEvaluationError,
    JudgeInputError,
)
from .models import (
    DimensionScore,
    EvaluationFingerprint,
    EvaluationRecord,
    EvaluatorFingerprint,
    Finding,
)
from .rubric import Rubric, RubricError, load_rubric
from .rules import RuleConfig, RuleEvaluator
from .reward_hacking import (
    RewardHackingConfig,
    RewardHackingConflictError,
    RewardHackingDetector,
    RewardHackingEvaluationError,
    run_reward_hacking_formal_validation,
    run_reward_hacking_validation,
)
from .runner import (
    BatchEvaluationConfig,
    BatchEvaluationResult,
    BatchEvaluationRunner,
    EvaluationConflictError,
    TraceOutcome,
)

__all__ = [
    "BatchEvaluationConfig",
    "BatchEvaluationResult",
    "BatchEvaluationRunner",
    "CitationVerificationConfig",
    "CitationVerificationConflictError",
    "CitationVerificationError",
    "CitationVerifier",
    "DimensionScore",
    "EvaluationConflictError",
    "EvaluationFingerprint",
    "EvaluationRecord",
    "EvaluatorFingerprint",
    "Finding",
    "FrozenTrace",
    "Hy3JudgeEvaluator",
    "JudgeConfig",
    "JudgeEvaluationError",
    "JudgeInputError",
    "Rubric",
    "RubricError",
    "RuleConfig",
    "RuleEvaluator",
    "RewardHackingConfig",
    "RewardHackingConflictError",
    "RewardHackingDetector",
    "RewardHackingEvaluationError",
    "TraceInputError",
    "TraceOutcome",
    "combine_evaluations",
    "load_frozen_trace",
    "load_rubric",
    "run_reward_hacking_validation",
    "run_reward_hacking_formal_validation",
    "run_citation_verification_formal_validation",
    "run_citation_verification_validation",
    "summarize_batch",
]
