"""Versioned, strictly validated scoring rubrics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any


class RubricError(ValueError):
    """Raised when a rubric file does not match the supported schema."""


@dataclass(frozen=True, slots=True)
class RubricDimension:
    """One quality dimension and its observable 0-4 anchors."""

    dimension_id: str
    name: str
    description: str
    anchors: tuple[str, str, str, str, str]
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class Rubric:
    """Loaded rubric plus the hash of its exact source bytes."""

    rubric_id: str
    version: str
    dimensions: tuple[RubricDimension, ...]
    judge_gate_codes: tuple[str, ...]
    sha256: str

    @property
    def dimension_ids(self) -> tuple[str, ...]:
        return tuple(dimension.dimension_id for dimension in self.dimensions)

    def render_dimensions(self) -> str:
        """Render concise dimension anchors for the Judge prompt."""

        sections: list[str] = []
        for dimension in self.dimensions:
            anchors = "\n".join(
                f"  {score} 分：{description}"
                for score, description in enumerate(dimension.anchors)
            )
            sections.append(
                f"- `{dimension.dimension_id}` / {dimension.name}\n"
                f"  判断重点：{dimension.description}\n{anchors}"
            )
        return "\n\n".join(sections)


def _required_text(payload: dict[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RubricError(f"{context}.{key} must be a non-empty string.")
    return value.strip()


def _parse_dimension(payload: Any, index: int) -> RubricDimension:
    context = f"dimensions[{index}]"
    if not isinstance(payload, dict):
        raise RubricError(f"{context} must be an object.")
    allowed = {"id", "name", "description", "anchors", "weight"}
    extra = set(payload) - allowed
    if extra:
        raise RubricError(f"{context} has unsupported fields: {sorted(extra)}")

    anchors_payload = payload.get("anchors")
    if not isinstance(anchors_payload, dict):
        raise RubricError(f"{context}.anchors must be an object.")
    if set(anchors_payload) != {"0", "1", "2", "3", "4"}:
        raise RubricError(f"{context}.anchors must define exactly scores 0 through 4.")
    anchors = tuple(
        _required_text(anchors_payload, str(score), f"{context}.anchors")
        for score in range(5)
    )

    weight = payload.get("weight", 1.0)
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or weight <= 0
    ):
        raise RubricError(f"{context}.weight must be greater than zero.")
    return RubricDimension(
        dimension_id=_required_text(payload, "id", context),
        name=_required_text(payload, "name", context),
        description=_required_text(payload, "description", context),
        anchors=anchors,  # type: ignore[arg-type]
        weight=float(weight),
    )


def parse_rubric(payload: Any, *, sha256: str) -> Rubric:
    """Parse an already-decoded rubric payload."""

    if not isinstance(payload, dict):
        raise RubricError("rubric root must be an object.")
    allowed = {"schema_version", "rubric_id", "version", "dimensions", "judge_gates"}
    extra = set(payload) - allowed
    if extra:
        raise RubricError(f"rubric has unsupported fields: {sorted(extra)}")
    if payload.get("schema_version") != "1.0":
        raise RubricError("rubric.schema_version must be '1.0'.")

    dimensions_payload = payload.get("dimensions")
    if not isinstance(dimensions_payload, list) or not dimensions_payload:
        raise RubricError("rubric.dimensions must be a non-empty list.")
    dimensions = tuple(
        _parse_dimension(dimension, index)
        for index, dimension in enumerate(dimensions_payload)
    )
    dimension_ids = [dimension.dimension_id for dimension in dimensions]
    if len(set(dimension_ids)) != len(dimension_ids):
        raise RubricError("rubric dimension ids must be unique.")

    gates_payload = payload.get("judge_gates")
    if not isinstance(gates_payload, list) or not gates_payload:
        raise RubricError("rubric.judge_gates must be a non-empty list.")
    gates: list[str] = []
    for index, gate in enumerate(gates_payload):
        if not isinstance(gate, str) or not gate.strip():
            raise RubricError(
                f"rubric.judge_gates[{index}] must be a non-empty string."
            )
        gates.append(gate.strip())
    if len(set(gates)) != len(gates):
        raise RubricError("rubric judge gate codes must be unique.")

    return Rubric(
        rubric_id=_required_text(payload, "rubric_id", "rubric"),
        version=_required_text(payload, "version", "rubric"),
        dimensions=dimensions,
        judge_gate_codes=tuple(gates),
        sha256=sha256,
    )


def load_rubric(path: Path) -> Rubric:
    """Read and validate a UTF-8 JSON rubric."""

    try:
        content = path.read_bytes()
    except OSError as exc:
        raise RubricError(f"Could not read rubric: {path}") from exc
    digest = hashlib.sha256(content).hexdigest()
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RubricError(f"Rubric is not valid UTF-8 JSON: {path}") from exc
    return parse_rubric(payload, sha256=digest)


__all__ = [
    "Rubric",
    "RubricDimension",
    "RubricError",
    "load_rubric",
    "parse_rubric",
]
