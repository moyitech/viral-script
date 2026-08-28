"""Small strict-JSON helpers shared by generation agents."""

from __future__ import annotations

import json
import re
from typing import Any


class StructuredOutputError(ValueError):
    """Raised when a model response violates a structured-output contract."""


def json_object(response: str) -> dict[str, Any]:
    """Parse a JSON object, accepting one optional Markdown JSON fence."""

    normalized = response.strip()
    fence_match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        normalized,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        normalized = fence_match.group(1)
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError:
        raise StructuredOutputError("Response is not valid JSON.") from None
    if not isinstance(payload, dict):
        raise StructuredOutputError("Response must be a JSON object.")
    return payload


def required_text(
    payload: dict[str, Any],
    name: str,
    *,
    max_length: int,
) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise StructuredOutputError(f"Response is missing {name}.")
    normalized = value.strip()
    if len(normalized) > max_length:
        raise StructuredOutputError(f"Response contains an overlong {name}.")
    return normalized


def text_list(
    payload: dict[str, Any],
    name: str,
    *,
    minimum: int,
    maximum: int,
    item_max_length: int,
) -> tuple[str, ...]:
    value = payload.get(name)
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise StructuredOutputError(f"Response contains an invalid {name} list.")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise StructuredOutputError(f"Response contains an invalid {name} item.")
        normalized = item.strip()
        if len(normalized) > item_max_length:
            raise StructuredOutputError(f"Response contains an overlong {name} item.")
        if normalized not in items:
            items.append(normalized)
    if len(items) < minimum:
        raise StructuredOutputError(f"Response contains duplicate {name} items.")
    return tuple(items)
