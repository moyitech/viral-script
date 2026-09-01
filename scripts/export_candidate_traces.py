"""Export frozen editorial candidates as scoreable shadow traces."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create offline-evaluation traces for candidates already frozen "
            "inside editorial generation traces. No generation or scoring runs."
        )
    )
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--parent-run-id",
        action="append",
        default=[],
        help="Export only this parent run id; repeat for multiple parents.",
    )
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
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
        os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _read_trace(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load trace: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("run_id"), str):
        raise ValueError(f"Invalid trace: {path}")
    return payload


def _candidate_run_id(parent_run_id: str, candidate_id: str) -> str:
    digest = hashlib.sha256(
        f"{parent_run_id}\0{candidate_id}".encode("utf-8")
    ).hexdigest()[:20]
    return f"run-candidate-{digest}"


def _shadow_trace(
    parent: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "candidate_id",
        "strategy",
        "outline",
        "script_text",
        "reference_ids",
        "character_count",
        "prompt_version",
    }
    if not required.issubset(candidate):
        raise ValueError("Frozen candidate is missing required fields.")
    candidate_id = candidate["candidate_id"]
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("Frozen candidate has an invalid candidate_id.")

    shadow = deepcopy(parent)
    parent_run_id = parent["run_id"]
    shadow_run_id = _candidate_run_id(parent_run_id, candidate_id)
    shadow["run_id"] = shadow_run_id
    artifact = dict(shadow.get("script_artifact") or {})
    artifact.update(
        {
            "outline": candidate["outline"],
            "script_text": candidate["script_text"],
            "character_count": candidate["character_count"],
            "prompt_version": candidate["prompt_version"],
            "reference_ids": candidate["reference_ids"],
            "generation_attempt_count": 1,
            "llm_usages": [],
            "generation_mode": "single",
            "generation_candidates": [],
            "selected_candidate_ids": [],
            "editor_prompt_version": None,
            "editor_attempt_count": 0,
            "length_repair_attempted": False,
        }
    )
    target = (shadow.get("task") or {}).get("target_length")
    actual = candidate["character_count"]
    artifact["length_within_tolerance"] = bool(
        isinstance(target, int)
        and isinstance(actual, int)
        and abs(actual - target) / target <= 0.10
    )
    shadow["script_artifact"] = artifact

    retained_ids = set(candidate["reference_ids"])
    shadow["selected_evidence"] = [
        item
        for item in shadow.get("selected_evidence", [])
        if item.get("evidence_id") in retained_ids
    ]
    config = dict(shadow.get("config") or {})
    experiment = dict(config.get("experiment") or {})
    experiment.update(
        {
            "mode": "frozen_editorial_candidate_shadow",
            "parent_run_id": parent_run_id,
            "candidate_id": candidate_id,
            "candidate_strategy": candidate["strategy"],
        }
    )
    config["experiment"] = experiment
    shadow["config"] = config
    lineage = dict(shadow.get("lineage") or {})
    lineage.update(
        {
            "parent_run_id": parent_run_id,
            "candidate_id": candidate_id,
            "candidate_strategy": candidate["strategy"],
            "script_reference_ids": list(candidate["reference_ids"]),
        }
    )
    shadow["lineage"] = lineage
    return shadow


def run(args: argparse.Namespace) -> int:
    trace_dir = args.trace_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not trace_dir.is_dir():
        raise ValueError(f"Trace directory does not exist: {trace_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = set(args.parent_run_id)
    records: list[dict[str, Any]] = []
    seen_parents: set[str] = set()
    for path in sorted(trace_dir.rglob("*.json")):
        parent = _read_trace(path)
        parent_run_id = parent["run_id"]
        if requested and parent_run_id not in requested:
            continue
        candidates = (parent.get("script_artifact") or {}).get(
            "generation_candidates"
        )
        if not isinstance(candidates, list) or not candidates:
            continue
        seen_parents.add(parent_run_id)
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError(f"Trace contains an invalid candidate: {path}")
            shadow = _shadow_trace(parent, candidate)
            candidate_id = candidate["candidate_id"]
            destination = output_dir / (
                f"{path.stem}-{candidate_id}-{shadow['run_id']}.json"
            )
            _write_json(destination, shadow)
            records.append(
                {
                    "parent_run_id": parent_run_id,
                    "candidate_id": candidate_id,
                    "candidate_run_id": shadow["run_id"],
                    "trace": str(destination),
                }
            )
    missing = sorted(requested - seen_parents)
    if missing:
        raise ValueError(f"Requested parent runs have no candidates: {', '.join(missing)}")
    if not records:
        raise ValueError("No frozen editorial candidates were found.")
    _write_json(
        output_dir / "manifest.json",
        {
            "mode": "frozen_editorial_candidate_shadow_export",
            "source_trace_dir": str(trace_dir),
            "count": len(records),
            "records": records,
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "count": len(records)}))
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    raise SystemExit(main())
