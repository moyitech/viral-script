"""Offline paired-preference statistics; no generation or external services."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def keyed(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    result = {row[key]: row for row in rows}
    if len(result) != len(rows) or "" in result:
        raise ValueError(f"Duplicate or empty {key}")
    return result


def summarize_preferences(
    reviews: list[dict[str, str]], mapping: list[dict[str, str]]
) -> dict:
    """Decode all votes, keeping abstentions distinct from missing observations."""
    responses = keyed(reviews, "blind_id")
    sources = keyed(mapping, "blind_id")
    if responses.keys() != sources.keys():
        raise ValueError("Review and mapping IDs do not match")
    counts = Counter()
    usability = {source: Counter() for source in ("editorial_candidates", "single_shot")}
    lengths = defaultdict(Counter)
    clusters = defaultdict(list)
    allowed_use = {"可直接采用", "小改可用", "需大改", "不可用", "无法判断", ""}
    for blind_id, review in responses.items():
        pair = sources[blind_id]
        if {pair["A_source"], pair["B_source"]} != set(usability):
            raise ValueError(f"Invalid source pair: {blind_id}")
        choice = review["preference"]
        if choice not in {"A", "B", "无明显偏好", "无法判断", ""}:
            raise ValueError(f"Invalid preference: {blind_id}")
        outcome = pair[f"{choice}_source"] if choice in {"A", "B"} else {
            "无明显偏好": "tie", "无法判断": "unable", "": "missing"
        }[choice]
        counts[outcome] += 1
        lengths[pair["target_length"]][outcome] += 1
        # Every topic stays a cluster even if all its observations are missing.
        cluster = clusters[pair["topic_cluster_id"]]
        if outcome not in {"unable", "missing"}:
            cluster.append(int(outcome == "editorial_candidates"))
        for side in ("A", "B"):
            value = review[f"{side}_usability"]
            if value not in allowed_use:
                raise ValueError(f"Invalid usability: {blind_id}")
            usability[pair[f"{side}_source"]][value or "missing"] += 1
    denominator = sum(counts[k] for k in ("editorial_candidates", "single_shot", "tie"))
    return {
        "pairs": len(responses), "counts": dict(counts),
        "preference_denominator": denominator,
        "editorial_preference_rate": counts["editorial_candidates"] / denominator if denominator else None,
        "by_target_length": {k: dict(v) for k, v in sorted(lengths.items())},
        "usability": {k: dict(v) for k, v in usability.items()},
        "clusters": dict(clusters),
    }


def bootstrap_interval(clusters: dict[str, list[int]], *, seed: int, repeats: int) -> list[float]:
    """Percentile interval for pooled votes, resampling whole topic clusters."""
    if repeats < 2 or not clusters:
        raise ValueError("Bootstrap requires clusters and at least two repeats")
    groups = [(sum(clusters[k]), len(clusters[k])) for k in sorted(clusters)]
    rng = random.Random(seed)
    samples = []
    for _ in range(repeats):
        selected = rng.choices(groups, k=len(groups))
        denominator = sum(n for _, n in selected)
        if denominator:
            samples.append(sum(w for w, _ in selected) / denominator)
    if not samples:
        raise ValueError("No evaluable preferences")
    samples.sort()

    def quantile(p: float) -> float:
        position = (len(samples) - 1) * p
        lower = int(position)
        upper = min(lower + 1, len(samples) - 1)
        return samples[lower] + (samples[upper] - samples[lower]) * (position - lower)

    return [quantile(.025), quantile(.975)]


def recompute(archive: Path, paired_results: Path) -> dict:
    manifest = json.loads((archive / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((archive / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Archive hash mismatch: {name}")
    if hashlib.sha256(paired_results.read_bytes()).hexdigest() != manifest["paired_results_sha256"]:
        raise ValueError("Paired-results hash mismatch")
    reviews = read_rows(archive / "ratings.csv")
    mapping = read_rows(archive / "source_mapping.csv")
    result = summarize_preferences(reviews, mapping)
    clusters = result.pop("clusters")
    result["bootstrap"] = {
        "method": "Python random.Random.choices; topic cluster percentile bootstrap",
        "seed": 20260909, "repeats": 20000, "topic_count": len(clusters),
        "editorial_rate_ci95": bootstrap_interval(clusters, seed=20260909, repeats=20000),
    }
    pairs = keyed(read_rows(paired_results), "task_id")
    eligible = set()
    for row in mapping:
        pair = pairs[row["task_id"]]
        for side in ("A", "B"):
            prefix = "baseline" if row[f"{side}_source"] == "editorial_candidates" else "candidate"
            if row[f"{side}_run_id"] != pair[f"{prefix}_run_id"]:
                raise ValueError("Mapping run ID does not match paired results")
        if pair["baseline_gate_failed"] == pair["candidate_gate_failed"] == "False":
            eligible.add(row["blind_id"])
    gated = summarize_preferences(
        [r for r in reviews if r["blind_id"] in eligible],
        [r for r in mapping if r["blind_id"] in eligible],
    )
    gated.pop("clusters")
    result["both_gates_pass"] = gated
    return result
