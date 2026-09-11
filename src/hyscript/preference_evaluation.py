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
    mapping = read_rows(archive / "source_mapping.csv")
    if len(mapping) != manifest["pair_count"]:
        raise ValueError("Manifest pair count mismatch")
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
    reviewer_specs = manifest["reviewers"]
    if len(reviewer_specs) != manifest["reviewer_count"]:
        raise ValueError("Manifest reviewer count mismatch")
    keyed(reviewer_specs, "reviewer_id")
    if len({s["ratings"] for s in reviewer_specs}) != len(reviewer_specs):
        raise ValueError("Reviewers must have distinct ratings files")
    all_reviews = {}
    for spec in reviewer_specs:
        for field in ("ratings", "workbook"):
            if spec[field] not in manifest["sha256"]:
                raise ValueError(f"Unhashed reviewer input: {spec[field]}")
        all_reviews[spec["reviewer_id"]] = read_rows(archive / spec["ratings"])
    result = summarize_reviewers(all_reviews, mapping)
    result["both_gates_pass"] = summarize_reviewers(
        {k: [r for r in rows if r["blind_id"] in eligible] for k, rows in all_reviews.items()},
        [r for r in mapping if r["blind_id"] in eligible],
    )
    return result


def summarize_reviewers(reviews: dict[str, list[dict[str, str]]], mapping: list[dict[str, str]]) -> dict:
    """Overall preference counts only; repeated ratings are not independent pairs."""
    if not reviews:
        raise ValueError("At least one reviewer is required")
    summaries = {}
    combined = Counter()
    categories = ("editorial_candidates", "single_shot", "tie", "unable", "missing")
    for reviewer_id, rows in reviews.items():
        summary = summarize_preferences(rows, mapping)
        counts = {key: summary["counts"].get(key, 0) for key in categories}
        summaries[reviewer_id] = {
            "judgments": summary["pairs"], "counts": counts,
            "preference_denominator": summary["preference_denominator"],
            "editorial_preference_rate": summary["editorial_preference_rate"],
        }
        combined.update(counts)
    denominator = sum(combined[k] for k in categories[:3])
    rates = [s["editorial_preference_rate"] for s in summaries.values()]
    return {
        "reviewer_count": len(reviews), "unique_pairs": len(mapping),
        "topic_count": len({r["topic_cluster_id"] for r in mapping}),
        "reviewers": summaries,
        "combined": {
            "judgments": sum(combined.values()), "counts": dict(combined),
            "preference_denominator": denominator,
            "editorial_preference_rate": combined["editorial_candidates"] / denominator if denominator else None,
            "equal_reviewer_editorial_rate": sum(rates) / len(rates) if all(r is not None for r in rates) else None,
        },
    }
