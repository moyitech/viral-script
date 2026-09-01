# Frozen generation artifacts

The offline Runner saves one generation artifact per dataset item before any
scoring starts. Each artifact records the task and configuration, generated
queries, raw normalized search results, timestamps, selected background
references, citation metadata, script, errors, latency, and token usage. It must not contain an
evaluation score.

Treat a completed generation artifact as immutable. Evaluation outputs are
stored under `eval/results/runs/` and reference it by `run_id`, so generation
and scoring can be resumed or replayed independently. Generated runs belong in
`eval/traces/runs/` and are ignored by Git; commit only small, redacted examples
when needed for documentation.

Evaluation-ready traces currently use schema version `1.0`. Evidence and claim
lists may be empty. When present, their ids must be unique and claims must mark
at least one core claim. Stable fields are:

- `task.topic`, with optional `target_length` and `forbidden_phrases`;
- `script_artifact.script_text`;
- `selected_evidence[*].evidence_id` for the background references selected by
  the writer (the historical field name is retained for schema compatibility);
- `claims[*].claim_id`, `is_core`, and `evidence_ids` when claims are present.

`example_trace.json` is a small schema demonstration. Its example.com source is
not research evidence and must not be included in formal experiments.
