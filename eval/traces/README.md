# Frozen generation artifacts

The offline Runner saves one generation artifact per dataset item before any
scoring starts. Each artifact records the task and configuration, generated
queries, raw normalized search results, timestamps, selected evidence,
citations, script, errors, latency, and token usage. It must not contain an
evaluation score.

Treat a completed generation artifact as immutable. Evaluation outputs are
stored under `eval/results/runs/` and reference it by `run_id`, so generation
and scoring can be resumed or replayed independently. Generated runs belong in
`eval/traces/runs/` and are ignored by Git; commit only small, redacted examples
when needed for documentation.
