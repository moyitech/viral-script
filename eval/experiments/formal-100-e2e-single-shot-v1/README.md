# Formal 100-topic frozen-research single-shot experiment v1

This paired experiment reuses the 100 frozen research snapshots from
`formal-100-v1` and expands them into 280/450/700 targets. Each output has one
content-generation call. JSON-only repair calls are unbounded but must preserve
the frozen outline, script text, and reference ids exactly. Successful traces
are immutable and reruns select only missing tasks.

No query-planning or Tavily call is made by this replay. The source research was
collected with Tavily concurrency eight. Hy3 generation and both Judge rounds
are configured with a client-side limit of 512.

```bash
uv run --no-sync python scripts/run_end_to_end_experiment.py prepare
uv run --no-sync python scripts/run_end_to_end_experiment.py generate
uv run --no-sync python scripts/run_end_to_end_experiment.py score
uv run --no-sync python scripts/run_end_to_end_experiment.py repeat
uv run --no-sync python scripts/run_end_to_end_experiment.py report
```

The completed comparison is in `report/comparison.md`; its paired table contains
exactly 300 rows in `report/paired_results.csv`.
