# Evaluation runners

Reusable offline experiment orchestration belongs here. A Runner has two
separable phases:

1. read the existing-topic dataset, invoke the shared generation workflow in
   batches, and freeze all intermediate artifacts;
2. read those frozen artifacts, run reward-hacking and citation gates before
   rule/Judge quality evaluation, and write
   results linked by `run_id` without modifying generation files.

Command-line wrappers belong in `scripts/`. Applications may invoke the shared
formal evaluator only after an explicit creator action against a frozen trace;
generation never scores automatically. Human review remains a separate diagnostic.

The scoring phase is implemented. Run deterministic rules without loading
`.env` or calling an external service:

```bash
uv run --no-sync python scripts/run_evaluation.py score \
  --trace eval/traces/example_trace.json \
  --evaluators rules \
  --output-dir /tmp/hyscript-eval
```

Full evaluation defaults to both gates followed by rules and the Hy3 Judge:

```bash
uv run --no-sync python scripts/run_evaluation.py score \
  --trace-dir eval/traces/runs/<batch-id> \
  --evaluators rules,judge \
  --output-dir eval/results/runs/<evaluation-id> \
  --concurrency 2
```

Both gates use Hy3; citation verification can also use Tavily. A blocked trace
gets reasons and no quality score. Provider errors remain incomplete and resumable.
Rules-only and Judge-only runs are component diagnostics, without a final eight-dimensional score.
The shared existing-topic generation workflow
is available through `run_live_batch.py`; the formal runner composes its
research-only mode with frozen-background length replay.

For the fixed 100-topic, 300-output submission experiment, use
`scripts/run_formal_experiment.py`. It creates immutable attempt directories,
selects one successful artifact per task into exact manifests, and keeps research,
generation, scoring, and reporting independently resumable.

Results resume only under the same input hashes and full evaluation
fingerprint. Rule-only runs never load `.env`; selecting `judge` loads Hy3
settings, records every format-repair attempt and accumulates usage across all
requests.

Existing completed detector records can be supplied with `--reward-gate-cache`
and `--citation-gate-cache`; missing or mismatched records fail without live fallback.
`--reuse-results-dir` reuses matching historical rule/Judge records only after both
gates pass. Formal experiments write `results-gated-v1/` separately from historical
results. The existing 600 outputs need no new generation, detection, or Judge calls.
