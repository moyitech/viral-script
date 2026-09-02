# Evaluation runners

Reusable offline experiment orchestration belongs here. A Runner has two
separable phases:

1. read the existing-topic dataset, invoke the shared generation workflow in
   batches, and freeze all intermediate artifacts;
2. read those frozen artifacts, run rule/Judge/human evaluation, and write
   results linked by `run_id` without modifying generation files.

Command-line wrappers belong in `scripts/`. The Web and API applications must
not import or invoke these runners.

The scoring phase is implemented. Run deterministic rules without loading
`.env` or calling an external service:

```bash
uv run --no-sync python scripts/run_evaluation.py score \
  --trace eval/traces/example_trace.json \
  --evaluators rules \
  --output-dir /tmp/hyscript-eval
```

Add the Hy3 Judge explicitly when API-backed evaluation is intended:

```bash
uv run --no-sync python scripts/run_evaluation.py score \
  --trace-dir eval/traces/runs/<batch-id> \
  --evaluators rules,judge \
  --output-dir eval/results/runs/<evaluation-id> \
  --concurrency 2
```

Judge requests consume API quota. The shared existing-topic generation workflow
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
