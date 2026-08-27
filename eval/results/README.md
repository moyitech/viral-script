# Evaluation results

Store offline rule, Judge, and reviewed human-evaluation outputs here. Every
result must reference an immutable generation artifact by `run_id` and record
the evaluator and Rubric versions. Never copy a score back into the generation
trace.

Generated results belong in `eval/results/runs/` and are ignored by Git. Commit
only small, redacted examples when they are required to explain the method.

Each scoring invocation writes:

```text
<output-dir>/
  manifest.json
  summary.json
  failures.json
  items/<run_id>/rules.json
  items/<run_id>/hy3_judge.json   # only when Judge is selected
  items/<run_id>/combined.json
```

`rules.json` contains deterministic metrics and gates, not the eight quality
scores. `hy3_judge.json` contains the eight 0-4 scores. `combined.json` keeps
those diagnostic scores but sets `final_score` to null whenever a
non-compensable gate fires or a complete Judge score is unavailable.

Resume requires the same input trace set and the same full evaluation
fingerprint: Rubric, rule thresholds, Judge model and prompt, request/context
limits, sampling parameters, and aggregator version. Use a new output
directory or explicit `--overwrite` when any of them changes.
