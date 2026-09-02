# Evaluation results

Store offline rule, Judge, and reviewed human-evaluation outputs here. Every
result must reference an immutable generation artifact by `run_id` and record
the evaluator and Rubric versions. Never copy a score back into the generation
trace.

Generated results belong in `eval/results/runs/` and are ignored by Git. Commit
only small, redacted examples when they are required to explain the method.
The versioned formal submission experiment instead stores complete item records under
`eval/experiments/formal-100-v1/`, with item JSON in Git LFS and summaries/tables in
ordinary Git.

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

The default v1 Rubric sends the original seven 1-3 dimensions to Judge. Length
is scored independently by rules, added to the seven Judge scores, and then
normalized by the combined maximum.
`selected_evidence` is interpreted as citation/background metadata; citations
do not need to appear in the spoken body and no claim-level or grounding-review
gate is added. References only help Judge assess fabrication and argument quality
inside `theme_information`; they are not an eighth score. `combined.json` sets
`final_score` to null when the complete seven-dimension-plus-length score is unavailable.

The v2 Rubric and its deterministic length/evidence gates remain available only
for reproducing the historical evidence-chain experiments.

Resume requires the same input trace set and the same full evaluation
fingerprint: Rubric, rule thresholds, Judge model and prompt, request/context
limits, sampling parameters, and aggregator version. Use a new output
directory or explicit `--overwrite` when any of them changes.

In `summary.json`, `counts_scope` is `current_invocation`: `completed`,
`skipped`, and `failed` describe only what that command invocation did. On a
resume, an already valid result is `skipped`, so `completed` is not cumulative.
Use `record_coverage.combined_record_count` and `record_coverage.complete` to
interpret the final stored result coverage, and use `aggregate.record_count`
for the number of combined records included in aggregate metrics.
