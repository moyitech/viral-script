# Formal 100-topic experiment v1

This directory is the versioned, Git-indexed record for the submission experiment.
It uses 100 fixed public-topic prompts, one live research pass per topic, and three
script lengths (280/450/700), producing 300 frozen generation traces.

The prompts were constructed for this personal/activity project rather than copied
from a private benchmark. `topics.json` preserves the original dataset index and adds
deterministic domain and challenge tags for finance, public services, technology,
workplace, education, health, environment/energy, and consumer/social trade-offs.
The tags are experimental strata, not ground-truth answers. Dataset and Rubric hashes
are frozen in `experiment.json` before any result is observed.

Prepare deterministic inputs without network access:

```bash
uv run --no-sync python scripts/run_formal_experiment.py prepare
```

The experiment is already complete; its 300 outputs do not need to be rerun.
Current quality results are in `results-gated-v1/`: both reward-hacking and citation
gates run before rules/Judge scoring. Gate rejection leaves the score empty.
This directory reuses existing detector and scoring records; the original `results/`
and generation traces remain unchanged. The paired experiment follows the same protocol.

For a new experiment, `generate` and `score` may make live API calls; `report` only
reads stored records. Each command is resumable;
completed immutable artifacts are selected into top-level manifests and are never
overwritten:

```bash
uv run --no-sync python scripts/run_formal_experiment.py generate
uv run --no-sync python scripts/run_formal_experiment.py score
uv run --no-sync python scripts/run_formal_experiment.py report
```

`run` performs all four phases and therefore consumes both Tavily and Hy3 quota.
Defaults are 32 task pipelines, 64 global Hy3 requests, 8 Tavily requests, and 64
Judge requests in flight. Override them with the corresponding concurrency flags.

Raw research snapshots, generation traces, and item-level evaluator records are
tracked by Git LFS. Manifests, hashes, full CSV tables, summaries, and reports use
ordinary Git so reviewers can inspect them without downloading every LFS object.

After full scoring, 20 distinct gate-passing topics are selected across domain and length
strata. Each yields blinded good/medium/bad/adversarial cases; the answer key records
the exact edit recipe separately. The report includes strict triplet and pairwise
ordering. The two prerequisite gates and their attack results are documented in
the [task report](../../../docs/task1-evaluation-report.md).

After report export, give two reviewers separate copies of
`validation/human_review_template.csv`. Import the completed files with:

```bash
uv run --no-sync python scripts/import_human_annotations.py \
  --reviewer reviewer-a.csv --reviewer reviewer-b.csv \
  --arbitration arbitration.csv
```

The arbitration file is only required for runs whose two independent ratings differ.
Reviewers must use the 1–3 behavioral anchors in
`../../rubrics/script_quality_v1.json`; `reviewer_id`, all eight dimension scores,
`gate_failed`, and the unchanged blind batch/run/hash fields are mandatory. The
importer rejects duplicate rows, out-of-range scores, changed hashes, mismatched
batches, non-overlapping 50-item sets, and arbitration by either original reviewer.

The existing Judge repeats need no new API calls. Export component stability for
already-passed traces from stored results:

```bash
uv run --no-sync python scripts/report_judge_stability.py
```

The default output is `validation/stability/report-gated-v1/`, including the selected
trace manifest. Explicit manifests remain available for historical component diagnostics.
The repeat pass is compared only with the original Hy3 Judge records. Deterministic
rules and derived combined scores are not treated as independent repeat judgments.
The stability report exports per-dimension exact agreement, quadratic weighted Kappa,
normalized-score Spearman/MAE, and a full item-level disagreement table.
