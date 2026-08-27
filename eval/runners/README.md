# Evaluation runners

Reusable offline experiment orchestration belongs here. A Runner has two
separable phases:

1. read the existing-topic dataset, invoke the shared generation workflow in
   batches, and freeze all intermediate artifacts;
2. read those frozen artifacts, run rule/Judge/human evaluation, and write
   results linked by `run_id` without modifying generation files.

Command-line wrappers belong in `scripts/`. The Web and API applications must
not import or invoke these runners.
