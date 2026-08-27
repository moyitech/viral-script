# Evaluation results

Store offline rule, Judge, and reviewed human-evaluation outputs here. Every
result must reference an immutable generation artifact by `run_id` and record
the evaluator and Rubric versions. Never copy a score back into the generation
trace.

Generated results belong in `eval/results/runs/` and are ignored by Git. Commit
only small, redacted examples when they are required to explain the method.
