# Hy3 vs GLM-5.3-Flash Judge comparison v1

This experiment reuses the 300 frozen traces from each of `formal-100-v1` and
`formal-100-e2e-single-shot-v1`. It does not regenerate scripts, run Tavily, or
score the 80 synthetic discrimination/adversarial cases.

Hy3's existing `high` results are the reference. GLM-5.3-Flash uses API model id
and display name `glm-5.3-flash`, with reasoning effort `max`. These are the
highest supported reasoning settings selected for the respective models; the
effort strings are intentionally different. GLM's `max` value is sent as the
top-level `reasoning_effort`; Hy3's `high` remains in `chat_template_kwargs`.

Prepare the immutable source lock without calling an external service:

```bash
uv run --no-sync python scripts/run_glm_judge_model_comparison.py prepare
```

The remaining scoring commands make explicit live API calls. Both use a fixed
global Judge request concurrency of 512 and are resumable without overwriting
completed records:

```bash
uv run --no-sync python scripts/run_glm_judge_model_comparison.py score
uv run --no-sync python scripts/run_glm_judge_model_comparison.py repeat
uv run --no-sync python scripts/run_glm_judge_model_comparison.py report
```

`score` runs deterministic rules plus the canonical GLM Judge pass. `repeat`
runs only a second GLM Judge pass for stability analysis. `run` executes all
four phases. Across both generation workflows the two GLM passes produce 1,200
item-level Judge records and at least 4,800 model requests before any format
repair.

The report compares the two generation workflows under each canonical Judge,
measures Hy3–GLM agreement on identical traces, and reports two-pass internal
stability for both models. The 1,200 `hy3_judge.json` records are tracked with
Git LFS; configuration, summaries, Markdown, CSV, rule, and combined records use
ordinary Git.
