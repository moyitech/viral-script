# Project instructions

## External API testing

- Unit tests must not access external services. Use mocks or fakes for LLM and search clients.
- Real LLM API calls are allowed when they materially help debugging or end-to-end validation. Keep calls targeted and avoid unnecessary retries or large token budgets.
- Search API calls are expensive. Mock them by default and make a real search request only when it is necessary to validate behavior that cannot be checked locally.
- When a real search request is necessary, minimize query count, result count, search depth, and repeated runs.
- Keep live API checks separate from the default test suite and make them explicitly opt-in.
- Load credentials only through `hyscript.config.settings`. Never print, log, hard-code, or commit API keys.

## Architecture boundaries

- Keep reusable implementation in `src/hyscript/`. Examples and application entrypoints should call that implementation instead of duplicating provider logic.
- Use native async interfaces for LLM and search I/O. Do not add synchronous network calls to Agent, API, or evaluation workflows.
- Keep offline evaluation separate from the online application. Application code must not invoke scoring as part of the creator-facing generation flow.
- Freeze generation traces before scoring. Evaluation results must be stored separately and linked by `run_id` and trace hash; evaluators must not rewrite generation artifacts.
- In formal end-to-end evaluation, the Agent must generate search queries and perform live retrieval. Static search fixtures are only for unit tests and component diagnostics.
- Topic recommendations should come from current public hot lists. Do not introduce creator profiles or persistent user profiling unless the project scope is explicitly changed.

## Development workflow

- Run the default unit suite with `uv run --no-sync python -m unittest discover -s tests/unit -p 'test_*.py' -v`.
- Preserve unrelated working-tree changes. Keep commits scoped to the requested task.
