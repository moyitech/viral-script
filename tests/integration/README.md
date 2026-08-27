# Integration tests

Test real Hy3 and Tavily connectivity here. Tests require the matching API Key,
must skip cleanly when required environment variables are absent, and must never
print secrets or full authorization headers.

Live tests require an explicit opt-in and consume Tavily credits:

```bash
HYSCRIPT_RUN_LIVE_TESTS=1 uv run --no-sync \
  python -m unittest \
  tests.integration.test_hy3_live \
  tests.integration.test_tavily_live -v
```

The central configuration module automatically reads the project-root `.env`.
The explicit process variable above overrides its default opt-out value.
