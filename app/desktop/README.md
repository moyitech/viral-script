# Desktop application

Launch from the project root with `uv run --no-sync python main.py` or
`uv run --no-sync python -m app.desktop`. Both use the same desktop entry point.
It loads static assets from
`assets/`, exposes the narrow JSON API in `controller.py`, and runs all HyScript
I/O on a dedicated asyncio thread so the native webview remains responsive.

Do not move provider calls, prompts, scoring logic, secrets, or unrestricted
filesystem access into the JavaScript bridge. Reusable orchestration belongs in
`hyscript.workflows`; the desktop layer only validates UI input, manages jobs,
and maps application results to JSON-safe payloads.

The explicit quality-report action evaluates an already-frozen trace through
reward-hacking and citation gates, then runs the eight-dimensional rubric only
if both pass. The UI shows each gate's verdict and reason; blocked outputs have
no score. Evaluation records and cache fingerprints remain separate from generation.
