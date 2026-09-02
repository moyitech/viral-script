# Desktop application

The desktop entry point is `python -m app.desktop`. It loads static assets from
`assets/`, exposes the narrow JSON API in `controller.py`, and runs all HyScript
I/O on a dedicated asyncio thread so the native webview remains responsive.

Do not move provider calls, prompts, scoring logic, secrets, or unrestricted
filesystem access into the JavaScript bridge. Reusable orchestration belongs in
`hyscript.workflows`; the desktop layer only validates UI input, manages jobs,
and maps application results to JSON-safe payloads.
