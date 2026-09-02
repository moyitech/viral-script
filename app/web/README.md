# Web application

Place the creator-facing Web UI here. Keep prompt templates and search/generation
logic in `src/hyscript/` rather than duplicating them in the UI. The UI ends at
the generated script, evidence mapping, and application run trace. A future Web
UI may expose the same explicit, post-generation quality report as the desktop
application, but it must score only an already-frozen trace and keep evaluation
outputs separate from generation artifacts.
