"""Launch the cross-platform HyScript desktop GUI."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

from hyscript.config import SettingsError, get_settings
from hyscript.workflows import CreatorEvaluationWorkflow, CreatorWorkflow

from .controller import DesktopController
from .diagnostics import diagnose_backend


def _controller() -> DesktopController:
    diagnostic = diagnose_backend()
    try:
        settings = get_settings()
    except SettingsError as exc:
        return DesktopController(
            None,
            None,
            diagnostic=diagnostic,
            configuration_error=str(exc),
        )
    return DesktopController(
        CreatorWorkflow(settings),
        CreatorEvaluationWorkflow(settings),
        diagnostic=diagnostic,
        log_level=settings.runtime.log_level,
    )


def main() -> int:
    diagnostic = diagnose_backend()
    if not diagnostic.ready:
        print(diagnostic.message, file=sys.stderr)
        return 1

    import webview

    controller = _controller()
    asset_path = Path(__file__).resolve().parent / "assets/index.html"
    window = webview.create_window(
        "HyScript · 口播文案工作台",
        url=asset_path.as_uri(),
        js_api=controller,
        width=720,
        height=660,
        min_size=(620, 500),
    )
    controller.attach_window(window)
    window.events.closed += controller.shutdown
    try:
        kwargs = {"gui": "qt"} if diagnostic.platform == "linux" else {}
        webview.start(debug=False, **kwargs)
    finally:
        controller.shutdown()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
