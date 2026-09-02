"""Small, side-effect-free checks for the selected pywebview backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import os
import platform


@dataclass(frozen=True, slots=True)
class BackendDiagnostic:
    platform: str
    backend: str
    ready: bool
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def diagnose_backend() -> BackendDiagnostic:
    """Return the backend pywebview should use on this desktop."""

    system = platform.system().lower()
    if importlib.util.find_spec("webview") is None:
        return BackendDiagnostic(
            platform=system or "unknown",
            backend="unavailable",
            ready=False,
            message="缺少 pywebview，请先运行 uv sync。",
        )
    if system == "linux":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        has_qt = any(
            importlib.util.find_spec(module) is not None
            for module in ("PyQt6", "PySide6", "PyQt5", "PySide2")
        )
        if not has_display:
            return BackendDiagnostic(
                platform="linux",
                backend="qt",
                ready=False,
                message="Linux 桌面应用需要 DISPLAY 或 WAYLAND_DISPLAY 图形会话。",
            )
        if not has_qt:
            return BackendDiagnostic(
                platform="linux",
                backend="qt",
                ready=False,
                message="缺少 Qt 后端，请使用包含 pywebview[qt] 的项目依赖运行 uv sync。",
            )
        return BackendDiagnostic(
            platform="linux",
            backend="qt",
            ready=True,
            message="Linux Qt backend 已就绪。",
        )
    if system == "windows":
        return BackendDiagnostic(
            platform="windows",
            backend="edgechromium",
            ready=True,
            message="将使用 Windows WebView2；若启动失败，请安装 Microsoft Edge WebView2 Runtime。",
        )
    if system == "darwin":
        return BackendDiagnostic(
            platform="macos",
            backend="cocoa",
            ready=True,
            message="macOS WKWebView backend 已就绪。",
        )
    return BackendDiagnostic(
        platform=system or "unknown",
        backend="default",
        ready=False,
        message=f"当前系统不在支持范围内：{platform.system() or 'unknown'}。",
    )
