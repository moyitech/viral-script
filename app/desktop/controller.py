"""Thread-safe JSON bridge between pywebview and async HyScript workflows."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import threading
from typing import Any, Awaitable, Callable, Protocol
from uuid import uuid4

from hyscript.agent import ScriptTask, TopicGenerationError
from hyscript.config import SettingsError
from hyscript.llm import EmbeddingProviderError, LLMProviderError
from hyscript.search import SearchProviderError
from hyscript.trends import HotlistProviderError
from hyscript.workflows import (
    CreatorGenerationError,
    GeneratedScriptRun,
    QualityReport,
    QualityReportError,
)

from .diagnostics import BackendDiagnostic


class CreatorWorkflowProtocol(Protocol):
    async def recommend_topics(self, *, count: int = 20) -> Any: ...

    async def generate_script(self, task: ScriptTask) -> GeneratedScriptRun: ...


class EvaluationWorkflowProtocol(Protocol):
    async def score_trace(self, trace_path: Path) -> QualityReport: ...


JobSerializer = Callable[[Any], dict[str, Any]]

_CURRENT_JOB_ID: ContextVar[str | None] = ContextVar(
    "hyscript_desktop_job_id",
    default=None,
)

_STAGE_SIZES: dict[str, tuple[int, int]] = {
    "compose": (720, 660),
    "recommendation_progress": (760, 560),
    "recommendation_picker": (940, 720),
    "generation_progress": (760, 560),
    "result": (1000, 760),
    "evaluation_progress": (760, 560),
    "report": (1080, 820),
}


@dataclass(slots=True)
class _Job:
    job_id: str
    kind: str
    status: str = "queued"
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    future: Future[Any] | None = None


class _JobLogHandler(logging.Handler):
    def __init__(self, emit_log: Callable[[str, str, str], None]) -> None:
        super().__init__(level=logging.INFO)
        self._emit_log = emit_log

    def emit(self, record: logging.LogRecord) -> None:
        try:
            job_id = _CURRENT_JOB_ID.get()
            if job_id is not None:
                self._emit_log(job_id, record.levelname.lower(), record.getMessage())
        except Exception:
            self.handleError(record)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _safe_error(exc: BaseException) -> str:
    known = (
        CreatorGenerationError,
        QualityReportError,
        TopicGenerationError,
        SettingsError,
        LLMProviderError,
        EmbeddingProviderError,
        SearchProviderError,
        HotlistProviderError,
        ValueError,
    )
    if isinstance(exc, known) and str(exc).strip():
        return str(exc).strip()
    return "任务执行失败，请检查配置或网络后重试。"


def _recommendations_payload(batch: Any) -> dict[str, Any]:
    return {
        "recommendations": [
            {
                "id": f"rec-{index:02d}",
                "title": item.title,
                "angle": item.angle,
                "why_now": item.why_now,
                "sources": [asdict(source) for source in item.sources],
            }
            for index, item in enumerate(batch.recommendations, start=1)
        ],
        "prompt_version": batch.prompt_version,
    }


def _generation_payload(run: GeneratedScriptRun) -> dict[str, Any]:
    return {
        "run_id": run.trace.run_id,
        "topic": run.task.topic,
        "target_length": run.task.target_length,
        "character_count": run.script.character_count,
        "script_text": run.script.script_text,
    }


class DesktopController:
    """A pywebview API with one background and one foreground job slot."""

    def __init__(
        self,
        workflow: CreatorWorkflowProtocol | None,
        evaluation_workflow: EvaluationWorkflowProtocol | None,
        *,
        diagnostic: BackendDiagnostic,
        configuration_error: str | None = None,
        log_level: str = "INFO",
    ) -> None:
        self._workflow = workflow
        self._evaluation_workflow = evaluation_workflow
        self._diagnostic = diagnostic
        self._configuration_error = configuration_error
        self._lock = threading.RLock()
        self._jobs: dict[str, _Job] = {}
        self._recommendation_job_id: str | None = None
        self._foreground_job_id: str | None = None
        self._trace_paths: dict[str, Path] = {}
        self._window: Any | None = None
        self._closed = False

        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="hyscript-desktop-asyncio",
            daemon=True,
        )
        self._thread.start()
        self._loop_ready.wait(timeout=5)

        self._package_logger = logging.getLogger("hyscript")
        self._previous_log_level = self._package_logger.level
        self._package_logger.setLevel(getattr(logging, log_level, logging.INFO))
        self._log_handler = _JobLogHandler(self._event)
        self._package_logger.addHandler(self._log_handler)

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._loop_ready.set()
        loop.run_forever()
        loop.close()

    def bootstrap(self) -> dict[str, Any]:
        """Return non-secret runtime capabilities and fixed UI constraints."""

        error = self._configuration_error
        return {
            "ready": error is None and self._diagnostic.ready,
            "configuration_error": error,
            "diagnostic": self._diagnostic.to_dict(),
            "length": {
                "min": 100,
                "max": 1000,
                "default": 450,
                "snap_points": [280, 450, 700],
                "snap_enter_pixels": 14,
                "snap_release_pixels": 24,
            },
        }

    def attach_window(self, window: Any) -> None:
        """Bind the native window after pywebview creates it."""

        with self._lock:
            self._window = window

    def set_stage(self, stage: Any) -> dict[str, Any]:
        """Resize only to an allowlisted application stage."""

        if not isinstance(stage, str) or stage not in _STAGE_SIZES:
            return {"ok": False, "error": "未知的界面阶段。"}
        with self._lock:
            window = self._window
        if window is None:
            return {"ok": False, "error": "原生窗口尚未就绪。"}
        width, height = _STAGE_SIZES[stage]
        screen = getattr(window, "screen", None)
        screen_width = getattr(screen, "width", None)
        screen_height = getattr(screen, "height", None)
        if isinstance(screen_width, int) and screen_width > 0:
            width = max(620, min(width, screen_width - 40))
        if isinstance(screen_height, int) and screen_height > 0:
            height = max(500, min(height, screen_height - 60))
        try:
            window.resize(width, height)
        except Exception:
            return {"ok": False, "error": "窗口尺寸调整失败。"}
        return {"ok": True, "stage": stage, "width": width, "height": height}

    def _event(self, job_id: str, level: str, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.events.append(
                {
                    "seq": len(job.events) + 1,
                    "timestamp": _now(),
                    "level": level,
                    "message": message,
                }
            )

    async def _execute(
        self,
        job_id: str,
        operation: Callable[[], Awaitable[Any]],
        serializer: JobSerializer,
    ) -> None:
        token = _CURRENT_JOB_ID.set(job_id)
        try:
            with self._lock:
                job = self._jobs[job_id]
                job.status = "running"
            self._event(job_id, "info", "任务已开始")
            try:
                value = await operation()
                payload = serializer(value)
                with self._lock:
                    job = self._jobs[job_id]
                    job.result = payload
                    job.status = "succeeded"
                    if job.kind == "generation" and isinstance(value, GeneratedScriptRun):
                        self._trace_paths[value.trace.run_id] = value.trace_path.resolve()
                self._event(job_id, "success", "任务已完成")
            except asyncio.CancelledError:
                with self._lock:
                    self._jobs[job_id].status = "cancelled"
                self._event(job_id, "warning", "任务已取消")
                raise
            except BaseException as exc:
                message = _safe_error(exc)
                with self._lock:
                    job = self._jobs[job_id]
                    job.status = "failed"
                    job.error = message
                self._event(job_id, "error", message)
        finally:
            _CURRENT_JOB_ID.reset(token)
            with self._lock:
                if self._foreground_job_id == job_id:
                    self._foreground_job_id = None

    def _submit(
        self,
        kind: str,
        operation: Callable[[], Awaitable[Any]],
        serializer: JobSerializer,
        *,
        background: bool,
    ) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                return {"ok": False, "error": "应用正在关闭。"}
            if self._configuration_error is not None:
                return {"ok": False, "error": self._configuration_error}
            if not self._diagnostic.ready:
                return {"ok": False, "error": self._diagnostic.message}
            occupied_job_id = (
                self._recommendation_job_id if background else self._foreground_job_id
            )
            if occupied_job_id is not None:
                return {
                    "ok": False,
                    "error": "已有任务正在运行，请等待完成或先取消。",
                    "active_job_id": occupied_job_id,
                }
            if self._loop is None or not self._loop.is_running():
                return {"ok": False, "error": "后台任务循环尚未就绪。"}
            job_id = f"job-{uuid4().hex}"
            job = _Job(job_id=job_id, kind=kind)
            self._jobs[job_id] = job
            if background:
                self._recommendation_job_id = job_id
            else:
                self._foreground_job_id = job_id
            job.future = asyncio.run_coroutine_threadsafe(
                self._execute(job_id, operation, serializer),
                self._loop,
            )
            job.future.add_done_callback(
                lambda future, submitted_job_id=job_id: self._future_done(
                    submitted_job_id,
                    future,
                )
            )
            return {"ok": True, "job_id": job_id}

    def _future_done(self, job_id: str, future: Future[Any]) -> None:
        """Finalize cancellation even if it happened before the coroutine started."""

        if not future.cancelled():
            return
        should_emit = False
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and job.status != "cancelled":
                job.status = "cancelled"
                should_emit = True
            if self._foreground_job_id == job_id:
                self._foreground_job_id = None
        if should_emit:
            self._event(job_id, "warning", "任务已取消")

    def start_recommendations(self) -> dict[str, Any]:
        if self._workflow is None:
            return {"ok": False, "error": "生成工作流不可用。"}
        with self._lock:
            existing_id = self._recommendation_job_id
            existing = self._jobs.get(existing_id or "")
            if existing is not None and existing.status in {
                "queued",
                "running",
                "cancelling",
                "succeeded",
            }:
                return {
                    "ok": True,
                    "job_id": existing.job_id,
                    "reused": True,
                    "status": existing.status,
                }
            self._recommendation_job_id = None
        return self._submit(
            "recommendations",
            lambda: self._workflow.recommend_topics(count=20),
            _recommendations_payload,
            background=True,
        )

    def start_generation(self, payload: Any) -> dict[str, Any]:
        if self._workflow is None:
            return {"ok": False, "error": "生成工作流不可用。"}
        if not isinstance(payload, dict):
            return {"ok": False, "error": "生成参数格式无效。"}
        topic = payload.get("topic")
        angle = payload.get("angle", "")
        target_length = payload.get("target_length")
        if not isinstance(topic, str) or not topic.strip():
            return {"ok": False, "error": "请输入选题。"}
        if not isinstance(angle, str):
            return {"ok": False, "error": "创作角度格式无效。"}
        if (
            isinstance(target_length, bool)
            or not isinstance(target_length, int)
            or not 100 <= target_length <= 1000
        ):
            return {"ok": False, "error": "字数必须是 100 至 1000 之间的整数。"}
        try:
            task = ScriptTask(
                topic=topic,
                angle=angle,
                target_length=target_length,
            )
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return self._submit(
            "generation",
            lambda: self._workflow.generate_script(task),
            _generation_payload,
            background=False,
        )

    def start_evaluation(self, run_id: Any) -> dict[str, Any]:
        if self._evaluation_workflow is None:
            return {"ok": False, "error": "评分工作流不可用。"}
        if not isinstance(run_id, str) or not run_id.strip():
            return {"ok": False, "error": "run_id 无效。"}
        with self._lock:
            trace_path = self._trace_paths.get(run_id)
        if trace_path is None:
            return {"ok": False, "error": "当前会话中找不到对应的生成 trace。"}
        return self._submit(
            "evaluation",
            lambda: self._evaluation_workflow.score_trace(trace_path),
            lambda report: report.to_dict(),
            background=False,
        )

    def poll_job(self, job_id: Any, after_seq: Any = 0) -> dict[str, Any]:
        if not isinstance(job_id, str):
            return {"ok": False, "error": "job_id 无效。"}
        if isinstance(after_seq, bool) or not isinstance(after_seq, int) or after_seq < 0:
            return {"ok": False, "error": "after_seq 无效。"}
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return {"ok": False, "error": "任务不存在。"}
            return {
                "ok": True,
                "job_id": job.job_id,
                "kind": job.kind,
                "status": job.status,
                "events": [
                    dict(event) for event in job.events if event["seq"] > after_seq
                ],
                "result": job.result if job.status == "succeeded" else None,
                "error": job.error,
            }

    def cancel_job(self, job_id: Any) -> dict[str, Any]:
        if not isinstance(job_id, str):
            return {"ok": False, "error": "job_id 无效。"}
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return {"ok": False, "error": "任务不存在。"}
            if job.status not in {"queued", "running"} or job.future is None:
                return {"ok": False, "error": "任务已经结束。"}
            job.status = "cancelling"
            job.future.cancel()
        self._event(job_id, "warning", "正在取消任务；已发出的远端请求可能仍产生费用")
        return {"ok": True, "status": "cancelling"}

    def shutdown(self) -> None:
        """Cancel active work, detach logging, and stop the background loop."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._package_logger.removeHandler(self._log_handler)
        self._package_logger.setLevel(self._previous_log_level)
        if self._loop is not None and self._loop.is_running():
            async def drain() -> None:
                current = asyncio.current_task()
                tasks = [
                    task
                    for task in asyncio.all_tasks()
                    if task is not current and not task.done()
                ]
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            try:
                asyncio.run_coroutine_threadsafe(drain(), self._loop).result(timeout=2)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2)
