"""Offline tests for the pywebview JSON bridge and async job lifecycle."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
import threading
import time
import unittest

from app.desktop.controller import DesktopController
from app.desktop.diagnostics import BackendDiagnostic
from hyscript.agent import ScriptArtifact, ScriptTask, TopicRecommendation, TopicRecommendationBatch
from hyscript.artifacts import RunTrace
from hyscript.workflows import GeneratedScriptRun, QualityReport


READY = BackendDiagnostic(
    platform="test",
    backend="fake",
    ready=True,
    message="ready",
)


def _recommendations() -> TopicRecommendationBatch:
    return TopicRecommendationBatch(
        recommendations=(
            TopicRecommendation(
                title="测试推荐",
                angle="测试角度",
                why_now="正在发生",
                sources=(),
            ),
        ),
        prompt_version="topic-v1",
        input_hotspot_count=20,
        deduplicated_event_count=20,
        selected_event_count=20,
        generation_batch_count=4,
        embedding_request_count=1,
        llm_request_count=4,
        embedding_model="fake",
        similarity_threshold=0.72,
    )


def _generated() -> GeneratedScriptRun:
    task = ScriptTask(topic="测试选题", target_length=450)
    text = "这是桌面桥测试文案。"
    script = ScriptArtifact(
        outline=("开场",),
        script_text=text,
        claim_usages=(),
        character_count=len(text),
        prompt_version="script-v1",
        generation_attempt_count=1,
    )
    trace = RunTrace(
        run_id="run-desktop-test",
        created_at="2026-09-02T00:00:00Z",
        task={"topic": task.topic, "target_length": task.target_length},
        script_artifact={"script_text": text},
    )
    return GeneratedScriptRun(
        task=task,
        script=script,
        trace=trace,
        trace_path=Path("/tmp/run-desktop-test.json"),
    )


def _report() -> QualityReport:
    return QualityReport(
        evaluation_id="eval-test",
        run_id="run-desktop-test",
        summary="测试报告",
        score_percent=80.0,
        eligible=True,
        gate_failed=False,
        rubric_version="1.1.0",
        judge_model="fake",
        dimensions=(),
        findings=(),
        judge_groups=(),
        oral_subscores={},
        cached=False,
    )


class _Workflow:
    async def recommend_topics(self, *, count: int = 20):
        logging.getLogger("hyscript.desktop-test").info("推荐日志")
        return _recommendations()

    async def generate_script(self, task):
        return _generated()


class _BlockingWorkflow(_Workflow):
    async def recommend_topics(self, *, count: int = 20):
        await asyncio.sleep(3600)


class _ConcurrentWorkflow(_Workflow):
    def __init__(self) -> None:
        self.release_recommendations = threading.Event()

    async def recommend_topics(self, *, count: int = 20):
        logging.getLogger("hyscript.desktop-test").info("仅推荐日志")
        while not self.release_recommendations.is_set():
            await asyncio.sleep(0.01)
        return _recommendations()

    async def generate_script(self, task):
        logging.getLogger("hyscript.desktop-test").info("仅生成日志")
        return _generated()


class _BlockingGenerationWorkflow(_Workflow):
    async def generate_script(self, task):
        await asyncio.sleep(3600)


class _Evaluation:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    async def score_trace(self, trace_path: Path):
        self.paths.append(trace_path)
        return _report()


def _terminal(controller: DesktopController, job_id: str, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    after = 0
    events = []
    while time.monotonic() < deadline:
        result = controller.poll_job(job_id, after)
        events.extend(result.get("events", []))
        if events:
            after = events[-1]["seq"]
        if result.get("status") in {"succeeded", "failed", "cancelled"}:
            result["all_events"] = events
            return result
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def _wait_until_running(
    controller: DesktopController,
    job_id: str,
    timeout: float = 2.0,
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = controller.poll_job(job_id)
        if result.get("status") == "running":
            return result
        time.sleep(0.01)
    raise AssertionError("job did not start")


@dataclass
class _Screen:
    width: int
    height: int


class _Window:
    def __init__(self, width: int = 1920, height: int = 1080) -> None:
        self.screen = _Screen(width=width, height=height)
        self.sizes: list[tuple[int, int]] = []

    def resize(self, width: int, height: int) -> None:
        self.sizes.append((width, height))


class DesktopControllerTests(unittest.TestCase):
    def _controller(self, workflow=None, evaluation=None) -> DesktopController:
        controller = DesktopController(
            workflow or _Workflow(),
            evaluation or _Evaluation(),
            diagnostic=READY,
        )
        self.addCleanup(controller.shutdown)
        return controller

    def test_bootstrap_and_recommendation_events_are_incremental(self) -> None:
        controller = self._controller()

        length = controller.bootstrap()["length"]
        self.assertEqual(length["default"], 450)
        self.assertEqual(length["snap_points"], [280, 450, 700])
        started = controller.start_recommendations()
        result = _terminal(controller, started["job_id"])

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["result"]["recommendations"][0]["title"], "测试推荐")
        self.assertTrue(any(event["message"] == "推荐日志" for event in result["all_events"]))
        last = result["all_events"][-1]["seq"]
        self.assertEqual(controller.poll_job(started["job_id"], last)["events"], [])

    def test_recommendation_job_is_reused_after_success(self) -> None:
        controller = self._controller()

        first = controller.start_recommendations()
        result = _terminal(controller, first["job_id"])
        second = controller.start_recommendations()

        self.assertEqual(result["status"], "succeeded")
        self.assertTrue(second["ok"])
        self.assertTrue(second["reused"])
        self.assertEqual(second["job_id"], first["job_id"])
        self.assertEqual(second["status"], "succeeded")
        self.assertEqual(
            controller.poll_job(second["job_id"])["result"],
            result["result"],
        )

    def test_generation_validation_and_current_session_evaluation(self) -> None:
        evaluation = _Evaluation()
        controller = self._controller(evaluation=evaluation)

        self.assertFalse(controller.start_generation({"topic": "", "target_length": 450})["ok"])
        self.assertFalse(controller.start_generation({"topic": "测试", "target_length": 99})["ok"])
        generated = controller.start_generation(
            {"topic": "测试选题", "angle": "", "target_length": 450},
        )
        generation_result = _terminal(controller, generated["job_id"])
        self.assertEqual(generation_result["result"]["run_id"], "run-desktop-test")

        evaluated = controller.start_evaluation("run-desktop-test")
        evaluation_result = _terminal(controller, evaluated["job_id"])
        self.assertEqual(evaluation_result["result"]["score_percent"], 80.0)
        self.assertEqual(evaluation.paths, [Path("/tmp/run-desktop-test.json").resolve()])
        self.assertFalse(controller.start_evaluation("unknown-run")["ok"])

    def test_recommendations_and_generation_run_concurrently_without_log_leaks(self) -> None:
        workflow = _ConcurrentWorkflow()
        controller = self._controller(workflow=workflow)

        recommendation = controller.start_recommendations()
        recommendation_running = _wait_until_running(
            controller,
            recommendation["job_id"],
        )
        generated = controller.start_generation(
            {"topic": "测试", "angle": "", "target_length": 450},
        )
        generation_result = _terminal(controller, generated["job_id"])
        workflow.release_recommendations.set()
        recommendation_result = _terminal(controller, recommendation["job_id"])

        recommendation_messages = {
            event["message"]
            for event in recommendation_running["events"]
            + recommendation_result["all_events"]
        }
        generation_messages = {
            event["message"] for event in generation_result["all_events"]
        }
        self.assertTrue(generated["ok"])
        self.assertEqual(recommendation_result["status"], "succeeded")
        self.assertEqual(generation_result["status"], "succeeded")
        self.assertIn("仅推荐日志", recommendation_messages)
        self.assertNotIn("仅生成日志", recommendation_messages)
        self.assertIn("仅生成日志", generation_messages)
        self.assertNotIn("仅推荐日志", generation_messages)

    def test_foreground_jobs_are_mutually_exclusive_and_cancellable(self) -> None:
        controller = self._controller(workflow=_BlockingGenerationWorkflow())

        started = controller.start_generation(
            {"topic": "第一个", "angle": "", "target_length": 450},
        )
        _wait_until_running(controller, started["job_id"])
        blocked = controller.start_generation(
            {"topic": "第二个", "angle": "", "target_length": 450},
        )

        self.assertFalse(blocked["ok"])
        self.assertEqual(blocked["active_job_id"], started["job_id"])
        self.assertTrue(controller.cancel_job(started["job_id"])["ok"])
        self.assertEqual(_terminal(controller, started["job_id"])["status"], "cancelled")

    def test_recommendation_job_is_cancellable(self) -> None:
        controller = self._controller(workflow=_BlockingWorkflow())

        started = controller.start_recommendations()
        _wait_until_running(controller, started["job_id"])
        self.assertTrue(controller.cancel_job(started["job_id"])["ok"])
        self.assertEqual(_terminal(controller, started["job_id"])["status"], "cancelled")

    def test_set_stage_uses_allowlisted_screen_aware_sizes(self) -> None:
        controller = self._controller()

        self.assertFalse(controller.set_stage("compose")["ok"])
        window = _Window()
        controller.attach_window(window)
        self.assertEqual(
            controller.set_stage("compose"),
            {
                "ok": True,
                "stage": "compose",
                "width": 720,
                "height": 660,
            },
        )
        self.assertEqual(window.sizes[-1], (720, 660))
        self.assertEqual(
            controller.set_stage("recommendation_picker"),
            {
                "ok": True,
                "stage": "recommendation_picker",
                "width": 940,
                "height": 720,
            },
        )
        self.assertEqual(window.sizes[-1], (940, 720))
        self.assertFalse(controller.set_stage("unknown")["ok"])

        compact_window = _Window(width=800, height=650)
        controller.attach_window(compact_window)
        report = controller.set_stage("report")
        self.assertEqual((report["width"], report["height"]), (760, 590))
        self.assertEqual(compact_window.sizes[-1], (760, 590))

    def test_configuration_error_disables_work(self) -> None:
        controller = DesktopController(
            None,
            None,
            diagnostic=READY,
            configuration_error="Missing configuration: HY3_API_KEY",
        )
        self.addCleanup(controller.shutdown)

        self.assertFalse(controller.bootstrap()["ready"])
        self.assertFalse(controller.start_recommendations()["ok"])
