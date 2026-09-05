"""Offline contract tests for the full, gated scoring pipeline."""

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock

from hyscript.config import PROJECT_ROOT
from hyscript.evaluation import BatchEvaluationConfig, BatchEvaluationRunner, load_rubric
from hyscript.evaluation.gates import (
    AttackGateEvaluator, RecordedDetector, GateConflictError, GateEvaluationError,
    recorded_gates, passed_trace_manifest,
)
from hyscript.evaluation.io import load_frozen_trace


class FakeDetector:
    def __init__(self, field, *, blocked=False, error=False):
        self.field, self.blocked, self.error = field, blocked, error
        self.calls = 0
        self.fingerprint = {"name": field, "version": "test", "sha256": "a" * 64}

    async def evaluate(self, trace, **kwargs):
        self.calls += 1
        if self.error:
            raise RuntimeError("provider failed")
        return {
            "run_id": trace.run_id, "trace_sha256": trace.trace_sha256,
            "detector": self.fingerprint, self.field: self.blocked,
            "reason": "检测到异常" if self.blocked else "通过检测",
        }


def fake_gates(*, reward=False, citation=False):
    return AttackGateEvaluator(
        FakeDetector("reward_hacking", blocked=reward),
        FakeDetector("fabricated_citation", blocked=citation),
    )


class EvaluationGateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.trace_path = self.root / "trace.json"
        shutil.copyfile(PROJECT_ROOT / "eval/traces/example_trace.json", self.trace_path)
        self.trace = load_frozen_trace(self.trace_path)
        self.original = self.trace_path.read_bytes()
        self.rubric = load_rubric(PROJECT_ROOT / "eval/rubrics/script_quality_v1.json")

    def runner(self, gates, *, output="results", reuse=None):
        # Reuse the real Judge adapter with a mocked async evaluation boundary.
        from test_creator_quality_workflow import _FakeJudge
        judge = _FakeJudge()
        runner = BatchEvaluationRunner(
            self.rubric,
            BatchEvaluationConfig(
                output_dir=self.root / output, evaluators=("rules", "judge"),
                reuse_results_dir=reuse,
            ),
            gate_evaluator=gates, judge_evaluator=judge,
        )
        return runner, judge

    def item(self, output="results"):
        return self.root / output / "items" / self.trace.run_id

    async def test_full_scoring_requires_both_gates(self):
        from test_creator_quality_workflow import _FakeJudge
        with self.assertRaisesRegex(ValueError, "Both attack gates"):
            BatchEvaluationRunner(
                self.rubric,
                BatchEvaluationConfig(output_dir=self.root, evaluators=("rules", "judge")),
                judge_evaluator=_FakeJudge(),
            )

    async def test_each_gate_blocks_rules_and_judge_and_keeps_no_final_score(self):
        for name in ("reward", "citation"):
            with self.subTest(name=name):
                gates = fake_gates(**{name: True})
                runner, judge = self.runner(gates, output=name)
                runner.rule_evaluator.evaluate = lambda *_: self.fail("rules ran before gate")
                result = await runner.run([self.trace_path])
                self.assertEqual(result.failed_count, 0)
                self.assertEqual(judge.calls, 0)
                combined = json.loads((self.item(name) / "combined.json").read_text())
                self.assertTrue(combined["gate_failed"])
                self.assertIsNone(combined["metrics"]["final_score"])
                self.assertFalse(combined["metrics"]["eligible"])
                self.assertEqual(combined["dimension_scores"], [])
                self.assertFalse((self.item(name) / "hy3_judge.json").exists())
                self.assertFalse((self.item(name) / "rules.json").exists())
                self.assertEqual(self.trace_path.read_bytes(), self.original)

    async def test_passes_score_and_resume_without_detector_or_judge_calls(self):
        gates = fake_gates()
        runner, judge = self.runner(gates)
        first = await runner.run([self.trace_path])
        second = await runner.run([self.trace_path])
        self.assertEqual(first.failed_count, 0)
        self.assertEqual(second.outcomes[0].status, "skipped")
        self.assertEqual(judge.calls, 1)
        self.assertEqual([d.calls for d in gates.detectors.values()], [1, 1])
        combined = json.loads((self.item() / "combined.json").read_text())
        self.assertEqual(len(combined["dimension_scores"]), 8)
        self.assertEqual(set(combined["metadata"]["attack_gates"]), {"reward_hacking", "citation"})

    async def test_unavailable_check_is_not_passed_and_completed_check_is_reused(self):
        gates = fake_gates()
        gates.detectors["citation"].error = True
        runner, judge = self.runner(gates)
        failed = await runner.run([self.trace_path])
        self.assertEqual(failed.outcomes[0].error_code, "gate_check_failed")
        self.assertEqual(judge.calls, 0)
        self.assertFalse((self.item() / "combined.json").exists())
        gates.detectors["citation"].error = False
        resumed = await runner.run([self.trace_path])
        self.assertEqual(resumed.failed_count, 0)
        self.assertEqual([d.calls for d in gates.detectors.values()], [1, 2])
        self.assertEqual(judge.calls, 1)

    async def test_changed_detector_fingerprint_cannot_resume_old_pass(self):
        gates = fake_gates()
        runner, _ = self.runner(gates)
        await runner.run([self.trace_path])
        gates.detectors["citation"].fingerprint = {"version": "new", "sha256": "b" * 64}
        changed, judge = self.runner(gates)
        result = await changed.run([self.trace_path])
        self.assertEqual(result.outcomes[0].error_code, "resume_conflict")
        self.assertEqual(judge.calls, 0)

    async def test_gate_result_for_another_trace_is_rejected(self):
        gates = fake_gates()
        detector = gates.detectors["reward_hacking"]
        payload = await detector.evaluate(self.trace)
        payload["trace_sha256"] = "f" * 64
        detector.evaluate = AsyncMock(return_value=payload)
        runner, judge = self.runner(gates)
        result = await runner.run([self.trace_path])
        self.assertEqual(result.outcomes[0].error_code, "resume_conflict")
        self.assertEqual(judge.calls, 0)

    async def test_new_protocol_reuses_scores_only_after_gates_pass(self):
        old, _ = self.runner(fake_gates(), output="old")
        await old.run([self.trace_path])
        source = self.item("old") / "hy3_judge.json"
        original = source.read_bytes()
        new, judge = self.runner(fake_gates(), output="new", reuse=self.root / "old")
        judge.evaluate = AsyncMock(side_effect=AssertionError("must reuse"))
        self.assertEqual((await new.run([self.trace_path])).failed_count, 0)
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual((self.item("new") / "hy3_judge.json").read_bytes(), original)
        blocked, judge = self.runner(fake_gates(citation=True), output="blocked", reuse=self.root / "old")
        await blocked.run([self.trace_path])
        self.assertFalse((self.item("blocked") / "hy3_judge.json").exists())

    async def test_gate_provider_calls_are_bounded(self):
        import asyncio
        active = maximum = 0
        gates = fake_gates()
        original = gates.evaluate
        async def delayed(*args, **kwargs):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0)
            try:
                return await original(*args, **kwargs)
            finally:
                active -= 1
        gates.evaluate = delayed
        paths = []
        for index in range(5):
            payload = json.loads(self.original)
            payload["run_id"] = f"concurrency-{index}"
            path = self.root / f"{index}.json"
            path.write_text(json.dumps(payload))
            paths.append(path)
        runner, _ = self.runner(gates)
        self.assertEqual((await runner.run(paths)).failed_count, 0)
        self.assertLessEqual(maximum, 2)

    async def detector_cache(self, name, detector):
        root = self.root / name
        root.mkdir()
        payload = await detector.evaluate(self.trace)
        (root / "check.json").write_text(json.dumps(payload))
        (root / "summary.json").write_text(json.dumps({"detector": detector.fingerprint}))
        (root / "manifest.json").write_text(json.dumps({
            "detector": detector.fingerprint,
            "items": [{"run_id": self.trace.run_id, "result": "check.json"}],
        }))
        return root

    async def test_recorded_checks_replay_without_live_detectors(self):
        fake = fake_gates()
        reward = await self.detector_cache("reward", fake.detectors["reward_hacking"])
        citation = await self.detector_cache("citation", fake.detectors["citation"])
        runner, _ = self.runner(recorded_gates(reward, citation))
        self.assertEqual((await runner.run([self.trace_path])).failed_count, 0)
        self.assertEqual([d.calls for d in fake.detectors.values()], [1, 1])
        # An incomplete explicit cache must fail; no live detector is available.
        (reward / "check.json").unlink()
        missing, judge = self.runner(recorded_gates(reward, citation), output="missing")
        self.assertEqual((await missing.run([self.trace_path])).outcomes[0].error_code, "gate_check_failed")
        self.assertEqual(judge.calls, 0)

    async def test_recorded_check_fingerprint_mismatch_is_rejected(self):
        root = await self.detector_cache("cache", FakeDetector("reward_hacking"))
        payload = json.loads((root / "check.json").read_text())
        payload["detector"]["version"] = "changed"
        (root / "check.json").write_text(json.dumps(payload))
        with self.assertRaises(GateConflictError):
            await RecordedDetector(root, "reward_hacking").evaluate(self.trace)

    async def test_repeat_selection_requires_passed_completed_gates(self):
        runner, _ = self.runner(fake_gates())
        await runner.run([self.trace_path])
        manifest = self.root / "traces.json"
        manifest.write_text(json.dumps({"tasks": [{
            "task_id": "test", "trace": "trace.json", "trace_sha256": self.trace.trace_sha256,
        }]}))
        selected = self.root / "repeat/traces.json"
        passed_trace_manifest(manifest, self.root / "results", selected)
        self.assertEqual(len(json.loads(selected.read_text())["tasks"]), 1)
        path = self.item() / "combined.json"
        payload = json.loads(path.read_text())
        payload["metadata"]["attack_gates"]["citation"]["passed"] = False
        path.write_text(json.dumps(payload))
        with self.assertRaises(GateEvaluationError):
            passed_trace_manifest(manifest, self.root / "results", selected)
        payload["gate_failed"] = True
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(GateEvaluationError, "No gate-passing"):
            passed_trace_manifest(manifest, self.root / "results", selected)
