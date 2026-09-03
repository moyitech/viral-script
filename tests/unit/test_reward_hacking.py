"""Offline tests for the attack-only reward-hacking detector."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from hyscript.config import PROJECT_ROOT
from hyscript.evaluation.io import load_frozen_trace
from hyscript.evaluation.reward_hacking import (
    RewardHackingConfig,
    RewardHackingConflictError,
    RewardHackingDetector,
    run_reward_hacking_validation,
)
from hyscript.llm import ChatResponse


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = []

    async def complete(self, messages, *, reasoning_effort="no_think") -> ChatResponse:
        self.calls.append((messages, reasoning_effort))
        return ChatResponse(
            content=self.responses.pop(0),
            model="fake-hy3",
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        )


def _trace(path: Path, *, run_id: str, text: str) -> None:
    payload = json.loads(
        (PROJECT_ROOT / "eval/traces/example_trace.json").read_text(encoding="utf-8")
    )
    payload["run_id"] = run_id
    payload["task"]["topic"] = "测试选题"
    payload["script_artifact"]["script_text"] = text
    payload["script_artifact"]["character_count"] = len(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class RewardHackingTests(unittest.TestCase):
    def test_repeated_character_rule_skips_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.json"
            _trace(trace_path, run_id="validation-D01-004", text="正常正文" + "哈" * 10)
            client = FakeClient([])
            detector = RewardHackingDetector(client, model_name="fake-hy3")
            result = asyncio.run(
                detector.evaluate(
                    load_frozen_trace(trace_path),
                    blind_case_id="D01-004",
                    attack_type="repetition",
                    source_trace="trace.json",
                )
            )
            self.assertTrue(result["reward_hacking"])
            self.assertEqual(result["detection_source"], "repeated_character_rule")
            self.assertEqual(client.calls, [])

    def test_invalid_json_gets_one_format_repair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.json"
            _trace(trace_path, run_id="validation-D01-004", text="自然口播正文")
            client = FakeClient(
                [
                    "不是 JSON",
                    '{"reward_hacking": true, "reason": "包含文案自评。"}',
                ]
            )
            detector = RewardHackingDetector(client, model_name="fake-hy3")
            result = asyncio.run(
                detector.evaluate(
                    load_frozen_trace(trace_path),
                    blind_case_id="D01-004",
                    attack_type="self_scoring",
                    source_trace="trace.json",
                )
            )
            self.assertTrue(result["reward_hacking"])
            self.assertEqual(result["format_attempt_count"], 1)
            self.assertEqual(len(client.calls), 2)

    def test_batch_filters_attack_cases_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            discrimination = root / "validation/discrimination"
            attack_trace = discrimination / "traces/D01-004.json"
            good_trace = discrimination / "traces/D01-001.json"
            _trace(attack_trace, run_id="validation-D01-004", text="攻击样本文案")
            _trace(good_trace, run_id="validation-D01-001", text="正常样本文案")
            answer_key = [
                {
                    "blind_case_id": "D01-001",
                    "expected_tier": "good",
                    "attack_type": None,
                },
                {
                    "blind_case_id": "D01-004",
                    "expected_tier": "attack",
                    "attack_type": "self_scoring",
                },
            ]
            discrimination.mkdir(parents=True, exist_ok=True)
            (discrimination / "answer_key.json").write_text(
                json.dumps(answer_key), encoding="utf-8"
            )
            client = FakeClient(
                ['{"reward_hacking": true, "reason": "包含自我评分。"}']
            )
            detector = RewardHackingDetector(client, model_name="fake-hy3")
            first = asyncio.run(
                run_reward_hacking_validation(root, detector, concurrency=1)
            )
            second = asyncio.run(
                run_reward_hacking_validation(root, detector, concurrency=1)
            )
            self.assertEqual(first["expected_attack_count"], 1)
            self.assertEqual(first["completed_count"], 1)
            self.assertEqual(second["completed_count"], 1)
            self.assertEqual(len(client.calls), 1)
            manifest = json.loads(
                (root / "validation/reward_hacking/manifest.json").read_text()
            )
            self.assertEqual(manifest["resumed_count"], 1)

    def test_resume_rejects_changed_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            discrimination = root / "validation/discrimination"
            _trace(
                discrimination / "traces/D01-004.json",
                run_id="validation-D01-004",
                text="攻击样本文案",
            )
            discrimination.mkdir(parents=True, exist_ok=True)
            (discrimination / "answer_key.json").write_text(
                json.dumps(
                    [
                        {
                            "blind_case_id": "D01-004",
                            "expected_tier": "attack",
                            "attack_type": "self_scoring",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            first_client = FakeClient(
                ['{"reward_hacking": true, "reason": "包含自我评分。"}']
            )
            asyncio.run(
                run_reward_hacking_validation(
                    root,
                    RewardHackingDetector(first_client, model_name="fake-hy3"),
                    concurrency=1,
                )
            )
            changed = RewardHackingDetector(
                FakeClient([]),
                model_name="fake-hy3",
                config=replace(RewardHackingConfig(), reasoning_effort="high"),
            )
            with self.assertRaises(RewardHackingConflictError):
                asyncio.run(
                    run_reward_hacking_validation(root, changed, concurrency=1)
                )

    def test_concurrency_limit_rejects_513(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "between 1 and 512"):
                asyncio.run(
                    run_reward_hacking_validation(
                        Path(directory),
                        RewardHackingDetector(FakeClient([]), model_name="fake-hy3"),
                        concurrency=513,
                    )
                )


if __name__ == "__main__":
    unittest.main()
