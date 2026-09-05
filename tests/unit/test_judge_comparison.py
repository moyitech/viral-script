"""Tests for the versioned Hy3-versus-Luna Judge comparison."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hyscript.evaluation import judge_comparison as comparison


DIMENSIONS = ("topic_alignment", "length_compliance")


def _workflow_rows(score: float) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for index in range(comparison.EXPECTED_TRACE_COUNT):
        task_id = f"T{index + 1:03d}-L280"
        rows[task_id] = {
            "task_id": task_id,
            "run_id": f"run-{index}",
            "topic": f"topic-{index}",
            "target_length": 280,
            "domain": "test",
            "challenge_tags": "test",
            "trace_sha256": f"{index:064x}"[-64:],
            "gate_failed": False,
            "final_score": score,
            "topic_alignment": 3 if score == 1.0 else 2,
            "length_compliance": 3,
        }
    return rows


def _judge_record(run_id: str, trace_sha256: str, scores: tuple[int, int], model: str) -> dict:
    return {
        "run_id": run_id,
        "trace_sha256": trace_sha256,
        "status": "completed",
        "gate_failed": False,
        "dimension_scores": [
            {"dimension_id": "topic_alignment", "score": scores[0]},
            {"dimension_id": "theme_information", "score": scores[1]},
        ],
        "metrics": {"normalized_score": sum(scores) / 6},
        "metadata": {
            "evaluator_fingerprint": {
                "sha256": ("a" if model == "hy3" else "b") * 64,
                "model": model,
            }
        },
    }


class JudgeComparisonTests(unittest.TestCase):
    def test_workflow_comparison_requires_and_scores_300_pairs(self) -> None:
        baseline = _workflow_rows(1.0)
        single_shot = _workflow_rows(0.9)

        summary, rows = comparison._paired_workflows(
            baseline, single_shot, DIMENSIONS
        )

        self.assertEqual(summary["pair_count"], 300)
        self.assertEqual(summary["losses"], 300)
        self.assertAlmostEqual(summary["mean_delta"], -0.1)
        self.assertEqual(len(rows), 300)
        with self.assertRaisesRegex(ValueError, "300 exact task pairs"):
            comparison._paired_workflows(
                dict(list(baseline.items())[:-1]), single_shot, DIMENSIONS
            )

    def test_cross_judge_reports_dimension_and_gate_agreement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hy3_dir = root / "hy3"
            luna_dir = root / "luna"
            manifest: dict[str, dict] = {}
            for index, (hy3_scores, luna_scores) in enumerate(
                (((3, 3), (2, 3)), ((2, 3), (2, 3))), start=1
            ):
                task_id = f"T{index:03d}-L280"
                run_id = f"run-{index}"
                trace_sha256 = str(index) * 64
                manifest[task_id] = {
                    "task_id": task_id,
                    "run_id": run_id,
                    "trace_sha256": trace_sha256,
                    "topic": task_id,
                    "target_length": 280,
                }
                for target, record in (
                    (hy3_dir, _judge_record(run_id, trace_sha256, hy3_scores, "hy3")),
                    (
                        luna_dir,
                        _judge_record(
                            run_id,
                            trace_sha256,
                            luna_scores,
                            comparison.LUNA_MODEL_ID,
                        ),
                    ),
                ):
                    item = target / "items" / run_id
                    item.mkdir(parents=True, exist_ok=True)
                    (item / "hy3_judge.json").write_text(
                        json.dumps(record), encoding="utf-8"
                    )

            summary, rows = comparison._cross_judge(
                hy3_dir,
                luna_dir,
                manifest,
                ("topic_alignment", "theme_information"),
            )
            glm_summary, glm_rows = comparison._cross_judge(
                hy3_dir,
                luna_dir,
                manifest,
                ("topic_alignment", "theme_information"),
                candidate_key="glm",
            )

        self.assertEqual(summary["record_count"], 2)
        self.assertEqual(summary["overall"]["dimension_exact_agreement_rate"], 0.75)
        self.assertEqual(summary["overall"]["all_dimensions_exact_rate"], 0.5)
        self.assertIn(
            "quadratic_weighted_kappa", summary["dimensions"]["topic_alignment"]
        )
        self.assertAlmostEqual(
            summary["dimensions"]["theme_information"]["quadratic_weighted_kappa"],
            1.0,
        )
        self.assertEqual(rows[0]["changed_dimensions"], "topic_alignment")
        self.assertIn("glm_fingerprint_sha256", glm_summary)
        self.assertIn("glm_normalized_score", glm_rows[0])
        self.assertIn("glm_topic_alignment", glm_rows[0])
        self.assertNotIn("luna_normalized_score", glm_rows[0])

    def test_main_shortfalls_uses_three_largest_negative_deltas(self) -> None:
        workflow_summary = {
            "dimensions": {
                "small": {"mean_delta": -0.01},
                "largest": {"mean_delta": -0.5},
                "positive": {"mean_delta": 0.2},
                "third": {"mean_delta": -0.1},
                "second": {"mean_delta": -0.3},
            }
        }

        self.assertEqual(
            comparison._main_shortfalls(workflow_summary),
            ["largest", "second", "third"],
        )

    def test_score_commands_pin_luna_model_xhigh_and_512(self) -> None:
        config = {
            "rubric": "rubric.json",
            "judges": {
                "hy3": {
                    "model_id": "hy3",
                    "display_name": "hy3",
                    "reasoning_effort": "high",
                    "result_source": "existing",
                },
                "luna": comparison.LUNA_CANDIDATE.config(),
            },
            "sources": {
                "baseline": {"trace_manifest": "baseline.json"},
                "single_shot": {"trace_manifest": "single.json"},
            },
        }
        manifests = {"baseline": {}, "single_shot": {}}
        commands: list[list[str]] = []
        with patch.object(
            comparison, "_load_experiment", return_value=(config, manifests)
        ), patch.object(
            comparison,
            "_run_command",
            side_effect=lambda arguments: commands.append(list(arguments)) or 0,
        ), patch.object(comparison, "_validate_result_set", return_value={}):
            comparison.score_comparison(Path("comparison"))

        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertEqual(command[command.index("--concurrency") + 1], "512")
            self.assertEqual(
                command[command.index("--judge-model-id") + 1],
                comparison.LUNA_MODEL_ID,
            )
            self.assertEqual(
                command[command.index("--reasoning-effort") + 1], "xhigh"
            )
            self.assertEqual(command[command.index("--evaluators") + 1], "rules,judge")

    def test_score_commands_pin_glm_model_max_and_512(self) -> None:
        config = {
            "rubric": "rubric.json",
            "judges": {
                "hy3": {
                    "model_id": "hy3",
                    "display_name": "hy3",
                    "reasoning_effort": "high",
                    "result_source": "existing",
                },
                "glm": comparison.GLM_CANDIDATE.config(),
            },
            "sources": {
                "baseline": {"trace_manifest": "baseline.json"},
                "single_shot": {"trace_manifest": "single.json"},
            },
        }
        manifests = {"baseline": {}, "single_shot": {}}
        commands: list[list[str]] = []
        with patch.object(
            comparison, "_load_experiment", return_value=(config, manifests)
        ), patch.object(
            comparison,
            "_run_command",
            side_effect=lambda arguments: commands.append(list(arguments)) or 0,
        ), patch.object(comparison, "_validate_result_set", return_value={}):
            comparison.score_comparison(Path("comparison"))

        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertEqual(command[command.index("--concurrency") + 1], "512")
            self.assertEqual(
                command[command.index("--judge-model-id") + 1],
                comparison.GLM_MODEL_ID,
            )
            self.assertEqual(command[command.index("--reasoning-effort") + 1], "max")
            self.assertIn("/glm/pass-001", command[command.index("--output-dir") + 1])

    def test_prepare_records_alias_endpoint_hash_and_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            single = root / "single"
            output = root / "comparison"
            rubric = root / "rubric.json"
            rubric.write_text("{}", encoding="utf-8")
            for source in (baseline, single):
                source.mkdir()
                (source / "experiment.json").write_text(
                    json.dumps({"rubric": "../rubric.json"}), encoding="utf-8"
                )
            source_descriptors = {
                str(baseline.resolve()): {"trace_manifest": "../baseline-manifest.json"},
                str(single.resolve()): {"trace_manifest": "../single-manifest.json"},
            }
            items = {
                f"T{index + 1:03d}-L280": {"run_id": f"run-{index}"}
                for index in range(comparison.EXPECTED_TRACE_COUNT)
            }

            def descriptor(_output: Path, source: Path) -> dict:
                return source_descriptors[str(source.resolve())]

            with patch.object(
                comparison, "_source_descriptor", side_effect=descriptor
            ), patch.object(
                comparison, "_manifest_items", return_value=items
            ), patch.object(
                comparison,
                "get_settings",
                return_value=SimpleNamespace(
                    hy3=SimpleNamespace(openai_base_url="https://example.test/v1")
                ),
            ):
                config = comparison.prepare_comparison(
                    output, baseline_dir=baseline, single_shot_dir=single
                )

        self.assertEqual(config["concurrency"]["judge"], 512)
        self.assertEqual(
            config["judges"]["luna"]["display_name"], "gpt-5.6-luna"
        )
        self.assertEqual(config["excluded_validation_sets"], ["discrimination"])
        self.assertEqual(len(config["provider"]["endpoint_sha256"]), 64)

    def test_prepare_accepts_and_records_glm_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            single = root / "single"
            output = root / "comparison"
            rubric = root / "rubric.json"
            rubric.write_text("{}", encoding="utf-8")
            for source in (baseline, single):
                source.mkdir()
                (source / "experiment.json").write_text(
                    json.dumps({"rubric": "../rubric.json"}), encoding="utf-8"
                )
            items = {
                f"T{index + 1:03d}-L280": {"run_id": f"run-{index}"}
                for index in range(comparison.EXPECTED_TRACE_COUNT)
            }
            with patch.object(
                comparison,
                "_source_descriptor",
                side_effect=(
                    {"trace_manifest": "../baseline-manifest.json"},
                    {"trace_manifest": "../single-manifest.json"},
                ),
            ), patch.object(
                comparison, "_manifest_items", return_value=items
            ), patch.object(
                comparison,
                "get_settings",
                return_value=SimpleNamespace(
                    hy3=SimpleNamespace(openai_base_url="https://example.test/v1")
                ),
            ):
                config = comparison.prepare_comparison(
                    output,
                    baseline_dir=baseline,
                    single_shot_dir=single,
                    candidate=comparison.GLM_CANDIDATE,
                )

        self.assertEqual(config["judges"]["glm"], comparison.GLM_CANDIDATE.config())
        self.assertNotIn("luna", config["judges"])


if __name__ == "__main__":
    unittest.main()
