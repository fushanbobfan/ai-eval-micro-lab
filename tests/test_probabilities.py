import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import ai_eval_micro_lab
from ai_eval_micro_lab.probabilities import (
    evaluate_probabilistic_classification,
    main,
)


class ProbabilisticClassificationTests(unittest.TestCase):
    def test_report_contains_proper_scores_top_k_and_calibration(self):
        records = [
            {
                "expected": "cat",
                "scores": {"cat": 0.7, "dog": 0.2, "bird": 0.1},
            },
            {
                "expected": "dog",
                "scores": {"cat": 0.5, "dog": 0.4, "bird": 0.1},
            },
            {
                "expected": "bird",
                "scores": {"cat": 0.2, "dog": 0.2, "bird": 0.6},
            },
        ]

        report = evaluate_probabilistic_classification(
            records,
            bins=2,
            top_k=(1, 2),
            gate_top_k=2,
        )

        self.assertIs(
            ai_eval_micro_lab.evaluate_probabilistic_classification,
            evaluate_probabilistic_classification,
        )
        self.assertEqual(report["labels"], ["bird", "cat", "dog"])
        self.assertEqual(report["metrics"]["count"], 3)
        self.assertAlmostEqual(report["metrics"]["accuracy"], 2 / 3)
        self.assertEqual(report["metrics"]["top_k_accuracy"], {"1": 2 / 3, "2": 1.0})
        self.assertAlmostEqual(
            report["metrics"]["log_loss"],
            -(math.log(0.7) + math.log(0.4) + math.log(0.6)) / 3,
        )
        self.assertAlmostEqual(report["metrics"]["brier_score"], 1.0 / 3)
        self.assertAlmostEqual(
            report["metrics"]["expected_calibration_error"],
            abs(0.6 - 2 / 3),
        )
        self.assertEqual(
            [item["label"] for item in report["per_class"]],
            ["bird", "cat", "dog"],
        )
        self.assertEqual(report["per_class"][1]["predicted_count"], 2)

    def test_equal_probabilities_use_label_order_for_stable_ranking(self):
        report = evaluate_probabilistic_classification(
            [
                {
                    "expected": "beta",
                    "scores": {"beta": 0.5, "alpha": 0.5},
                }
            ],
            top_k=(1, 2),
            gate_top_k=2,
        )

        self.assertEqual(report["metrics"]["accuracy"], 0.0)
        self.assertEqual(report["metrics"]["top_k_accuracy"]["2"], 1.0)
        self.assertEqual(report["per_class"][0]["predicted_count"], 1)

    def test_threshold_failures_are_reported_in_stable_order(self):
        report = evaluate_probabilistic_classification(
            [{"expected": "yes", "scores": {"yes": 0.1, "no": 0.9}}],
            min_accuracy=0.5,
            min_top_k_accuracy=0.5,
            max_log_loss=1.0,
            max_brier_score=0.5,
            max_ece=0.5,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            [
                "accuracy",
                "top_1_accuracy",
                "log_loss",
                "brier_score",
                "expected_calibration_error",
            ],
        )

    def test_probability_and_label_validation_is_strict(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            evaluate_probabilistic_classification(
                [{"expected": "x", "scores": {"x": 0.7, "y": 0.4}}]
            )
        with self.assertRaisesRegex(ValueError, "same score labels"):
            evaluate_probabilistic_classification(
                [
                    {"expected": "x", "scores": {"x": 0.5, "y": 0.5}},
                    {"expected": "x", "scores": {"x": 0.5, "z": 0.5}},
                ]
            )
        with self.assertRaisesRegex(ValueError, "missing from scores"):
            evaluate_probabilistic_classification(
                [{"expected": "z", "scores": {"x": 0.5, "y": 0.5}}]
            )

    def test_zero_expected_probability_uses_the_configured_log_floor(self):
        report = evaluate_probabilistic_classification(
            [{"expected": "yes", "scores": {"yes": 0.0, "no": 1.0}}],
            log_floor=1e-6,
        )

        self.assertEqual(report["metrics"]["zero_expected_probability_count"], 1)
        self.assertAlmostEqual(report["metrics"]["log_loss"], -math.log(1e-6))
        self.assertTrue(math.isfinite(report["metrics"]["log_loss"]))

    def test_invalid_configuration_and_scores_are_rejected(self):
        records = [{"expected": "x", "scores": {"x": 0.8, "y": 0.2}}]
        for kwargs in (
            {"bins": True},
            {"top_k": (0,)},
            {"top_k": (1,), "gate_top_k": True},
            {"top_k": (3,)},
            {"probability_tolerance": 0.2},
            {"log_floor": 0},
            {"max_brier_score": 2.1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    evaluate_probabilistic_classification(records, **kwargs)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            evaluate_probabilistic_classification(
                [{"expected": "x", "scores": {"x": True, "y": 0.0}}]
            )

    def test_cli_writes_a_report_and_returns_one_for_a_failed_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "probabilities.jsonl"
            output = root / "report.json"
            dataset.write_text(
                json.dumps(
                    {"expected": "yes", "scores": {"yes": 0.2, "no": 0.8}}
                )
                + "\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    str(dataset),
                    "--top-k",
                    "2",
                    "--gate-top-k",
                    "2",
                    "--min-top-k-accuracy",
                    "1",
                    "--max-log-loss",
                    "1",
                    "--output",
                    str(output),
                ]
            )

            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 1)
            self.assertFalse(report["passed"])
            self.assertEqual(report["metrics"]["top_k_accuracy"]["2"], 1.0)
            self.assertEqual(report["failures"][0]["metric"], "log_loss")

    def test_cli_prints_a_passing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "probabilities.jsonl"
            dataset.write_text(
                json.dumps(
                    {"expected": "yes", "scores": {"yes": 0.9, "no": 0.1}}
                )
                + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main([str(dataset), "--min-accuracy", "1"])

            self.assertEqual(exit_code, 0)
            self.assertTrue(json.loads(stdout.getvalue())["passed"])

    def test_cli_returns_two_for_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "probabilities.jsonl"
            dataset.write_text("{\n", encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                exit_code = main([str(dataset)])

            self.assertEqual(exit_code, 2)

    def test_cli_refuses_to_overwrite_the_source_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "probabilities.jsonl"
            original = (
                json.dumps(
                    {"expected": "yes", "scores": {"yes": 0.9, "no": 0.1}}
                )
                + "\n"
            )
            dataset.write_text(original, encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                exit_code = main([str(dataset), "--output", str(dataset)])

            self.assertEqual(exit_code, 2)
            self.assertEqual(dataset.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
