import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import ai_eval_micro_lab
from ai_eval_micro_lab.selective import evaluate_selective_prediction, main


class SelectivePredictionTests(unittest.TestCase):
    def test_selective_api_is_available_from_package(self):
        self.assertIs(
            ai_eval_micro_lab.evaluate_selective_prediction,
            evaluate_selective_prediction,
        )

    def test_report_contains_operating_point_curve_and_area(self):
        report = evaluate_selective_prediction(
            [
                {"expected": "yes", "predicted": "yes", "confidence": 0.9},
                {"expected": "up", "predicted": "down", "confidence": 0.8},
                {"expected": "blue sky", "predicted": "Blue sky!", "confidence": 0.6},
                {"expected": "warm", "predicted": "cold", "confidence": 0.2},
            ],
            confidence_threshold=0.6,
            min_coverage=0.7,
            max_risk=0.4,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["operating_point"]["accepted_count"], 3)
        self.assertEqual(report["operating_point"]["abstained_count"], 1)
        self.assertEqual(report["operating_point"]["coverage"], 0.75)
        self.assertAlmostEqual(
            report["operating_point"]["selective_risk"], 1 / 3
        )
        self.assertEqual(
            [point["confidence_threshold"] for point in report["risk_coverage_curve"]],
            [0.9, 0.8, 0.6, 0.2],
        )
        self.assertAlmostEqual(report["risk_coverage_area"], 1 / 3)

    def test_equal_confidences_form_one_permutation_invariant_curve_point(self):
        records = [
            {"expected": "yes", "predicted": "yes", "confidence": 0.8},
            {"expected": "up", "predicted": "down", "confidence": 0.8},
            {"expected": "blue", "predicted": "blue", "confidence": 0.4},
        ]

        forward = evaluate_selective_prediction(records)
        reverse = evaluate_selective_prediction(list(reversed(records)))

        self.assertEqual(forward["risk_coverage_curve"], reverse["risk_coverage_curve"])
        self.assertEqual(len(forward["risk_coverage_curve"]), 2)
        self.assertEqual(forward["risk_coverage_curve"][0]["accepted_count"], 2)

    def test_threshold_failures_have_stable_order(self):
        report = evaluate_selective_prediction(
            [
                {"expected": "yes", "predicted": "no", "confidence": 0.8},
                {"expected": "up", "predicted": "up", "confidence": 0.2},
            ],
            confidence_threshold=0.8,
            min_coverage=0.75,
            max_risk=0.2,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            ["coverage", "selective_risk"],
        )
        self.assertEqual(report["failures"][0]["shortfall"], 0.25)
        self.assertEqual(report["failures"][1]["excess"], 0.8)

    def test_empty_selection_reports_null_risk(self):
        report = evaluate_selective_prediction(
            [{"expected": "yes", "predicted": "yes", "confidence": 0.8}],
            confidence_threshold=1.0,
        )

        self.assertFalse(report["passed"])
        self.assertIsNone(report["operating_point"]["selective_accuracy"])
        self.assertIsNone(report["failures"][0]["actual"])
        self.assertIn("no predictions", report["failures"][0]["reason"])

    def test_invalid_records_and_configuration_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            evaluate_selective_prediction([])
        with self.assertRaisesRegex(ValueError, "string expected"):
            evaluate_selective_prediction(
                [{"expected": "yes", "confidence": 0.8}]
            )
        for confidence in (-0.1, 1.1, True, float("nan"), "high"):
            with self.subTest(confidence=confidence):
                with self.assertRaisesRegex(ValueError, "confidence"):
                    evaluate_selective_prediction(
                        [
                            {
                                "expected": "yes",
                                "predicted": "yes",
                                "confidence": confidence,
                            }
                        ]
                    )
        for name, value in (
            ("confidence_threshold", -0.1),
            ("min_coverage", 1.1),
            ("max_risk", float("inf")),
        ):
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(ValueError, name):
                    evaluate_selective_prediction(
                        [
                            {
                                "expected": "yes",
                                "predicted": "yes",
                                "confidence": 0.8,
                            }
                        ],
                        **{name: value},
                    )

    def test_cli_returns_zero_for_a_passing_operating_point(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "selective.jsonl"
            dataset.write_text(
                json.dumps(
                    {"expected": "blue", "predicted": "blue", "confidence": 0.9}
                )
                + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        str(dataset),
                        "--confidence-threshold",
                        "0.8",
                        "--min-coverage",
                        "1.0",
                        "--max-risk",
                        "0.0",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(json.loads(stdout.getvalue())["passed"])

    def test_cli_returns_one_for_a_failed_operating_point(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "selective.jsonl"
            dataset.write_text(
                json.dumps(
                    {"expected": "blue", "predicted": "green", "confidence": 0.9}
                )
                + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main([str(dataset), "--max-risk", "0.2"])

            self.assertEqual(exit_code, 1)
            self.assertEqual(
                json.loads(stdout.getvalue())["failures"][0]["metric"],
                "selective_risk",
            )

    def test_cli_returns_two_for_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "selective.jsonl"
            dataset.write_text("{\n", encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                exit_code = main([str(dataset)])

            self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
