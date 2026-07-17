import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import ai_eval_micro_lab
from ai_eval_micro_lab.calibration import evaluate_calibration, main


class CalibrationTests(unittest.TestCase):
    def test_calibration_api_is_available_from_package(self):
        self.assertIs(ai_eval_micro_lab.evaluate_calibration, evaluate_calibration)

    def test_report_contains_weighted_ece_brier_score_and_bins(self):
        report = evaluate_calibration(
            [
                {"expected": "Blue Sky", "predicted": " blue sky ", "confidence": 0.9},
                {"expected": "warm", "predicted": "cold", "confidence": 0.8},
                {"expected": "dry", "predicted": "dry", "confidence": 0.6},
                {"expected": "up", "predicted": "down", "confidence": 0.2},
            ],
            bins=2,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["metrics"]["count"], 4)
        self.assertEqual(report["metrics"]["accuracy"], 0.5)
        self.assertEqual(report["metrics"]["mean_confidence"], 0.625)
        self.assertAlmostEqual(report["metrics"]["brier_score"], 0.2125)
        self.assertAlmostEqual(
            report["metrics"]["expected_calibration_error"], 0.125
        )
        self.assertEqual([item["index"] for item in report["bins"]], [0, 1])
        self.assertEqual([item["count"] for item in report["bins"]], [1, 3])

    def test_threshold_failures_are_reported_in_stable_order(self):
        report = evaluate_calibration(
            [{"expected": "yes", "predicted": "no", "confidence": 1.0}],
            bins=5,
            max_ece=0.2,
            max_brier=0.3,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            ["expected_calibration_error", "brier_score"],
        )
        self.assertEqual(report["failures"][0]["excess"], 0.8)

    def test_empty_or_malformed_records_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            evaluate_calibration([])
        with self.assertRaisesRegex(ValueError, "string expected"):
            evaluate_calibration([{"expected": "x", "confidence": 0.5}])
        for confidence in (-0.1, 1.1, True, float("nan"), "high"):
            with self.subTest(confidence=confidence):
                with self.assertRaisesRegex(ValueError, "confidence"):
                    evaluate_calibration(
                        [{"expected": "x", "predicted": "x", "confidence": confidence}]
                    )

    def test_invalid_bins_and_thresholds_are_rejected(self):
        records = [{"expected": "x", "predicted": "x", "confidence": 0.5}]
        for bins in (0, -1, 1.5, True):
            with self.subTest(bins=bins):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    evaluate_calibration(records, bins=bins)
        for maximum in (-0.1, 1.1, True, float("inf"), "low"):
            with self.subTest(maximum=maximum):
                with self.assertRaisesRegex(ValueError, "threshold"):
                    evaluate_calibration(records, max_ece=maximum)

    def test_cli_returns_zero_for_a_passing_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "calibration.jsonl"
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
                    [str(dataset), "--bins", "5", "--max-ece", "0.2", "--max-brier", "0.02"]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(json.loads(stdout.getvalue())["passed"])

    def test_cli_returns_one_and_reports_threshold_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "calibration.jsonl"
            dataset.write_text(
                json.dumps(
                    {"expected": "blue", "predicted": "green", "confidence": 0.9}
                )
                + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main([str(dataset), "--max-ece", "0.2"])

            report = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertFalse(report["passed"])
            self.assertEqual(
                report["failures"][0]["metric"], "expected_calibration_error"
            )

    def test_cli_returns_two_for_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "calibration.jsonl"
            dataset.write_text("{\n", encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                exit_code = main([str(dataset)])

            self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
