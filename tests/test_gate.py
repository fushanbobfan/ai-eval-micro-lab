import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import ai_eval_micro_lab
from ai_eval_micro_lab.gate import evaluate_gate, main


class QualityGateTests(unittest.TestCase):
    def test_gate_api_is_available_from_package(self):
        self.assertIs(ai_eval_micro_lab.evaluate_gate, evaluate_gate)

    def test_gate_passes_when_metrics_meet_thresholds(self):
        records = [
            {"expected": "blue sky", "predicted": "blue sky"},
            {"expected": "warm day", "predicted": "warm"},
        ]

        report = evaluate_gate(
            records,
            min_exact_match=0.5,
            min_token_f1=0.8,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["metrics"]["count"], 2)
        self.assertEqual(report["failures"], [])

    def test_gate_reports_every_failed_metric_in_stable_order(self):
        report = evaluate_gate(
            [{"expected": "blue sky", "predicted": "green"}],
            min_exact_match=0.75,
            min_token_f1=0.5,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            ["exact_match", "token_f1"],
        )
        self.assertEqual(report["failures"][0]["shortfall"], 0.75)

    def test_invalid_thresholds_are_rejected(self):
        for minimum in (-0.1, 1.1, True, "high"):
            with self.subTest(minimum=minimum):
                with self.assertRaisesRegex(ValueError, "threshold"):
                    evaluate_gate(
                        [{"expected": "x", "predicted": "x"}],
                        min_exact_match=minimum,
                    )

    def test_empty_or_malformed_records_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            evaluate_gate([])
        with self.assertRaisesRegex(ValueError, "string expected"):
            evaluate_gate([{"expected": "x"}])

    def test_cli_returns_zero_for_a_passing_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "predictions.jsonl"
            dataset.write_text(
                json.dumps({"expected": "blue sky", "predicted": "blue sky"})
                + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main([str(dataset), "--min-exact-match", "1"])

            self.assertEqual(exit_code, 0)
            self.assertTrue(json.loads(stdout.getvalue())["passed"])

    def test_cli_returns_one_and_explains_a_quality_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "predictions.jsonl"
            dataset.write_text(
                json.dumps({"expected": "blue sky", "predicted": "green"}) + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main([str(dataset), "--min-token-f1", "0.5"])

            report = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertFalse(report["passed"])
            self.assertEqual(report["failures"][0]["metric"], "token_f1")

    def test_cli_returns_two_for_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "predictions.jsonl"
            dataset.write_text("{\n", encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                exit_code = main([str(dataset)])

            self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
