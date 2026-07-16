import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from ai_eval_micro_lab.regression_gate import evaluate_regression_gate, main


class RegressionGateTests(unittest.TestCase):
    def test_gate_passes_when_lower_bounds_clear_minimums(self):
        records = [
            {"expected": "blue sky", "baseline": "green", "candidate": "blue sky"},
            {"expected": "warm day", "baseline": "cold", "candidate": "warm day"},
        ]

        report = evaluate_regression_gate(
            records,
            min_exact_match_difference=0.5,
            min_token_f1_difference=0.5,
            samples=50,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["failures"], [])
        self.assertEqual(
            report["comparison"]["difference"]["exact_match"]["lower"],
            1.0,
        )

    def test_gate_reports_lower_bound_shortfalls_in_stable_order(self):
        records = [
            {"expected": "blue sky", "baseline": "blue sky", "candidate": "green"},
            {"expected": "warm day", "baseline": "warm day", "candidate": "warm"},
        ]

        report = evaluate_regression_gate(records, samples=50)

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            ["exact_match", "token_f1"],
        )
        self.assertLess(report["failures"][0]["lower_bound"], 0.0)

    def test_invalid_minimums_are_rejected(self):
        records = [{"expected": "x", "baseline": "x", "candidate": "x"}]
        for minimum in (-1.1, 1.1, True, "zero"):
            with self.subTest(minimum=minimum):
                with self.assertRaisesRegex(ValueError, "minimum difference"):
                    evaluate_regression_gate(
                        records,
                        min_exact_match_difference=minimum,
                        samples=10,
                    )

    def test_comparison_validation_is_preserved(self):
        with self.assertRaisesRegex(ValueError, "at least one record"):
            evaluate_regression_gate([], samples=10)
        with self.assertRaisesRegex(ValueError, "string expected"):
            evaluate_regression_gate([{"expected": "x"}], samples=10)

    def test_cli_returns_zero_for_a_passing_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "comparison.jsonl"
            dataset.write_text(
                json.dumps({"expected": "x", "baseline": "y", "candidate": "x"})
                + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        str(dataset),
                        "--min-exact-match-difference",
                        "0.5",
                        "--samples",
                        "20",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(json.loads(stdout.getvalue())["passed"])

    def test_cli_returns_one_for_a_regression(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "comparison.jsonl"
            dataset.write_text(
                json.dumps({"expected": "x", "baseline": "x", "candidate": "y"})
                + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main([str(dataset), "--samples", "20"])

            self.assertEqual(exit_code, 1)
            self.assertFalse(json.loads(stdout.getvalue())["passed"])

    def test_cli_returns_two_for_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "comparison.jsonl"
            dataset.write_text("{\n", encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                exit_code = main([str(dataset), "--samples", "20"])

            self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
