import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from ai_eval_micro_lab.comparison import compare_records, main, paired_bootstrap


class ComparisonTests(unittest.TestCase):
    def test_constant_improvement_has_exact_interval(self):
        result = paired_bootstrap([0.0, 0.0], [1.0, 1.0], samples=50)

        self.assertEqual(result["mean_difference"], 1.0)
        self.assertEqual(result["lower"], 1.0)
        self.assertEqual(result["upper"], 1.0)

    def test_bootstrap_is_deterministic_for_a_seed(self):
        first = paired_bootstrap([0, 1, 0], [1, 0, 1], samples=100, seed=7)
        second = paired_bootstrap([0, 1, 0], [1, 0, 1], samples=100, seed=7)

        self.assertEqual(first, second)

    def test_invalid_bootstrap_arguments_are_rejected(self):
        cases = [
            ([], [], 10, 0.95),
            ([0], [0, 1], 10, 0.95),
            ([0], [1], 0, 0.95),
            ([0], [1], 10, 1.0),
        ]
        for baseline, candidate, samples, confidence in cases:
            with self.subTest(cases=(baseline, candidate, samples, confidence)):
                with self.assertRaises(ValueError):
                    paired_bootstrap(
                        baseline,
                        candidate,
                        samples=samples,
                        confidence=confidence,
                    )

    def test_record_comparison_reports_both_metrics(self):
        records = [
            {"expected": "blue sky", "baseline": "green", "candidate": "blue sky"},
            {"expected": "warm day", "baseline": "warm day", "candidate": "warm day"},
        ]

        report = compare_records(records, samples=100, seed=11)

        self.assertEqual(report["count"], 2)
        self.assertEqual(report["baseline"]["exact_match"], 0.5)
        self.assertEqual(report["candidate"]["exact_match"], 1.0)
        self.assertEqual(report["difference"]["exact_match"]["mean_difference"], 0.5)
        self.assertIn("token_f1", report["difference"])

    def test_missing_string_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "string expected"):
            compare_records([{"expected": "x", "baseline": "x"}], samples=10)

    def test_cli_reads_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "comparison.jsonl"
            dataset.write_text(
                json.dumps({"expected": "x", "baseline": "y", "candidate": "x"})
                + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main([str(dataset), "--samples", "20"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["count"], 1)

    def test_cli_reports_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "comparison.jsonl"
            dataset.write_text("{\n", encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                exit_code = main([str(dataset), "--samples", "20"])

            self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
