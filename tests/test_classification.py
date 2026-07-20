import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import ai_eval_micro_lab
from ai_eval_micro_lab.classification import evaluate_classification, main


class ClassificationTests(unittest.TestCase):
    def test_report_contains_deterministic_confusion_and_per_class_metrics(self):
        records = [
            {"expected": "cat", "predicted": "cat"},
            {"expected": "cat", "predicted": "dog"},
            {"expected": "dog", "predicted": "dog"},
            {"expected": "dog", "predicted": "bird"},
        ]

        report = evaluate_classification(records)

        self.assertIs(
            ai_eval_micro_lab.evaluate_classification,
            evaluate_classification,
        )
        self.assertEqual(report["labels"], ["bird", "cat", "dog"])
        self.assertEqual(
            report["confusion_matrix"],
            [[0, 0, 0], [0, 1, 1], [1, 0, 1]],
        )
        self.assertEqual(report["metrics"]["count"], 4)
        self.assertEqual(report["metrics"]["accuracy"], 0.5)
        self.assertAlmostEqual(report["metrics"]["macro_f1"], 7 / 18)
        self.assertAlmostEqual(report["metrics"]["weighted_f1"], 7 / 12)
        self.assertEqual(
            [item["label"] for item in report["per_class"]],
            ["bird", "cat", "dog"],
        )
        self.assertEqual(report["per_class"][0]["support"], 0)
        self.assertEqual(report["per_class"][2]["predicted_count"], 2)

    def test_threshold_failures_are_reported_in_stable_order(self):
        report = evaluate_classification(
            [
                {"expected": "cat", "predicted": "cat"},
                {"expected": "dog", "predicted": "cat"},
            ],
            min_accuracy=0.75,
            min_macro_f1=0.6,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            ["accuracy", "macro_f1"],
        )
        self.assertEqual(report["failures"][0]["shortfall"], 0.25)

    def test_empty_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            evaluate_classification([{"expected": "", "predicted": "cat"}])

    def test_empty_or_malformed_records_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            evaluate_classification([])
        with self.assertRaisesRegex(ValueError, "string expected"):
            evaluate_classification([{"expected": "cat", "predicted": 3}])

    def test_invalid_thresholds_are_rejected(self):
        records = [{"expected": "cat", "predicted": "cat"}]
        for minimum in (-0.1, 1.1, True, float("inf"), "high"):
            with self.subTest(minimum=minimum):
                with self.assertRaisesRegex(ValueError, "threshold"):
                    evaluate_classification(records, min_macro_f1=minimum)

    def test_cli_returns_one_with_structured_threshold_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "classification.jsonl"
            dataset.write_text(
                json.dumps({"expected": "cat", "predicted": "dog"}) + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main([str(dataset), "--min-accuracy", "0.5"])

            report = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertFalse(report["passed"])
            self.assertEqual(report["failures"][0]["metric"], "accuracy")

    def test_cli_returns_zero_for_a_passing_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "classification.jsonl"
            dataset.write_text(
                json.dumps({"expected": "cat", "predicted": "cat"}) + "\n",
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main([str(dataset), "--min-macro-f1", "1"])

            self.assertEqual(exit_code, 0)

    def test_cli_returns_two_for_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "classification.jsonl"
            dataset.write_text("{\n", encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                exit_code = main([str(dataset)])

            self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
