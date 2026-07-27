import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import ai_eval_micro_lab
from ai_eval_micro_lab.multilabel import evaluate_multilabel, main


class MultilabelEvaluationTests(unittest.TestCase):
    def test_report_contains_deterministic_per_label_and_aggregate_metrics(self):
        report = evaluate_multilabel(
            [
                {
                    "expected": ["cat", "indoor"],
                    "predicted": ["indoor", "cat"],
                },
                {"expected": ["dog", "outdoor"], "predicted": ["dog"]},
                {"expected": ["cat", "outdoor"], "predicted": ["dog", "outdoor"]},
            ]
        )

        self.assertIs(ai_eval_micro_lab.evaluate_multilabel, evaluate_multilabel)
        self.assertEqual(report["labels"], ["cat", "dog", "indoor", "outdoor"])
        self.assertEqual(
            [item["label"] for item in report["per_label"]],
            report["labels"],
        )
        self.assertEqual(report["per_label"][0]["support"], 2)
        self.assertEqual(report["per_label"][1]["false_positive"], 1)
        self.assertEqual(report["per_label"][2]["true_negative"], 2)
        self.assertAlmostEqual(report["metrics"]["subset_accuracy"], 1 / 3)
        self.assertAlmostEqual(report["metrics"]["hamming_loss"], 0.25)
        self.assertAlmostEqual(report["metrics"]["samples_jaccard"], 11 / 18)
        self.assertAlmostEqual(report["metrics"]["micro_f1"], 8 / 11)
        self.assertAlmostEqual(report["metrics"]["macro_f1"], 0.75)
        self.assertAlmostEqual(report["metrics"]["weighted_f1"], 13 / 18)
        self.assertEqual(report["metrics"]["average_expected_labels"], 2)
        self.assertAlmostEqual(
            report["metrics"]["average_predicted_labels"],
            5 / 3,
        )

    def test_empty_label_sets_are_valid_when_a_label_exists_elsewhere(self):
        report = evaluate_multilabel(
            [
                {"expected": [], "predicted": []},
                {"expected": ["cat"], "predicted": []},
            ]
        )

        self.assertEqual(report["labels"], ["cat"])
        self.assertEqual(report["metrics"]["samples_jaccard"], 0.5)
        self.assertEqual(report["metrics"]["micro_f1"], 0.0)

    def test_threshold_failures_are_reported_in_stable_order(self):
        report = evaluate_multilabel(
            [{"expected": ["cat"], "predicted": ["dog"]}],
            min_micro_f1=0.5,
            min_macro_f1=0.25,
            max_hamming_loss=0.5,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            ["micro_f1", "macro_f1", "hamming_loss"],
        )
        self.assertEqual(report["failures"][0]["shortfall"], 0.5)
        self.assertEqual(report["failures"][2]["excess"], 0.5)

    def test_invalid_records_are_rejected(self):
        cases = [
            ([], "at least one record"),
            ([{"expected": [], "predicted": []}], "at least one label"),
            ([{"expected": "cat", "predicted": ["cat"]}], "array"),
            ([{"expected": ["cat", "cat"], "predicted": []}], "duplicate"),
            ([{"expected": ["cat"], "predicted": [3]}], "non-empty string"),
            ([{"expected": ["cat"], "predicted": [""]}], "non-empty string"),
            ([3], "object"),
        ]
        for records, message in cases:
            with self.subTest(records=records):
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_multilabel(records)

    def test_invalid_thresholds_are_rejected(self):
        records = [{"expected": ["cat"], "predicted": ["cat"]}]
        for value in (-0.1, 1.1, True, float("inf"), "high"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "threshold"):
                    evaluate_multilabel(records, min_micro_f1=value)

    def test_cli_returns_zero_for_a_passing_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "multilabel.jsonl"
            dataset.write_text(
                json.dumps(
                    {"expected": ["cat", "indoor"], "predicted": ["indoor", "cat"]}
                )
                + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        str(dataset),
                        "--min-micro-f1",
                        "1",
                        "--min-macro-f1",
                        "1",
                        "--max-hamming-loss",
                        "0",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(json.loads(stdout.getvalue())["passed"])

    def test_cli_returns_one_with_structured_threshold_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "multilabel.jsonl"
            dataset.write_text(
                json.dumps({"expected": ["cat"], "predicted": ["dog"]}) + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        str(dataset),
                        "--min-micro-f1",
                        "0.5",
                        "--max-hamming-loss",
                        "0.5",
                    ]
                )

            report = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertEqual(
                [failure["metric"] for failure in report["failures"]],
                ["micro_f1", "hamming_loss"],
            )

    def test_cli_returns_two_for_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "multilabel.jsonl"
            dataset.write_text("{\n", encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                exit_code = main([str(dataset)])

            self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
