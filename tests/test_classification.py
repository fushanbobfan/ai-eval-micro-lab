import unittest

import ai_eval_micro_lab
from ai_eval_micro_lab.classification import evaluate_classification


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


if __name__ == "__main__":
    unittest.main()
