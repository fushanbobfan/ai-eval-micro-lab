import math
import unittest

import ai_eval_micro_lab
from ai_eval_micro_lab.probabilities import evaluate_probabilistic_classification


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


if __name__ == "__main__":
    unittest.main()
