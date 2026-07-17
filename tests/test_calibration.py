import unittest

from ai_eval_micro_lab.calibration import evaluate_calibration


class CalibrationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
