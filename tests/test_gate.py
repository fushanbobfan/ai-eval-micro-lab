import unittest

from ai_eval_micro_lab.gate import evaluate_gate


class QualityGateTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
