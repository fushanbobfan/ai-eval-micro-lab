import unittest

from ai_eval_micro_lab.position_bias import audit_position_bias


class PositionBiasAuditTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"pair_id": "robust", "first": "a", "second": "b", "winner": "first"},
            {"pair_id": "robust", "first": "b", "second": "a", "winner": "second"},
            {"pair_id": "tie", "first": "a", "second": "b", "winner": "tie"},
            {"pair_id": "tie", "first": "b", "second": "a", "winner": "tie"},
            {"pair_id": "first-flip", "first": "a", "second": "b", "winner": "first"},
            {"pair_id": "first-flip", "first": "b", "second": "a", "winner": "first"},
            {"pair_id": "second-flip", "first": "a", "second": "b", "winner": "second"},
            {"pair_id": "second-flip", "first": "b", "second": "a", "winner": "second"},
            {"pair_id": "tie-change", "first": "a", "second": "b", "winner": "tie"},
            {"pair_id": "tie-change", "first": "b", "second": "a", "winner": "first"},
            {"pair_id": "missing", "first": "a", "second": "b", "winner": "first"},
        ]

    def test_report_classifies_reversed_presentations(self):
        report = audit_position_bias(self.records, max_details=1)
        metrics = report["metrics"]

        self.assertEqual(metrics["record_count"], 11)
        self.assertEqual(metrics["pair_count"], 6)
        self.assertEqual(metrics["complete_pairs"], 5)
        self.assertEqual(metrics["incomplete_pairs"], 1)
        self.assertEqual(metrics["robust_preference_pairs"], 1)
        self.assertEqual(metrics["stable_tie_pairs"], 1)
        self.assertEqual(metrics["first_position_flip_pairs"], 1)
        self.assertEqual(metrics["second_position_flip_pairs"], 1)
        self.assertEqual(metrics["tie_instability_pairs"], 1)
        self.assertEqual(metrics["position_flip_pairs"], 2)
        self.assertAlmostEqual(metrics["robust_preference_rate"], 0.2)
        self.assertAlmostEqual(metrics["stable_tie_rate"], 0.2)
        self.assertAlmostEqual(metrics["position_flip_rate"], 0.4)
        self.assertAlmostEqual(metrics["tie_instability_rate"], 0.2)
        self.assertEqual(
            report["unstable_details"],
            [{"pair_id": "first-flip", "classification": "first_position_flip"}],
        )
        self.assertEqual(
            report["incomplete_details"],
            [{"pair_id": "missing", "presentations": 1}],
        )
        self.assertTrue(report["details_truncated"])

    def test_gates_report_failures_in_stable_order(self):
        report = audit_position_bias(
            self.records,
            min_complete_pairs=6,
            min_robust_preference_rate=0.3,
            max_position_flip_rate=0.3,
            max_tie_instability_rate=0.1,
            max_incomplete_pairs=0,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            [
                "complete_pairs",
                "robust_preference_rate",
                "position_flip_rate",
                "tie_instability_rate",
                "incomplete_pairs",
            ],
        )

    def test_no_complete_pairs_keeps_rates_undefined(self):
        records = [
            {"pair_id": "missing", "first": "a", "second": "b", "winner": "first"}
        ]

        report = audit_position_bias(
            records,
            min_robust_preference_rate=0.0,
            max_position_flip_rate=1.0,
        )

        self.assertIsNone(report["metrics"]["robust_preference_rate"])
        self.assertIsNone(report["metrics"]["position_flip_rate"])
        self.assertEqual(
            [failure["reason"] for failure in report["failures"]],
            ["no_complete_pairs", "no_complete_pairs"],
        )

    def test_invalid_records_and_configuration_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            audit_position_bias([])
        with self.assertRaisesRegex(ValueError, "distinct"):
            audit_position_bias(self.records, winner_field="first")
        with self.assertRaisesRegex(ValueError, "candidates must be distinct"):
            audit_position_bias(
                [{"pair_id": "p", "first": "a", "second": "a", "winner": "tie"}]
            )
        with self.assertRaisesRegex(ValueError, "first, second, or tie"):
            audit_position_bias(
                [{"pair_id": "p", "first": "a", "second": "b", "winner": "a"}]
            )
        with self.assertRaisesRegex(ValueError, "duplicates a presentation"):
            audit_position_bias([self.records[0], self.records[0]])
        with self.assertRaisesRegex(ValueError, "exactly reverse"):
            audit_position_bias(
                [
                    {"pair_id": "p", "first": "a", "second": "b", "winner": "first"},
                    {"pair_id": "p", "first": "a", "second": "c", "winner": "first"},
                ]
            )
        for value in (True, -0.1, 1.1, float("inf")):
            with self.subTest(rate=value):
                with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                    audit_position_bias(self.records, max_position_flip_rate=value)
        for value in (True, -1, 1.5):
            with self.subTest(count=value):
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    audit_position_bias(self.records, min_complete_pairs=value)


if __name__ == "__main__":
    unittest.main()
