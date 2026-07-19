import unittest

import ai_eval_micro_lab
from ai_eval_micro_lab.consistency import evaluate_consistency


class ConsistencyTests(unittest.TestCase):
    def test_consistency_api_is_available_from_package(self):
        self.assertIs(ai_eval_micro_lab.evaluate_consistency, evaluate_consistency)

    def test_report_uses_pair_weighted_agreement_and_tracks_singletons(self):
        report = evaluate_consistency(
            [
                {"case_id": "a", "predicted": "Blue sky!"},
                {"case_id": "a", "predicted": "blue sky"},
                {"case_id": "a", "predicted": "blue ocean"},
                {"case_id": "b", "predicted": "yes"},
                {"case_id": "b", "predicted": "no"},
                {"case_id": "single", "predicted": "ignored"},
            ]
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["summary"]["record_count"], 6)
        self.assertEqual(report["summary"]["repeated_case_count"], 2)
        self.assertEqual(report["summary"]["singleton_case_count"], 1)
        self.assertEqual(report["summary"]["pair_count"], 4)
        self.assertEqual(report["summary"]["exact_agreement"], 0.25)
        self.assertEqual(report["summary"]["token_f1_agreement"], 0.5)
        self.assertEqual([case["case_id"] for case in report["cases"]], ["a", "b"])
        self.assertEqual(report["cases"][0]["majority_fraction"], 2 / 3)
        self.assertEqual(
            report["cases"][0]["modal_normalized_prediction"], "blue sky"
        )

    def test_threshold_failures_have_stable_order(self):
        report = evaluate_consistency(
            [
                {"case_id": "a", "predicted": "left"},
                {"case_id": "a", "predicted": "right"},
            ],
            min_exact_agreement=0.8,
            min_token_f1_agreement=0.6,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            ["exact_agreement", "token_f1_agreement"],
        )

    def test_case_order_and_modal_ties_are_deterministic(self):
        records = [
            {"case_id": "z", "predicted": "Zulu"},
            {"case_id": "z", "predicted": "alpha"},
            {"case_id": "a", "predicted": "same"},
            {"case_id": "a", "predicted": "same"},
        ]

        forward = evaluate_consistency(records)
        reverse = evaluate_consistency(list(reversed(records)))

        self.assertEqual(forward, reverse)
        self.assertEqual(
            forward["cases"][1]["modal_normalized_prediction"], "alpha"
        )

    def test_invalid_records_and_configuration_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one record"):
            evaluate_consistency([])
        with self.assertRaisesRegex(ValueError, "two or more"):
            evaluate_consistency([{"case_id": "one", "predicted": "value"}])
        for record in (
            {"predicted": "value"},
            {"case_id": "", "predicted": "value"},
            {"case_id": "one"},
            {"case_id": "one", "predicted": 1},
        ):
            with self.subTest(record=record):
                with self.assertRaisesRegex(ValueError, "record 0"):
                    evaluate_consistency([record])
        for name, value in (
            ("case_field", ""),
            ("prediction_field", 1),
            ("min_exact_agreement", -0.1),
            ("min_token_f1_agreement", float("nan")),
        ):
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(ValueError, name):
                    evaluate_consistency(
                        [
                            {"case_id": "a", "predicted": "same"},
                            {"case_id": "a", "predicted": "same"},
                        ],
                        **{name: value},
                    )


if __name__ == "__main__":
    unittest.main()
