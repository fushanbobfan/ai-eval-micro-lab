import unittest

from ai_eval_micro_lab.slices import evaluate_slices


class SliceEvaluationTests(unittest.TestCase):
    def test_reports_overall_and_sorted_slice_metrics(self):
        records = [
            {"expected": "yes", "predicted": "no", "topic": "zeta"},
            {"expected": "blue sky", "predicted": "blue sky", "topic": "alpha"},
            {"expected": "warm day", "predicted": "warm", "topic": "alpha"},
        ]

        report = evaluate_slices(records, slice_by="topic")

        self.assertEqual(report["overall"]["count"], 3)
        self.assertEqual(
            [item["value"] for item in report["slices"]], ["alpha", "zeta"]
        )
        self.assertEqual(report["slices"][0]["count"], 2)
        self.assertEqual(report["slices"][0]["exact_match"], 0.5)
        self.assertEqual(report["excluded"], {"slices": 0, "records": 0})

    def test_min_count_preserves_exclusion_counts(self):
        records = [
            {"expected": "a", "predicted": "a", "topic": "common"},
            {"expected": "b", "predicted": "b", "topic": "common"},
            {"expected": "c", "predicted": "c", "topic": "rare"},
        ]

        report = evaluate_slices(records, slice_by="topic", min_count=2)

        self.assertEqual([item["value"] for item in report["slices"]], ["common"])
        self.assertEqual(report["excluded"], {"slices": 1, "records": 1})
        self.assertEqual(report["overall"]["count"], 3)

    def test_invalid_arguments_and_records_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "slice_by"):
            evaluate_slices([], slice_by="")
        with self.assertRaisesRegex(ValueError, "min_count"):
            evaluate_slices([], slice_by="topic", min_count=0)
        with self.assertRaisesRegex(ValueError, "string topic"):
            evaluate_slices(
                [{"expected": "a", "predicted": "a", "topic": 1}],
                slice_by="topic",
            )
        with self.assertRaisesRegex(ValueError, "string expected"):
            evaluate_slices(
                [{"expected": "a", "topic": "x"}],
                slice_by="topic",
            )


if __name__ == "__main__":
    unittest.main()
