import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import ai_eval_micro_lab
from ai_eval_micro_lab.retrieval import evaluate_retrieval, main


class RetrievalEvaluationTests(unittest.TestCase):
    def test_retrieval_api_is_available_from_package(self):
        self.assertIs(ai_eval_micro_lab.evaluate_retrieval, evaluate_retrieval)

    def test_reports_hit_mrr_recall_and_ndcg_at_sorted_cutoffs(self):
        report = evaluate_retrieval(
            [
                {
                    "query_id": "q1",
                    "relevant": ["a", "c"],
                    "retrieved": ["x", "a", "c"],
                },
                {
                    "query_id": "q2",
                    "relevant": ["z"],
                    "retrieved": ["z", "y"],
                },
            ],
            cutoffs=[3, 1],
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["cutoffs"], [1, 3])
        self.assertEqual(report["query_count"], 2)
        at_one, at_three = report["metrics_at_cutoff"]
        self.assertEqual(at_one["hit_rate"], 0.5)
        self.assertEqual(at_one["mean_reciprocal_rank"], 0.5)
        self.assertEqual(at_one["mean_recall"], 0.5)
        self.assertEqual(at_one["mean_ndcg"], 0.5)
        self.assertEqual(at_three["hit_rate"], 1.0)
        self.assertEqual(at_three["mean_reciprocal_rank"], 0.75)
        self.assertEqual(at_three["mean_recall"], 1.0)
        q1_ndcg = (1 / math.log2(3) + 1 / math.log2(4)) / (
            1 + 1 / math.log2(3)
        )
        self.assertAlmostEqual(at_three["mean_ndcg"], (q1_ndcg + 1) / 2)
        self.assertEqual(report["queries"][0]["first_relevant_rank"], 2)

    def test_threshold_failures_are_reported_in_stable_order(self):
        report = evaluate_retrieval(
            [
                {
                    "query_id": "q",
                    "relevant": ["a", "b"],
                    "retrieved": ["a"],
                }
            ],
            cutoffs=[1],
            min_hit_rate=1.0,
            min_mrr=1.0,
            min_recall=0.75,
            min_ndcg=1.0,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            ["mean_recall"],
        )
        self.assertEqual(report["failures"][0]["shortfall"], 0.25)

    def test_empty_rankings_produce_explicit_zero_metrics(self):
        report = evaluate_retrieval(
            [{"query_id": "q", "relevant": ["a"], "retrieved": []}],
            cutoffs=[5],
        )

        self.assertEqual(report["queries"][0]["first_relevant_rank"], None)
        self.assertEqual(report["metrics_at_cutoff"][0]["mean_ndcg"], 0.0)

    def test_custom_field_names_are_supported(self):
        report = evaluate_retrieval(
            [{"case": "q", "gold": ["a"], "results": ["a"]}],
            cutoffs=[1],
            query_field="case",
            relevant_field="gold",
            retrieved_field="results",
        )

        self.assertEqual(report["metrics_at_cutoff"][0]["hit_rate"], 1.0)

    def test_invalid_records_and_configuration_are_rejected(self):
        valid = [{"query_id": "q", "relevant": ["a"], "retrieved": ["a"]}]
        cases = [
            ([], {}, "at least one"),
            (valid, {"cutoffs": []}, "cutoffs"),
            (valid, {"cutoffs": [1, 1]}, "duplicates"),
            (valid, {"cutoffs": [1], "gate_cutoff": 2}, "gate_cutoff"),
            (valid, {"min_ndcg": 1.1}, "min_ndcg"),
            (
                [{"query_id": "q", "relevant": [], "retrieved": []}],
                {},
                "must not be empty",
            ),
            (
                [{"query_id": "q", "relevant": ["a"], "retrieved": ["a", "a"]}],
                {},
                "duplicates",
            ),
        ]
        for records, kwargs, message in cases:
            with self.subTest(records=records, kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_retrieval(records, **kwargs)

        with self.assertRaisesRegex(ValueError, "repeats query"):
            evaluate_retrieval(valid + valid)

    def test_cli_returns_one_with_structured_gate_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "retrieval.jsonl"
            dataset.write_text(
                json.dumps(
                    {"query_id": "q", "relevant": ["a"], "retrieved": ["x"]}
                )
                + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        str(dataset),
                        "--cutoff",
                        "1",
                        "--min-hit-rate",
                        "0.5",
                    ]
                )

            report = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertEqual(report["failures"][0]["metric"], "hit_rate")

    def test_cli_returns_two_for_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "retrieval.jsonl"
            dataset.write_text("{\n", encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                exit_code = main([str(dataset)])

            self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
