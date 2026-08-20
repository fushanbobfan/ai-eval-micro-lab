import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from ai_eval_micro_lab.ranking_drift import audit_ranking_drift, main


class RankingDriftAuditTests(unittest.TestCase):
    def test_reports_top_k_set_overlap_and_shared_rank_displacement(self):
        report = audit_ranking_drift(
            [
                {
                    "query_id": "q1",
                    "baseline": ["a", "b", "c"],
                    "candidate": ["a", "c", "d"],
                },
                {
                    "query_id": "q2",
                    "baseline": ["x", "y"],
                    "candidate": ["y", "x"],
                },
            ],
            cutoffs=[2, 1],
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["query_count"], 2)
        self.assertEqual(report["cutoffs"], [1, 2])
        at_one, at_two = report["metrics_at_cutoff"]
        self.assertEqual(at_one["mean_top_k_jaccard"], 0.5)
        self.assertEqual(at_one["top_item_change_rate"], 0.5)
        self.assertAlmostEqual(at_two["mean_top_k_jaccard"], 2 / 3)
        self.assertAlmostEqual(at_two["mean_shared_rank_displacement"], 2 / 3)
        self.assertEqual(report["query_details"][0]["query_id"], "q1")

    def test_gates_report_stable_failures_at_selected_cutoff(self):
        report = audit_ranking_drift(
            [
                {
                    "query_id": "q",
                    "baseline": ["a", "b"],
                    "candidate": ["b", "c"],
                }
            ],
            cutoffs=[1, 2],
            gate_cutoff=2,
            min_mean_jaccard=0.5,
            max_top_item_change_rate=0.0,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            ["mean_top_k_jaccard", "top_item_change_rate"],
        )
        self.assertEqual(report["thresholds"]["gate_cutoff"], 2)

    def test_empty_rankings_are_explicitly_stable(self):
        report = audit_ranking_drift(
            [{"query_id": "q", "baseline": [], "candidate": []}],
            cutoffs=[3],
        )

        metrics = report["metrics_at_cutoff"][0]
        self.assertEqual(metrics["mean_top_k_jaccard"], 1.0)
        self.assertEqual(metrics["top_item_change_rate"], 0.0)
        self.assertIsNone(metrics["mean_shared_rank_displacement"])

    def test_custom_fields_and_bounded_details_are_supported(self):
        report = audit_ranking_drift(
            [
                {"case": "a", "old": ["x"], "new": ["y"]},
                {"case": "b", "old": ["x"], "new": ["z"]},
            ],
            cutoffs=[1],
            query_field="case",
            baseline_field="old",
            candidate_field="new",
            max_details=1,
        )

        self.assertEqual(len(report["query_details"]), 1)
        self.assertTrue(report["details_truncated"])
        self.assertEqual(report["omitted_query_count"], 1)

    def test_invalid_records_and_configuration_are_rejected(self):
        valid = [{"query_id": "q", "baseline": ["a"], "candidate": ["a"]}]
        cases = [
            ([], {}, "at least one"),
            (valid, {"cutoffs": []}, "cutoffs"),
            (valid, {"cutoffs": [1, 1]}, "duplicates"),
            (valid, {"gate_cutoff": 2, "cutoffs": [1]}, "gate_cutoff"),
            (valid, {"min_mean_jaccard": -0.1}, "min_mean_jaccard"),
            (valid, {"max_details": -1}, "max_details"),
            (
                [{"query_id": "q", "baseline": ["a", "a"], "candidate": []}],
                {},
                "duplicates",
            ),
            (
                [{"query_id": "q", "baseline": "a", "candidate": []}],
                {},
                "must be a list",
            ),
        ]
        for records, kwargs, message in cases:
            with self.subTest(records=records, kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    audit_ranking_drift(records, **kwargs)

        with self.assertRaisesRegex(ValueError, "repeats query"):
            audit_ranking_drift(valid + valid)

    def test_cli_returns_one_for_gate_failure_and_writes_output(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "rankings.jsonl"
            output = Path(directory) / "report.json"
            dataset.write_text(
                json.dumps(
                    {"query_id": "q", "baseline": ["a"], "candidate": ["b"]}
                )
                + "\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    str(dataset),
                    "--cutoff",
                    "1",
                    "--min-mean-jaccard",
                    "0.5",
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 1)
            self.assertFalse(json.loads(output.read_text(encoding="utf-8"))["passed"])

    def test_cli_rejects_output_alias_and_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "rankings.jsonl"
            dataset.write_text("{\n", encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([str(dataset)]), 2)
                self.assertEqual(main([str(dataset), "--output", str(dataset)]), 2)


if __name__ == "__main__":
    unittest.main()
