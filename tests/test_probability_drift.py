import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import ai_eval_micro_lab
from ai_eval_micro_lab.probability_drift import audit_probability_drift, main


class ProbabilityDriftTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "case_id": "a",
                "baseline_scores": {"cat": 0.8, "dog": 0.2},
                "candidate_scores": {"cat": 0.6, "dog": 0.4},
            },
            {
                "case_id": "b",
                "baseline_scores": {"cat": 0.4, "dog": 0.6},
                "candidate_scores": {"cat": 0.7, "dog": 0.3},
            },
        ]

    def test_reports_paired_probability_drift(self):
        report = audit_probability_drift(self.records)

        self.assertIs(ai_eval_micro_lab.audit_probability_drift, audit_probability_drift)
        self.assertTrue(report["passed"])
        self.assertEqual(report["labels"], ["cat", "dog"])
        self.assertAlmostEqual(report["metrics"]["mean_total_variation"], 0.25)
        self.assertAlmostEqual(report["metrics"]["maximum_total_variation"], 0.3)
        self.assertEqual(report["metrics"]["maximum_total_variation_case_id"], "b")
        self.assertEqual(report["metrics"]["top_label_change_count"], 1)
        self.assertAlmostEqual(report["metrics"]["top_label_change_rate"], 0.5)
        self.assertAlmostEqual(
            report["per_label"][0]["mean_probability_shift"], 0.05
        )
        self.assertAlmostEqual(
            report["per_label"][0]["mean_absolute_probability_change"], 0.25
        )

    def test_jensen_shannon_divergence_uses_bits_and_handles_zeros(self):
        report = audit_probability_drift(
            [
                {
                    "case_id": "opposite",
                    "baseline_scores": {"no": 0.0, "yes": 1.0},
                    "candidate_scores": {"no": 1.0, "yes": 0.0},
                }
            ]
        )

        self.assertAlmostEqual(report["metrics"]["mean_js_divergence_bits"], 1.0)
        self.assertAlmostEqual(report["metrics"]["maximum_total_variation"], 1.0)

    def test_threshold_failures_have_stable_order(self):
        report = audit_probability_drift(
            self.records,
            max_mean_total_variation=0.1,
            max_case_total_variation=0.2,
            max_mean_js_divergence=0.01,
            max_top_label_change_rate=0.4,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            [
                "mean_total_variation",
                "case_total_variation",
                "mean_js_divergence",
                "top_label_change_rate",
            ],
        )

    def test_details_are_bounded_and_sorted_by_divergence(self):
        records = self.records + [
            {
                "case_id": "c",
                "baseline_scores": {"cat": 0.55, "dog": 0.45},
                "candidate_scores": {"cat": 0.5, "dog": 0.5},
            }
        ]

        report = audit_probability_drift(records, max_details=1)

        self.assertEqual(report["case_details"][0]["case_id"], "b")
        self.assertTrue(report["details_truncated"])
        self.assertEqual(report["omitted_case_count"], 2)

    def test_equal_top_probabilities_use_label_order(self):
        report = audit_probability_drift(
            [
                {
                    "case_id": "tie",
                    "baseline_scores": {"zeta": 0.5, "alpha": 0.5},
                    "candidate_scores": {"zeta": 0.4, "alpha": 0.6},
                }
            ]
        )

        self.assertEqual(report["case_details"][0]["baseline_top_label"], "alpha")
        self.assertFalse(report["case_details"][0]["top_label_changed"])

    def test_custom_fields_are_supported(self):
        report = audit_probability_drift(
            [{"id": "x", "old": {"a": 1}, "new": {"a": 1}}],
            id_field="id",
            baseline_field="old",
            candidate_field="new",
        )

        self.assertEqual(report["case_count"], 1)
        self.assertEqual(report["settings"]["candidate_field"], "new")

    def test_invalid_records_and_configuration_are_rejected(self):
        cases = [
            ([], {}, "at least one"),
            (self.records, {"max_mean_total_variation": math.inf}, "between 0 and 1"),
            (self.records, {"probability_tolerance": 0}, "greater than 0"),
            (self.records, {"max_details": True}, "max_details"),
            (self.records, {"id_field": "baseline_scores"}, "distinct"),
            (
                [
                    {
                        "case_id": "a",
                        "baseline_scores": {"cat": 0.7, "dog": 0.4},
                        "candidate_scores": {"cat": 0.5, "dog": 0.5},
                    }
                ],
                {},
                "sum to 1",
            ),
            (
                [
                    {
                        "case_id": "a",
                        "baseline_scores": {"cat": True, "dog": 0.0},
                        "candidate_scores": {"cat": 0.5, "dog": 0.5},
                    }
                ],
                {},
                "between 0 and 1",
            ),
        ]
        for records, kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    audit_probability_drift(records, **kwargs)

        with self.assertRaisesRegex(ValueError, "same labels"):
            audit_probability_drift(
                [
                    {
                        "case_id": "a",
                        "baseline_scores": {"cat": 1.0},
                        "candidate_scores": {"dog": 1.0},
                    }
                ]
            )
        with self.assertRaisesRegex(ValueError, "repeats case"):
            audit_probability_drift(self.records + [self.records[0]])

    def test_cli_writes_a_report_and_returns_one_for_a_failed_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "probability-drift.jsonl"
            output = Path(directory) / "report.json"
            dataset.write_text(
                "".join(json.dumps(record) + "\n" for record in self.records),
                encoding="utf-8",
            )

            exit_code = main(
                [
                    str(dataset),
                    "--max-top-label-change-rate",
                    "0.25",
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 1)
            self.assertFalse(json.loads(output.read_text(encoding="utf-8"))["passed"])

    def test_cli_supports_custom_fields_and_a_passing_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "probability-drift.jsonl"
            dataset.write_text(
                json.dumps({"id": "x", "old": {"a": 1}, "new": {"a": 1}})
                + "\n",
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                exit_code = main(
                    [
                        str(dataset),
                        "--id-field",
                        "id",
                        "--baseline-field",
                        "old",
                        "--candidate-field",
                        "new",
                        "--max-mean-total-variation",
                        "0",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(json.loads(stdout.getvalue())["passed"])

    def test_cli_rejects_invalid_json_output_alias_and_oversized_input(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "probability-drift.jsonl"
            dataset.write_text("{\n", encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([str(dataset)]), 2)
                self.assertEqual(main([str(dataset), "--output", str(dataset)]), 2)
                self.assertEqual(main([str(dataset), "--max-file-bytes", "1"]), 2)


if __name__ == "__main__":
    unittest.main()
