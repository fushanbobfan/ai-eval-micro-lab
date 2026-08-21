import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import ai_eval_micro_lab
from ai_eval_micro_lab.score_drift import audit_score_drift, main


class ScoreDriftAuditTests(unittest.TestCase):
    def test_score_drift_api_is_available_from_package(self):
        self.assertIs(ai_eval_micro_lab.audit_score_drift, audit_score_drift)

    def test_reports_paired_score_drift_metrics(self):
        report = audit_score_drift(
            [
                {"case_id": "a", "baseline_score": 1.0, "candidate_score": 1.2},
                {"case_id": "b", "baseline_score": 3.0, "candidate_score": 2.6},
                {"case_id": "c", "baseline_score": 2.0, "candidate_score": 2.0},
            ],
            tolerance=0.2,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["case_count"], 3)
        self.assertAlmostEqual(report["baseline_mean"], 2.0)
        self.assertAlmostEqual(report["candidate_mean"], 1.9333333333333333)
        self.assertAlmostEqual(report["mean_shift"], -0.06666666666666665)
        self.assertAlmostEqual(report["mean_absolute_change"], 0.2)
        self.assertAlmostEqual(
            report["root_mean_square_change"], math.sqrt(0.2 / 3)
        )
        self.assertEqual(report["within_tolerance_count"], 2)
        self.assertEqual(report["increase_beyond_tolerance_count"], 0)
        self.assertEqual(report["decrease_beyond_tolerance_count"], 1)
        self.assertEqual(report["maximum_change_case_id"], "b")

    def test_gate_failures_have_stable_order(self):
        report = audit_score_drift(
            [
                {"case_id": "a", "baseline_score": 0.0, "candidate_score": 1.0},
                {"case_id": "b", "baseline_score": 0.0, "candidate_score": 0.5},
            ],
            tolerance=0.1,
            max_abs_mean_shift=0.2,
            max_mean_absolute_change=0.3,
            min_within_tolerance_rate=0.5,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            ["abs_mean_shift", "mean_absolute_change", "within_tolerance_rate"],
        )

    def test_details_are_bounded_and_sorted_by_absolute_change(self):
        report = audit_score_drift(
            [
                {"case_id": "b", "baseline_score": 0, "candidate_score": 1},
                {"case_id": "a", "baseline_score": 0, "candidate_score": -1},
                {"case_id": "c", "baseline_score": 0, "candidate_score": 0.5},
            ],
            max_details=2,
        )

        self.assertEqual(
            [detail["case_id"] for detail in report["case_details"]], ["a", "b"]
        )
        self.assertTrue(report["details_truncated"])
        self.assertEqual(report["omitted_case_count"], 1)

    def test_custom_fields_are_supported(self):
        report = audit_score_drift(
            [{"id": "x", "before": 4, "after": 4.25}],
            id_field="id",
            baseline_field="before",
            candidate_field="after",
            tolerance=0.25,
        )

        self.assertEqual(report["within_tolerance_rate"], 1.0)
        self.assertEqual(report["settings"]["candidate_field"], "after")

    def test_invalid_records_and_configuration_are_rejected(self):
        valid = [{"case_id": "a", "baseline_score": 1, "candidate_score": 1}]
        cases = [
            ([], {}, "at least one"),
            (valid, {"tolerance": -1}, "tolerance"),
            (valid, {"max_abs_mean_shift": math.inf}, "max_abs_mean_shift"),
            (valid, {"min_within_tolerance_rate": 1.1}, "between 0 and 1"),
            (valid, {"max_details": -1}, "max_details"),
            (valid, {"id_field": "baseline_score"}, "distinct"),
            ([{"case_id": "a", "baseline_score": True, "candidate_score": 1}], {}, "number"),
            ([{"case_id": "a", "baseline_score": 1, "candidate_score": math.nan}], {}, "finite"),
        ]
        for records, kwargs, message in cases:
            with self.subTest(records=records, kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    audit_score_drift(records, **kwargs)

        with self.assertRaisesRegex(ValueError, "repeats case"):
            audit_score_drift(valid + valid)

    def test_cli_writes_report_and_returns_one_for_failed_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "scores.jsonl"
            output = Path(directory) / "report.json"
            dataset.write_text(
                json.dumps(
                    {"case_id": "a", "baseline_score": 0, "candidate_score": 1}
                )
                + "\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    str(dataset),
                    "--max-abs-mean-shift",
                    "0.5",
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 1)
            self.assertFalse(json.loads(output.read_text(encoding="utf-8"))["passed"])

    def test_cli_supports_custom_fields_and_a_passing_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "scores.jsonl"
            dataset.write_text(
                '{"id":"a","old":2,"new":2.1}\n', encoding="utf-8"
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
                        "--tolerance",
                        "0.2",
                        "--min-within-tolerance-rate",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(json.loads(stdout.getvalue())["passed"])

    def test_cli_rejects_invalid_json_output_alias_and_oversized_input(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "scores.jsonl"
            dataset.write_text("{\n", encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([str(dataset)]), 2)
                self.assertEqual(main([str(dataset), "--output", str(dataset)]), 2)
                self.assertEqual(main([str(dataset), "--max-file-bytes", "1"]), 2)


if __name__ == "__main__":
    unittest.main()
