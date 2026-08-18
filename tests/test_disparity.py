import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from ai_eval_micro_lab.disparity import MAX_INPUT_BYTES, audit_group_disparity, main


class GroupDisparityTests(unittest.TestCase):
    def test_reports_sorted_group_accuracy_and_gap(self):
        records = [
            {"expected": "yes", "predicted": "yes", "group": "alpha"},
            {"expected": "Blue sky", "predicted": "blue-sky", "group": "alpha"},
            {"expected": "yes", "predicted": "yes", "group": "beta"},
            {"expected": "yes", "predicted": "no", "group": "beta"},
            {"expected": "up", "predicted": "down", "group": "beta"},
        ]

        report = audit_group_disparity(records)

        self.assertEqual([group["value"] for group in report["groups"]], ["alpha", "beta"])
        self.assertEqual(report["groups"][0]["correct_count"], 2)
        self.assertEqual(report["groups"][0]["accuracy"], 1.0)
        self.assertAlmostEqual(report["groups"][1]["accuracy"], 1 / 3)
        self.assertEqual(report["groups"][1]["accuracy_interval"]["method"], "wilson")
        self.assertEqual(report["metrics"]["count"], 5)
        self.assertEqual(report["metrics"]["correct_count"], 3)
        self.assertAlmostEqual(report["metrics"]["accuracy"], 0.6)
        self.assertAlmostEqual(report["metrics"]["accuracy_gap"], 2 / 3)
        self.assertEqual(report["metrics"]["best_groups"], ["alpha"])
        self.assertEqual(report["metrics"]["worst_groups"], ["beta"])

    def test_threshold_failures_are_reported_in_stable_order(self):
        records = [
            {"expected": "a", "predicted": "a", "group": "large"},
            {"expected": "b", "predicted": "b", "group": "large"},
            {"expected": "c", "predicted": "x", "group": "small"},
        ]

        report = audit_group_disparity(
            records,
            min_group_count=2,
            min_worst_group_accuracy=0.6,
            max_accuracy_gap=0.2,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            ["minimum_group_count", "worst_group_accuracy", "accuracy_gap"],
        )
        self.assertEqual(report["failures"][0]["groups"], ["small"])
        self.assertEqual(report["thresholds"]["minimum_group_count"], 2)

    def test_invalid_records_fields_and_thresholds_are_rejected(self):
        valid = [{"expected": "a", "predicted": "a", "group": "g"}]

        with self.assertRaisesRegex(ValueError, "at least one"):
            audit_group_disparity([])
        with self.assertRaisesRegex(ValueError, "field names must be distinct"):
            audit_group_disparity(valid, group_field="expected")
        with self.assertRaisesRegex(ValueError, "non-empty string group"):
            audit_group_disparity(
                [{"expected": "a", "predicted": "a", "group": ""}]
            )
        with self.assertRaisesRegex(ValueError, "predicted must be a string"):
            audit_group_disparity(
                [{"expected": "a", "predicted": None, "group": "g"}]
            )
        for confidence in (True, 0.0, 1.0, float("inf")):
            with self.subTest(confidence=confidence):
                with self.assertRaisesRegex(ValueError, "confidence"):
                    audit_group_disparity(valid, confidence=confidence)
        for minimum in (True, 0, 1.5):
            with self.subTest(min_group_count=minimum):
                with self.assertRaisesRegex(ValueError, "min_group_count"):
                    audit_group_disparity(valid, min_group_count=minimum)
        for threshold_name in ("min_worst_group_accuracy", "max_accuracy_gap"):
            with self.subTest(threshold=threshold_name):
                with self.assertRaisesRegex(ValueError, threshold_name):
                    audit_group_disparity(valid, **{threshold_name: 1.1})

    def test_cli_writes_a_report_and_returns_one_for_failed_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "groups.jsonl"
            output = Path(directory) / "report.json"
            dataset.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in (
                        {"answer": "a", "response": "a", "cohort": "x"},
                        {"answer": "b", "response": "x", "cohort": "y"},
                    )
                ),
                encoding="utf-8",
            )

            exit_code = main(
                [
                    str(dataset),
                    "--group-field",
                    "cohort",
                    "--expected-field",
                    "answer",
                    "--predicted-field",
                    "response",
                    "--min-worst-group-accuracy",
                    "0.5",
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertEqual(report["settings"]["group_field"], "cohort")

    def test_cli_rejects_an_oversized_input(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "large.jsonl"
            dataset.write_bytes(b" " * (MAX_INPUT_BYTES + 1))

            with contextlib.redirect_stderr(io.StringIO()):
                exit_code = main([str(dataset)])

            self.assertEqual(exit_code, 2)

    def test_cli_refuses_to_overwrite_the_source_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "groups.jsonl"
            dataset.write_text(
                json.dumps({"expected": "a", "predicted": "a", "group": "g"})
                + "\n",
                encoding="utf-8",
            )
            original = dataset.read_text(encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                exit_code = main([str(dataset), "--output", str(dataset)])

            self.assertEqual(exit_code, 2)
            self.assertEqual(dataset.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
