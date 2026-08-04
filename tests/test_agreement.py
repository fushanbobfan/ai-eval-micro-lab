import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import ai_eval_micro_lab
from ai_eval_micro_lab.agreement import evaluate_agreement, main


class AgreementTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"id": "a", "reference": "cat", "rater": "cat"},
            {"id": "b", "reference": "cat", "rater": "dog"},
            {"id": "c", "reference": "dog", "rater": "dog"},
            {"id": "d", "reference": "dog", "rater": "cat"},
            {"id": "e", "reference": "bird", "rater": "bird"},
            {"id": "f", "reference": "cat", "rater": "cat"},
        ]

    def test_report_contains_deterministic_agreement_metrics(self):
        report = evaluate_agreement(self.records, max_details=1)

        self.assertIs(ai_eval_micro_lab.evaluate_agreement, evaluate_agreement)
        self.assertEqual(report["labels"], ["bird", "cat", "dog"])
        self.assertEqual(
            report["confusion_matrix"],
            [[1, 0, 0], [0, 2, 1], [0, 1, 1]],
        )
        self.assertEqual(report["metrics"]["agreement_count"], 4)
        self.assertAlmostEqual(report["metrics"]["observed_agreement"], 2 / 3)
        self.assertAlmostEqual(report["metrics"]["expected_agreement"], 7 / 18)
        self.assertAlmostEqual(report["metrics"]["cohen_kappa"], 5 / 11)
        self.assertEqual(report["metrics"]["kappa_status"], "defined")
        self.assertEqual(
            report["disagreement_pairs"],
            [
                {"reference": "cat", "rater": "dog", "count": 1},
                {"reference": "dog", "rater": "cat", "count": 1},
            ],
        )
        self.assertEqual(
            report["disagreement_details"],
            [{"id": "b", "reference": "cat", "rater": "dog"}],
        )
        self.assertTrue(report["details_truncated"])

    def test_threshold_failures_are_reported_in_stable_order(self):
        report = evaluate_agreement(
            self.records,
            min_agreement=0.8,
            min_kappa=0.5,
            max_disagreements=1,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            ["observed_agreement", "cohen_kappa", "disagreement_count"],
        )

    def test_degenerate_single_label_kappa_is_explicitly_undefined(self):
        records = [
            {"id": "a", "reference": "same", "rater": "same"},
            {"id": "b", "reference": "same", "rater": "same"},
        ]

        report = evaluate_agreement(records)
        gated = evaluate_agreement(records, min_kappa=0.0)

        self.assertIsNone(report["metrics"]["cohen_kappa"])
        self.assertTrue(report["passed"])
        self.assertFalse(gated["passed"])
        self.assertEqual(
            gated["failures"][0]["reason"],
            "undefined_expected_agreement_one",
        )

    def test_invalid_records_fields_and_configuration_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            evaluate_agreement([])
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            evaluate_agreement([{"id": "", "reference": "a", "rater": "a"}])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate_agreement(
                [
                    {"id": "a", "reference": "a", "rater": "a"},
                    {"id": "a", "reference": "a", "rater": "a"},
                ]
            )
        with self.assertRaisesRegex(ValueError, "distinct"):
            evaluate_agreement(self.records, reference_field="id")
        for value in (True, -1, 1.5):
            with self.subTest(max_details=value):
                with self.assertRaisesRegex(ValueError, "max_details"):
                    evaluate_agreement(self.records, max_details=value)
        for value in (True, -0.1, 1.1, float("inf")):
            with self.subTest(min_agreement=value):
                with self.assertRaisesRegex(ValueError, "threshold"):
                    evaluate_agreement(self.records, min_agreement=value)

    def test_cli_writes_a_report_and_returns_one_for_a_failed_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "agreement.jsonl"
            output = Path(directory) / "report.json"
            dataset.write_text(
                "".join(json.dumps(record) + "\n" for record in self.records),
                encoding="utf-8",
            )

            exit_code = main(
                [
                    str(dataset),
                    "--min-kappa",
                    "0.5",
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertEqual(report["failures"][0]["metric"], "cohen_kappa")

    def test_cli_supports_custom_fields_and_a_passing_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "agreement.jsonl"
            dataset.write_text(
                json.dumps({"case": "a", "human": "yes", "judge": "yes"})
                + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        str(dataset),
                        "--id-field",
                        "case",
                        "--reference-field",
                        "human",
                        "--rater-field",
                        "judge",
                        "--min-agreement",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(json.loads(stdout.getvalue())["passed"])

    def test_cli_rejects_invalid_json_and_output_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "agreement.jsonl"
            dataset.write_text("{\n", encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([str(dataset)]), 2)

            dataset.write_text(
                json.dumps({"id": "a", "reference": "yes", "rater": "yes"})
                + "\n",
                encoding="utf-8",
            )
            original = dataset.read_text(encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main([str(dataset), "--output", str(dataset)]),
                    2,
                )
            self.assertEqual(dataset.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
