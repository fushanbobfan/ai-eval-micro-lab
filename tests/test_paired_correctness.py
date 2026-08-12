import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import ai_eval_micro_lab
from ai_eval_micro_lab.paired_correctness import evaluate_paired_correctness, main


class PairedCorrectnessTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"id": "q1", "expected": "Paris", "baseline": "Paris", "candidate": "Paris"},
            {"id": "q2", "expected": "4", "baseline": "5", "candidate": "4"},
            {"id": "q3", "expected": "blue whale", "baseline": "blue-whale", "candidate": "orca"},
            {"id": "q4", "expected": "Mercury", "baseline": "Venus", "candidate": "Mars"},
            {"id": "q5", "expected": "yes", "baseline": "no", "candidate": "YES"},
        ]

    def test_public_api_and_transition_metrics(self):
        self.assertIs(
            ai_eval_micro_lab.evaluate_paired_correctness,
            evaluate_paired_correctness,
        )
        report = evaluate_paired_correctness(self.records)

        self.assertTrue(report["passed"])
        self.assertEqual(report["metrics"]["baseline_correct"], 2)
        self.assertEqual(report["metrics"]["candidate_correct"], 3)
        self.assertAlmostEqual(report["metrics"]["accuracy_difference"], 0.2)
        self.assertEqual(
            report["metrics"]["transitions"],
            {
                "both_correct": 1,
                "both_incorrect": 1,
                "baseline_only_correct": 1,
                "candidate_only_correct": 2,
            },
        )
        self.assertEqual(report["metrics"]["discordant_count"], 3)
        self.assertAlmostEqual(
            report["metrics"]["candidate_discordant_win_rate"], 2 / 3
        )
        self.assertAlmostEqual(report["metrics"]["mcnemar_exact_p_value"], 1.0)

    def test_exact_mcnemar_value_for_one_sided_discordance(self):
        records = [
            {
                "id": f"case-{index}",
                "expected": "right",
                "baseline": "wrong",
                "candidate": "right",
            }
            for index in range(6)
        ]
        report = evaluate_paired_correctness(records, max_exact_p_value=0.05)

        self.assertTrue(report["passed"])
        self.assertEqual(report["metrics"]["mcnemar_exact_p_value"], 0.03125)
        interval = report["metrics"]["candidate_discordant_win_rate_interval"]
        self.assertEqual(interval["method"], "wilson")
        self.assertEqual(interval["upper"], 1.0)

    def test_p_value_gate_requires_candidate_direction(self):
        records = [
            {
                "id": f"case-{index}",
                "expected": "right",
                "baseline": "right",
                "candidate": "wrong",
            }
            for index in range(6)
        ]
        report = evaluate_paired_correctness(records, max_exact_p_value=0.05)

        self.assertFalse(report["passed"])
        self.assertEqual(
            report["failures"][0]["reason"],
            "candidate_not_better_on_discordant_cases",
        )

    def test_threshold_failures_are_reported_together(self):
        report = evaluate_paired_correctness(
            self.records,
            min_accuracy_difference=0.3,
            max_regressions=0,
            max_exact_p_value=0.05,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            ["accuracy_difference", "regressions", "mcnemar_exact_p_value"],
        )

    def test_identical_correctness_has_no_discordant_interval(self):
        records = [
            {"id": "same", "expected": "A", "baseline": "a", "candidate": "A"}
        ]
        report = evaluate_paired_correctness(records)

        self.assertEqual(report["metrics"]["discordant_count"], 0)
        self.assertIsNone(report["metrics"]["candidate_discordant_win_rate"])
        self.assertIsNone(
            report["metrics"]["candidate_discordant_win_rate_interval"]
        )
        self.assertEqual(report["metrics"]["mcnemar_exact_p_value"], 1.0)

    def test_details_are_id_only_and_bounded(self):
        report = evaluate_paired_correctness(self.records, max_details=1)

        self.assertEqual(
            report["discordant_details"],
            [{"id": "q2", "transition": "candidate_only_correct"}],
        )
        self.assertTrue(report["details_truncated"])
        self.assertNotIn("expected", str(report["discordant_details"]))

    def test_custom_fields_are_supported(self):
        report = evaluate_paired_correctness(
            [{"key": "x", "truth": "A", "old": "B", "new": "a"}],
            id_field="key",
            expected_field="truth",
            baseline_field="old",
            candidate_field="new",
        )

        self.assertEqual(
            report["metrics"]["transitions"]["candidate_only_correct"], 1
        )

    def test_invalid_inputs_are_rejected(self):
        invalid_calls = [
            ([], {}),
            ([{"id": "x", "expected": "a", "baseline": "a", "candidate": "a"}], {"id_field": "expected"}),
            ([{"id": "x", "expected": "a", "baseline": "a", "candidate": "a"}], {"confidence": 1.0}),
            ([{"id": "x", "expected": "a", "baseline": "a", "candidate": "a"}], {"max_details": -1}),
            ([{"id": "x", "expected": "a", "baseline": "a", "candidate": "a"}], {"min_accuracy_difference": 2.0}),
            ([{"id": "x", "expected": "a", "baseline": "a", "candidate": "a"}], {"max_regressions": True}),
        ]
        for records, kwargs in invalid_calls:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluate_paired_correctness(records, **kwargs)

        duplicate = [
            {"id": "x", "expected": "a", "baseline": "a", "candidate": "a"},
            {"id": "x", "expected": "b", "baseline": "b", "candidate": "b"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate_paired_correctness(duplicate)

    def test_cli_writes_report_and_uses_exit_codes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset = Path(temporary_directory) / "pairs.jsonl"
            output = Path(temporary_directory) / "report.json"
            dataset.write_text(
                "\n".join(json.dumps(record) for record in self.records) + "\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    str(dataset),
                    "--min-accuracy-difference",
                    "0.1",
                    "--max-regressions",
                    "1",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertAlmostEqual(
                json.loads(output.read_text(encoding="utf-8"))["metrics"][
                    "accuracy_difference"
                ],
                0.2,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main([str(dataset), "--max-regressions", "0"]), 1
                )

    def test_cli_rejects_invalid_json_and_output_alias(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset = Path(temporary_directory) / "pairs.jsonl"
            dataset.write_text("not json\n", encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(main([str(dataset)]), 2)
            self.assertIn("invalid JSON", stderr.getvalue())

            dataset.write_text(
                json.dumps(self.records[0]) + "\n", encoding="utf-8"
            )
            original = dataset.read_text(encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main([str(dataset), "--output", str(dataset)]), 2
                )
            self.assertEqual(dataset.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
