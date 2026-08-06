import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import ai_eval_micro_lab
from ai_eval_micro_lab.pairwise import evaluate_pairwise_preferences, main


class PairwisePreferenceTests(unittest.TestCase):
    def setUp(self):
        outcomes = ["candidate"] * 6 + ["baseline"] * 2 + ["tie"] * 2
        self.records = [
            {"id": f"case-{index}", "winner": outcome}
            for index, outcome in enumerate(outcomes, start=1)
        ]

    def test_report_contains_decisive_rates_and_wilson_interval(self):
        report = evaluate_pairwise_preferences(self.records, max_details=1)

        self.assertIs(
            ai_eval_micro_lab.evaluate_pairwise_preferences,
            evaluate_pairwise_preferences,
        )
        self.assertEqual(report["metrics"]["candidate_wins"], 6)
        self.assertEqual(report["metrics"]["baseline_wins"], 2)
        self.assertEqual(report["metrics"]["ties"], 2)
        self.assertEqual(report["metrics"]["decisive_count"], 8)
        self.assertAlmostEqual(report["metrics"]["candidate_win_rate"], 0.75)
        self.assertAlmostEqual(report["metrics"]["tie_rate"], 0.2)
        interval = report["metrics"]["candidate_win_rate_interval"]
        self.assertEqual(interval["method"], "wilson")
        self.assertLess(interval["lower"], 0.75)
        self.assertGreater(interval["upper"], 0.75)
        self.assertEqual(
            report["non_candidate_details"],
            [{"id": "case-7", "outcome": "baseline"}],
        )
        self.assertTrue(report["details_truncated"])

    def test_gates_report_failures_in_stable_order(self):
        report = evaluate_pairwise_preferences(
            self.records,
            min_candidate_win_rate=0.8,
            min_candidate_win_rate_lower_bound=0.5,
            max_tie_rate=0.1,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            [
                "candidate_win_rate",
                "candidate_win_rate_lower_bound",
                "tie_rate",
            ],
        )

    def test_all_ties_make_decisive_metrics_explicitly_undefined(self):
        records = [
            {"id": "a", "winner": "tie"},
            {"id": "b", "winner": "tie"},
        ]

        report = evaluate_pairwise_preferences(records)
        gated = evaluate_pairwise_preferences(
            records, min_candidate_win_rate_lower_bound=0.0
        )

        self.assertIsNone(report["metrics"]["candidate_win_rate"])
        self.assertIsNone(report["metrics"]["candidate_win_rate_interval"])
        self.assertTrue(report["passed"])
        self.assertEqual(
            gated["failures"][0]["reason"], "no_decisive_comparisons"
        )

    def test_invalid_records_fields_and_configuration_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            evaluate_pairwise_preferences([])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate_pairwise_preferences(
                [
                    {"id": "same", "winner": "candidate"},
                    {"id": "same", "winner": "baseline"},
                ]
            )
        with self.assertRaisesRegex(ValueError, "baseline, candidate, or tie"):
            evaluate_pairwise_preferences([{"id": "a", "winner": "other"}])
        with self.assertRaisesRegex(ValueError, "distinct"):
            evaluate_pairwise_preferences(self.records, outcome_field="id")
        for value in (True, 0.0, 1.0, float("inf")):
            with self.subTest(confidence=value):
                with self.assertRaisesRegex(ValueError, "confidence"):
                    evaluate_pairwise_preferences(self.records, confidence=value)
        for value in (True, -1, 1.1):
            with self.subTest(rate=value):
                with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                    evaluate_pairwise_preferences(
                        self.records, max_tie_rate=value
                    )

    def test_cli_supports_custom_fields_and_writes_a_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "preferences.jsonl"
            output = root / "report.json"
            dataset.write_text(
                "".join(
                    json.dumps({"case": item["id"], "choice": item["winner"]})
                    + "\n"
                    for item in self.records
                ),
                encoding="utf-8",
            )

            exit_code = main(
                [
                    str(dataset),
                    "--id-field",
                    "case",
                    "--outcome-field",
                    "choice",
                    "--min-candidate-win-rate",
                    "0.7",
                    "--max-tie-rate",
                    "0.2",
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(json.loads(output.read_text(encoding="utf-8"))["passed"])

    def test_cli_returns_one_for_gate_failure_and_two_for_bad_input(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "preferences.jsonl"
            dataset.write_text(
                "".join(json.dumps(item) + "\n" for item in self.records),
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main([str(dataset), "--min-candidate-win-rate", "0.9"]),
                    1,
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
