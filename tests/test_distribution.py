import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from ai_eval_micro_lab.distribution import audit_label_distribution, main


class LabelDistributionTests(unittest.TestCase):
    def test_reports_distribution_distances_and_new_labels(self):
        reference = [
            {"label": "a"},
            {"label": "a"},
            {"label": "a"},
            {"label": "b"},
        ]
        candidate = [
            {"label": "a"},
            {"label": "b"},
            {"label": "c"},
            {"label": "c"},
        ]

        report = audit_label_distribution(reference, candidate)

        self.assertTrue(report["passed"])
        self.assertEqual(report["metrics"]["reference_count"], 4)
        self.assertEqual(report["metrics"]["candidate_count"], 4)
        self.assertEqual(report["metrics"]["candidate_only_label_count"], 1)
        self.assertAlmostEqual(report["metrics"]["total_variation"], 0.5)
        self.assertAlmostEqual(
            report["metrics"]["jensen_shannon_divergence_bits"],
            0.3443609377704336,
        )
        self.assertEqual(
            [item["label"] for item in report["label_shifts"]], ["a", "c", "b"]
        )

    def test_equal_prevalence_with_different_counts_has_zero_drift(self):
        report = audit_label_distribution(
            [{"label": "yes"}, {"label": "no"}],
            [
                {"label": "yes"},
                {"label": "yes"},
                {"label": "no"},
                {"label": "no"},
            ],
        )

        self.assertEqual(report["metrics"]["total_variation"], 0.0)
        self.assertEqual(report["metrics"]["jensen_shannon_divergence_bits"], 0.0)

    def test_reports_every_threshold_failure(self):
        report = audit_label_distribution(
            [{"label": "a"}, {"label": "a"}],
            [{"label": "b"}, {"label": "b"}],
            max_total_variation=0.2,
            max_js_divergence=0.2,
            max_label_delta=0.2,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            {failure["metric"] for failure in report["failures"]},
            {
                "total_variation",
                "jensen_shannon_divergence_bits",
                "max_absolute_prevalence_delta",
            },
        )

    def test_bounds_details_after_sorting_by_absolute_shift(self):
        report = audit_label_distribution(
            [{"label": "a"}, {"label": "a"}, {"label": "b"}],
            [{"label": "b"}, {"label": "c"}, {"label": "c"}],
            max_details=1,
        )

        self.assertEqual(len(report["label_shifts"]), 1)
        self.assertEqual(report["label_shifts"][0]["label"], "a")
        self.assertTrue(report["details_truncated"])
        self.assertEqual(report["omitted_label_count"], 2)

    def test_rejects_invalid_inputs_and_thresholds(self):
        invalid_calls = [
            lambda: audit_label_distribution([], [{"label": "a"}]),
            lambda: audit_label_distribution([{"label": "a"}], []),
            lambda: audit_label_distribution([{"label": ""}], [{"label": "a"}]),
            lambda: audit_label_distribution(
                [{"label": "a"}], [{"label": "a"}], max_total_variation=math.nan
            ),
            lambda: audit_label_distribution(
                [{"label": "a"}], [{"label": "a"}], max_details=True
            ),
        ]

        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()

    def test_cli_returns_zero_or_one_for_valid_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            candidate = root / "candidate.jsonl"
            reference.write_text('{"class":"a"}\n{"class":"a"}\n', encoding="utf-8")
            candidate.write_text('{"class":"a"}\n{"class":"b"}\n', encoding="utf-8")
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        str(reference),
                        str(candidate),
                        "--label-field",
                        "class",
                        "--max-total-variation",
                        "0.6",
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertTrue(json.loads(stdout.getvalue())["passed"])

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        str(reference),
                        str(candidate),
                        "--label-field",
                        "class",
                        "--max-total-variation",
                        "0.2",
                    ]
                )
            self.assertEqual(exit_code, 1)

    def test_cli_returns_two_for_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            candidate = root / "candidate.jsonl"
            reference.write_text("{\n", encoding="utf-8")
            candidate.write_text('{"label":"a"}\n', encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                exit_code = main([str(reference), str(candidate)])

            self.assertEqual(exit_code, 2)

    def test_cli_refuses_to_overwrite_either_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            candidate = root / "candidate.jsonl"
            content = '{"label":"a"}\n'
            reference.write_text(content, encoding="utf-8")
            candidate.write_text(content, encoding="utf-8")

            for output in (reference, candidate):
                with self.subTest(output=output), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    exit_code = main(
                        [str(reference), str(candidate), "--output", str(output)]
                    )
                self.assertEqual(exit_code, 2)
                self.assertEqual(output.read_text(encoding="utf-8"), content)


if __name__ == "__main__":
    unittest.main()
