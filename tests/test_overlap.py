import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import ai_eval_micro_lab
from ai_eval_micro_lab.overlap import (
    audit_dataset_overlap,
    main,
    normalize_overlap_text,
)


class DatasetOverlapTests(unittest.TestCase):
    def test_overlap_api_is_available_from_package(self):
        self.assertIs(ai_eval_micro_lab.audit_dataset_overlap, audit_dataset_overlap)

    def test_reports_exact_and_normalized_matches_without_text(self):
        report = audit_dataset_overlap(
            [
                {"id": "r1", "text": "Model evaluation"},
                {"id": "r2", "text": "Full-width Ａ text"},
                {"id": "r3", "text": "reference only"},
            ],
            [
                {"id": "c1", "text": "Model evaluation"},
                {"id": "c2", "text": "  full-width a\tTEXT  "},
                {"id": "c3", "text": "candidate only"},
            ],
            max_overlap_rate=0.75,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["overlap"]["matching_group_count"], 2)
        self.assertEqual(report["overlap"]["overlapping_reference_count"], 2)
        self.assertEqual(report["overlap"]["overlapping_candidate_count"], 2)
        self.assertEqual(report["overlap"]["candidate_overlap_rate"], 2 / 3)
        self.assertEqual(report["overlap"]["exact_pair_count"], 1)
        self.assertEqual(report["overlap"]["normalized_only_pair_count"], 1)
        self.assertEqual(
            {match["match_type"] for match in report["overlap"]["matches"]},
            {"exact", "normalized"},
        )
        self.assertNotIn("text", str(report["overlap"]["matches"]))

    def test_counts_cross_product_pairs_and_bounds_details(self):
        report = audit_dataset_overlap(
            [
                {"id": "r1", "text": "same"},
                {"id": "r2", "text": "SAME"},
            ],
            [
                {"id": "c1", "text": "same"},
                {"id": "c2", "text": " same "},
            ],
            max_overlap_rate=1.0,
            max_details=2,
        )

        self.assertEqual(report["overlap"]["pair_count"], 4)
        self.assertEqual(report["overlap"]["overlapping_reference_count"], 2)
        self.assertEqual(report["overlap"]["overlapping_candidate_count"], 2)
        self.assertEqual(len(report["overlap"]["matches"]), 2)
        self.assertTrue(report["overlap"]["details_truncated"])

    def test_threshold_failure_reports_excess(self):
        report = audit_dataset_overlap(
            [{"id": "r", "text": "duplicate"}],
            [
                {"id": "c1", "text": "Duplicate"},
                {"id": "c2", "text": "unique"},
            ],
            max_overlap_rate=0.25,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["failures"][0]["actual"], 0.5)
        self.assertEqual(report["failures"][0]["excess"], 0.25)

    def test_custom_fields_and_unicode_normalization_are_supported(self):
        report = audit_dataset_overlap(
            [{"key": "r", "prompt": "Café"}],
            [{"key": "c", "prompt": "Cafe\u0301"}],
            id_field="key",
            text_field="prompt",
            max_overlap_rate=1.0,
        )

        self.assertEqual(report["overlap"]["normalized_only_pair_count"], 1)
        self.assertEqual(normalize_overlap_text("  A\tＢ  "), "a b")

    def test_invalid_records_and_configuration_are_rejected(self):
        reference = [{"id": "r", "text": "valid"}]
        candidate = [{"id": "c", "text": "valid"}]
        cases = [
            ([], candidate, {}, "reference dataset"),
            (reference, [], {}, "candidate dataset"),
            (reference, candidate, {"max_overlap_rate": 1.1}, "between 0 and 1"),
            (reference, candidate, {"max_details": -1}, "non-negative"),
            (reference, candidate, {"id_field": "text"}, "must be distinct"),
            ([{"id": "r", "text": "   "}], candidate, {}, "non-whitespace"),
            (
                [{"id": "r", "text": "a"}, {"id": "r", "text": "b"}],
                candidate,
                {},
                "repeats id",
            ),
        ]
        for left, right, kwargs, message in cases:
            with self.subTest(kwargs=kwargs, message=message):
                with self.assertRaisesRegex(ValueError, message):
                    audit_dataset_overlap(left, right, **kwargs)

    def test_cli_reports_gate_failure_and_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            reference_path = Path(directory) / "reference.jsonl"
            candidate_path = Path(directory) / "candidate.jsonl"
            reference_path.write_text(
                json.dumps({"id": "r", "text": "shared"}) + "\n",
                encoding="utf-8",
            )
            candidate_path.write_text(
                json.dumps({"id": "c", "text": "SHARED"}) + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main([str(reference_path), str(candidate_path)])

            self.assertEqual(exit_code, 1)
            report = json.loads(stdout.getvalue())
            self.assertEqual(
                report["failures"][0]["metric"], "candidate_overlap_rate"
            )

            candidate_path.write_text("{\n", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main([str(reference_path), str(candidate_path)]),
                    2,
                )

    def test_cli_passes_when_overlap_is_within_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            reference_path = Path(directory) / "reference.jsonl"
            candidate_path = Path(directory) / "candidate.jsonl"
            reference_path.write_text(
                json.dumps({"id": "r", "text": "shared"}) + "\n",
                encoding="utf-8",
            )
            candidate_path.write_text(
                json.dumps({"id": "c", "text": "different"}) + "\n",
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        str(reference_path),
                        str(candidate_path),
                        "--max-overlap-rate",
                        "0",
                    ]
                )

            self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
