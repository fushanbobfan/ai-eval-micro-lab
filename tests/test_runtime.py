import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import ai_eval_micro_lab
from ai_eval_micro_lab.runtime import audit_runtime, main


class RuntimeAuditTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "id": "case-a",
                "latency_ms": 100.0,
                "input_tokens": 10,
                "output_tokens": 5,
                "success": True,
            },
            {
                "id": "case-b",
                "latency_ms": 200.0,
                "input_tokens": 20,
                "output_tokens": 10,
                "success": False,
            },
            {
                "id": "case-c",
                "latency_ms": 150.0,
                "input_tokens": 15,
                "output_tokens": 5,
                "success": True,
            },
            {
                "id": "case-d",
                "latency_ms": 50.0,
                "input_tokens": 5,
                "output_tokens": 5,
                "success": True,
            },
        ]

    def test_public_api_and_runtime_metrics(self):
        self.assertIs(ai_eval_micro_lab.audit_runtime, audit_runtime)

        report = audit_runtime(self.records)

        self.assertTrue(report["passed"])
        self.assertEqual(report["metrics"]["count"], 4)
        self.assertEqual(report["metrics"]["success_count"], 3)
        self.assertEqual(report["metrics"]["failure_count"], 1)
        self.assertEqual(report["metrics"]["success_rate"], 0.75)
        self.assertEqual(report["metrics"]["latency_ms"]["mean"], 125.0)
        self.assertEqual(report["metrics"]["latency_ms"]["p50"], 100.0)
        self.assertEqual(report["metrics"]["latency_ms"]["p95"], 200.0)
        self.assertEqual(report["metrics"]["latency_ms"]["maximum"], 200.0)
        self.assertEqual(report["metrics"]["tokens"]["input_total"], 50)
        self.assertEqual(report["metrics"]["tokens"]["output_total"], 25)
        self.assertEqual(report["metrics"]["tokens"]["total"], 75)
        self.assertEqual(
            report["metrics"]["tokens"]["mean_total_per_attempt"], 18.75
        )
        self.assertEqual(
            report["metrics"]["tokens"]["p95_total_per_attempt"], 30
        )

    def test_threshold_failures_are_reported_together(self):
        report = audit_runtime(
            self.records,
            min_success_rate=0.8,
            max_p95_latency_ms=180.0,
            max_mean_total_tokens=18.0,
            max_failures=0,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            [failure["metric"] for failure in report["failures"]],
            [
                "success_rate",
                "p95_latency_ms",
                "mean_total_tokens",
                "failure_count",
            ],
        )

    def test_details_are_bounded_and_content_free(self):
        report = audit_runtime(self.records, max_details=1)

        self.assertEqual(
            report["slowest_attempts"],
            [
                {
                    "id": "case-b",
                    "latency_ms": 200.0,
                    "success": False,
                    "total_tokens": 30,
                }
            ],
        )
        self.assertEqual(report["failure_ids"], ["case-b"])
        self.assertEqual(
            report["details_truncated"],
            {"slowest_attempts": True, "failure_ids": False},
        )
        self.assertNotIn("prompt", json.dumps(report))

    def test_custom_fields_are_supported(self):
        report = audit_runtime(
            [{"key": "x", "ms": 2, "in": 1, "out": 3, "ok": True}],
            id_field="key",
            latency_field="ms",
            input_tokens_field="in",
            output_tokens_field="out",
            success_field="ok",
        )

        self.assertEqual(report["metrics"]["tokens"]["total"], 4)

    def test_invalid_inputs_are_rejected(self):
        valid = self.records[:1]
        invalid_calls = [
            ([], {}),
            (valid, {"latency_field": "id"}),
            (valid, {"min_success_rate": 1.1}),
            (valid, {"max_p95_latency_ms": -1}),
            (valid, {"max_failures": True}),
            (valid, {"max_details": -1}),
        ]
        for records, kwargs in invalid_calls:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                audit_runtime(records, **kwargs)

        duplicate = [self.records[0], dict(self.records[0])]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit_runtime(duplicate)

        invalid_records = [
            {**self.records[0], "latency_ms": -1},
            {**self.records[0], "latency_ms": True},
            {**self.records[0], "input_tokens": 1.5},
            {**self.records[0], "output_tokens": -1},
            {**self.records[0], "success": 1},
        ]
        for record in invalid_records:
            with self.subTest(record=record), self.assertRaises(ValueError):
                audit_runtime([record])

    def test_cli_writes_report_and_uses_gate_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "runtime.jsonl"
            output = Path(directory) / "report.json"
            dataset.write_text(
                "\n".join(json.dumps(record) for record in self.records) + "\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    str(dataset),
                    "--min-success-rate",
                    "0.8",
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 1)
            self.assertFalse(json.loads(output.read_text(encoding="utf-8"))["passed"])

    def test_cli_rejects_aliases_and_oversized_input(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "runtime.jsonl"
            dataset.write_text(
                json.dumps(self.records[0]) + "\n",
                encoding="utf-8",
            )
            with contextlib.redirect_stderr(io.StringIO()):
                alias_exit = main([str(dataset), "--output", str(dataset)])
                size_exit = main([str(dataset), "--max-file-bytes", "4"])

        self.assertEqual(alias_exit, 2)
        self.assertEqual(size_exit, 2)


if __name__ == "__main__":
    unittest.main()
