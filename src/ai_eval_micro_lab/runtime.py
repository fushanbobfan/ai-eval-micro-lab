"""Audit inference runtime efficiency from recorded model attempts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any


def _optional_non_negative_number(
    name: str, value: float | None
) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _optional_rate(name: str, value: float | None) -> float | None:
    validated = _optional_non_negative_number(name, value)
    if validated is not None and validated > 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return validated


def _optional_count(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nearest_rank(values: Sequence[float | int], percentile: float) -> float | int:
    ordered = sorted(values)
    index = max(1, math.ceil(percentile * len(ordered))) - 1
    return ordered[index]


def audit_runtime(
    records: Sequence[Mapping[str, Any]],
    *,
    id_field: str = "id",
    latency_field: str = "latency_ms",
    input_tokens_field: str = "input_tokens",
    output_tokens_field: str = "output_tokens",
    success_field: str = "success",
    min_success_rate: float | None = None,
    max_p95_latency_ms: float | None = None,
    max_mean_total_tokens: float | None = None,
    max_failures: int | None = None,
    max_details: int = 20,
) -> dict[str, Any]:
    """Return deterministic latency, token-use, success, and gate diagnostics."""

    if not records:
        raise ValueError("at least one runtime record is required")
    fields = {
        "id_field": id_field,
        "latency_field": latency_field,
        "input_tokens_field": input_tokens_field,
        "output_tokens_field": output_tokens_field,
        "success_field": success_field,
    }
    for name, field in fields.items():
        if not isinstance(field, str) or not field:
            raise ValueError(f"{name} must be a non-empty string")
    if len(set(fields.values())) != len(fields):
        raise ValueError("field names must be distinct")
    if isinstance(max_details, bool) or not isinstance(max_details, int) or max_details < 0:
        raise ValueError("max_details must be a non-negative integer")

    thresholds = {
        "success_rate": _optional_rate("min_success_rate", min_success_rate),
        "p95_latency_ms": _optional_non_negative_number(
            "max_p95_latency_ms", max_p95_latency_ms
        ),
        "mean_total_tokens": _optional_non_negative_number(
            "max_mean_total_tokens", max_mean_total_tokens
        ),
        "failure_count": _optional_count("max_failures", max_failures),
    }

    attempts = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {index} must be an object")
        case_id = record.get(id_field)
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"record {index} {id_field} must be a non-empty string")
        if case_id in seen_ids:
            raise ValueError(f"record {index} has duplicate {id_field} {case_id!r}")
        seen_ids.add(case_id)

        latency = record.get(latency_field)
        if (
            isinstance(latency, bool)
            or not isinstance(latency, Real)
            or not math.isfinite(latency)
            or latency < 0
        ):
            raise ValueError(
                f"record {index} {latency_field} must be finite and non-negative"
            )
        token_values = []
        for field in (input_tokens_field, output_tokens_field):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"record {index} {field} must be a non-negative integer"
                )
            token_values.append(value)
        success = record.get(success_field)
        if not isinstance(success, bool):
            raise ValueError(f"record {index} {success_field} must be a boolean")
        input_tokens, output_tokens = token_values
        attempts.append(
            {
                "id": case_id,
                "latency_ms": float(latency),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "success": success,
            }
        )

    count = len(attempts)
    success_count = sum(attempt["success"] for attempt in attempts)
    failure_count = count - success_count
    latencies = [attempt["latency_ms"] for attempt in attempts]
    total_tokens = [attempt["total_tokens"] for attempt in attempts]
    metrics = {
        "count": count,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": success_count / count,
        "latency_ms": {
            "mean": sum(latencies) / count,
            "p50": _nearest_rank(latencies, 0.50),
            "p95": _nearest_rank(latencies, 0.95),
            "maximum": max(latencies),
        },
        "tokens": {
            "input_total": sum(attempt["input_tokens"] for attempt in attempts),
            "output_total": sum(attempt["output_tokens"] for attempt in attempts),
            "total": sum(total_tokens),
            "mean_total_per_attempt": sum(total_tokens) / count,
            "p95_total_per_attempt": _nearest_rank(total_tokens, 0.95),
        },
    }

    failures = []
    minimum_success = thresholds["success_rate"]
    if minimum_success is not None and metrics["success_rate"] < minimum_success:
        failures.append(
            {
                "metric": "success_rate",
                "actual": metrics["success_rate"],
                "minimum": minimum_success,
                "shortfall": minimum_success - metrics["success_rate"],
            }
        )
    maximum_latency = thresholds["p95_latency_ms"]
    if maximum_latency is not None and metrics["latency_ms"]["p95"] > maximum_latency:
        failures.append(
            {
                "metric": "p95_latency_ms",
                "actual": metrics["latency_ms"]["p95"],
                "maximum": maximum_latency,
                "excess": metrics["latency_ms"]["p95"] - maximum_latency,
            }
        )
    maximum_tokens = thresholds["mean_total_tokens"]
    if maximum_tokens is not None and metrics["tokens"]["mean_total_per_attempt"] > maximum_tokens:
        failures.append(
            {
                "metric": "mean_total_tokens",
                "actual": metrics["tokens"]["mean_total_per_attempt"],
                "maximum": maximum_tokens,
                "excess": metrics["tokens"]["mean_total_per_attempt"] - maximum_tokens,
            }
        )
    maximum_failures = thresholds["failure_count"]
    if maximum_failures is not None and failure_count > maximum_failures:
        failures.append(
            {
                "metric": "failure_count",
                "actual": failure_count,
                "maximum": maximum_failures,
                "excess": failure_count - maximum_failures,
            }
        )

    slowest = sorted(attempts, key=lambda item: (-item["latency_ms"], item["id"]))
    failure_ids = sorted(
        attempt["id"] for attempt in attempts if not attempt["success"]
    )
    return {
        "passed": not failures,
        "metrics": metrics,
        "thresholds": thresholds,
        "failures": failures,
        "slowest_attempts": [
            {
                "id": attempt["id"],
                "latency_ms": attempt["latency_ms"],
                "success": attempt["success"],
                "total_tokens": attempt["total_tokens"],
            }
            for attempt in slowest[:max_details]
        ],
        "failure_ids": failure_ids[:max_details],
        "details_truncated": {
            "slowest_attempts": len(slowest) > max_details,
            "failure_ids": len(failure_ids) > max_details,
        },
        "settings": {
            **fields,
            "percentile_method": "nearest_rank",
            "latency_population": "all_attempts",
            "max_details": max_details,
        },
    }


def _load_jsonl(path: Path, *, max_file_bytes: int) -> list[Mapping[str, Any]]:
    maximum_bytes = _positive_count("max_file_bytes", max_file_bytes)
    if path.stat().st_size > maximum_bytes:
        raise ValueError(f"dataset exceeds max_file_bytes ({maximum_bytes})")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            records.append(record)
    return records


def _paths_alias(source: Path, output: Path) -> bool:
    if source.resolve() == output.resolve():
        return True
    try:
        return source.samefile(output)
    except (FileNotFoundError, OSError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--latency-field", default="latency_ms")
    parser.add_argument("--input-tokens-field", default="input_tokens")
    parser.add_argument("--output-tokens-field", default="output_tokens")
    parser.add_argument("--success-field", default="success")
    parser.add_argument("--min-success-rate", type=float)
    parser.add_argument("--max-p95-latency-ms", type=float)
    parser.add_argument("--max-mean-total-tokens", type=float)
    parser.add_argument("--max-failures", type=int)
    parser.add_argument("--max-details", type=int, default=20)
    parser.add_argument("--max-file-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.output is not None and _paths_alias(args.dataset, args.output):
            raise ValueError("output must not alias the source dataset")
        report = audit_runtime(
            _load_jsonl(args.dataset, max_file_bytes=args.max_file_bytes),
            id_field=args.id_field,
            latency_field=args.latency_field,
            input_tokens_field=args.input_tokens_field,
            output_tokens_field=args.output_tokens_field,
            success_field=args.success_field,
            min_success_rate=args.min_success_rate,
            max_p95_latency_ms=args.max_p95_latency_ms,
            max_mean_total_tokens=args.max_mean_total_tokens,
            max_failures=args.max_failures,
            max_details=args.max_details,
        )
        rendered = json.dumps(report, indent=2) + "\n"
        if args.output is None:
            print(rendered, end="")
        else:
            args.output.write_text(rendered, encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
