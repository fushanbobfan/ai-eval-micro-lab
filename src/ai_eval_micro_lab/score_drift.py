"""Audit paired numeric evaluator scores for bounded drift."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _finite_number(value: Any, *, field: str, record_index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"record {record_index} field {field!r} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"record {record_index} field {field!r} must be finite")
    return normalized


def _non_negative(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-negative finite number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return normalized


def _rate(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be between 0 and 1")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0 <= normalized <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return normalized


def audit_score_drift(
    records: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = 0.0,
    max_abs_mean_shift: float | None = None,
    max_mean_absolute_change: float | None = None,
    min_within_tolerance_rate: float = 0.0,
    max_details: int = 20,
    id_field: str = "case_id",
    baseline_field: str = "baseline_score",
    candidate_field: str = "candidate_score",
) -> dict[str, Any]:
    """Return deterministic paired score-drift metrics and gate failures."""
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("records must be a sequence")
    if not records:
        raise ValueError("at least one record is required")
    field_names = (id_field, baseline_field, candidate_field)
    if any(not isinstance(field, str) or not field for field in field_names):
        raise ValueError("field names must be non-empty strings")
    if len(set(field_names)) != len(field_names):
        raise ValueError("field names must be distinct")
    if isinstance(max_details, bool) or not isinstance(max_details, int) or max_details < 0:
        raise ValueError("max_details must be a non-negative integer")

    normalized_tolerance = _non_negative("tolerance", tolerance)
    assert normalized_tolerance is not None
    maximum_mean_shift = _non_negative("max_abs_mean_shift", max_abs_mean_shift)
    maximum_absolute_change = _non_negative(
        "max_mean_absolute_change", max_mean_absolute_change
    )
    minimum_within_rate = _rate(
        "min_within_tolerance_rate", min_within_tolerance_rate
    )

    seen_ids: set[str] = set()
    details = []
    baseline_total = 0.0
    candidate_total = 0.0
    delta_total = 0.0
    absolute_total = 0.0
    squared_total = 0.0
    within_count = 0
    increase_count = 0
    decrease_count = 0

    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {index} must be an object")
        case_id = record.get(id_field)
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(
                f"record {index} field {id_field!r} must be a non-empty string"
            )
        if case_id in seen_ids:
            raise ValueError(f"record {index} repeats case id {case_id!r}")
        seen_ids.add(case_id)
        baseline = _finite_number(
            record.get(baseline_field), field=baseline_field, record_index=index
        )
        candidate = _finite_number(
            record.get(candidate_field), field=candidate_field, record_index=index
        )
        delta = candidate - baseline
        absolute_change = abs(delta)
        within_tolerance = absolute_change <= normalized_tolerance
        within_count += int(within_tolerance)
        increase_count += int(delta > normalized_tolerance)
        decrease_count += int(delta < -normalized_tolerance)
        baseline_total += baseline
        candidate_total += candidate
        delta_total += delta
        absolute_total += absolute_change
        squared_total += delta * delta
        details.append(
            {
                "case_id": case_id,
                "baseline_score": baseline,
                "candidate_score": candidate,
                "delta": delta,
                "absolute_change": absolute_change,
                "within_tolerance": within_tolerance,
            }
        )

    case_count = len(details)
    mean_shift = delta_total / case_count
    mean_absolute_change = absolute_total / case_count
    within_tolerance_rate = within_count / case_count
    failures = []
    if maximum_mean_shift is not None and abs(mean_shift) > maximum_mean_shift:
        failures.append(
            {
                "metric": "abs_mean_shift",
                "actual": abs(mean_shift),
                "maximum": maximum_mean_shift,
                "excess": abs(mean_shift) - maximum_mean_shift,
            }
        )
    if (
        maximum_absolute_change is not None
        and mean_absolute_change > maximum_absolute_change
    ):
        failures.append(
            {
                "metric": "mean_absolute_change",
                "actual": mean_absolute_change,
                "maximum": maximum_absolute_change,
                "excess": mean_absolute_change - maximum_absolute_change,
            }
        )
    if within_tolerance_rate < minimum_within_rate:
        failures.append(
            {
                "metric": "within_tolerance_rate",
                "actual": within_tolerance_rate,
                "minimum": minimum_within_rate,
                "shortfall": minimum_within_rate - within_tolerance_rate,
            }
        )

    details.sort(key=lambda item: (-item["absolute_change"], item["case_id"]))
    return {
        "passed": not failures,
        "case_count": case_count,
        "baseline_mean": baseline_total / case_count,
        "candidate_mean": candidate_total / case_count,
        "mean_shift": mean_shift,
        "mean_absolute_change": mean_absolute_change,
        "root_mean_square_change": math.sqrt(squared_total / case_count),
        "maximum_absolute_change": details[0]["absolute_change"],
        "maximum_change_case_id": details[0]["case_id"],
        "within_tolerance_count": within_count,
        "within_tolerance_rate": within_tolerance_rate,
        "increase_beyond_tolerance_count": increase_count,
        "decrease_beyond_tolerance_count": decrease_count,
        "thresholds": {
            "tolerance": normalized_tolerance,
            "max_abs_mean_shift": maximum_mean_shift,
            "max_mean_absolute_change": maximum_absolute_change,
            "min_within_tolerance_rate": minimum_within_rate,
        },
        "failures": failures,
        "case_details": details[:max_details],
        "details_truncated": case_count > max_details,
        "omitted_case_count": max(0, case_count - max_details),
        "settings": {
            "id_field": id_field,
            "baseline_field": baseline_field,
            "candidate_field": candidate_field,
            "max_details": max_details,
        },
    }


def _load_jsonl(
    path: Path, *, max_file_bytes: int = 10 * 1024 * 1024
) -> list[Mapping[str, Any]]:
    if (
        isinstance(max_file_bytes, bool)
        or not isinstance(max_file_bytes, int)
        or max_file_bytes <= 0
    ):
        raise ValueError("max_file_bytes must be a positive integer")
    with path.open("rb") as handle:
        data = handle.read(max_file_bytes + 1)
    if len(data) > max_file_bytes:
        raise ValueError(f"dataset exceeds max_file_bytes ({max_file_bytes})")

    records = []
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
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
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--max-abs-mean-shift", type=float)
    parser.add_argument("--max-mean-absolute-change", type=float)
    parser.add_argument("--min-within-tolerance-rate", type=float, default=0.0)
    parser.add_argument("--max-details", type=int, default=20)
    parser.add_argument("--id-field", default="case_id")
    parser.add_argument("--baseline-field", default="baseline_score")
    parser.add_argument("--candidate-field", default="candidate_score")
    parser.add_argument("--max-file-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.output is not None and _paths_alias(args.dataset, args.output):
            raise ValueError("output must not alias the input dataset")
        report = audit_score_drift(
            _load_jsonl(args.dataset, max_file_bytes=args.max_file_bytes),
            tolerance=args.tolerance,
            max_abs_mean_shift=args.max_abs_mean_shift,
            max_mean_absolute_change=args.max_mean_absolute_change,
            min_within_tolerance_rate=args.min_within_tolerance_rate,
            max_details=args.max_details,
            id_field=args.id_field,
            baseline_field=args.baseline_field,
            candidate_field=args.candidate_field,
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
