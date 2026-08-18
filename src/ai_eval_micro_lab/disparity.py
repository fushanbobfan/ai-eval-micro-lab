"""Audit exact-match performance differences across named groups."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

from .metrics import exact_match

MAX_INPUT_BYTES = 10 * 1024 * 1024


def _validate_rate(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be between 0 and 1")
    return float(value)


def _wilson_interval(successes: int, trials: int, confidence: float) -> dict[str, Any]:
    proportion = successes / trials
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )
    return {
        "method": "wilson",
        "confidence": confidence,
        "lower": max(0.0, center - half_width),
        "upper": min(1.0, center + half_width),
    }


def audit_group_disparity(
    records: Sequence[Mapping[str, Any]],
    *,
    group_field: str = "group",
    expected_field: str = "expected",
    predicted_field: str = "predicted",
    confidence: float = 0.95,
    min_group_count: int = 1,
    min_worst_group_accuracy: float | None = None,
    max_accuracy_gap: float | None = None,
) -> dict[str, Any]:
    """Report deterministic exact-match accuracy differences between groups."""

    if not records:
        raise ValueError("at least one record is required")
    field_names = {
        "group_field": group_field,
        "expected_field": expected_field,
        "predicted_field": predicted_field,
    }
    for name, value in field_names.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if len(set(field_names.values())) != len(field_names):
        raise ValueError("field names must be distinct")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("confidence must be strictly between 0 and 1")
    if (
        isinstance(min_group_count, bool)
        or not isinstance(min_group_count, int)
        or min_group_count < 1
    ):
        raise ValueError("min_group_count must be a positive integer")
    min_worst_group_accuracy = _validate_rate(
        "min_worst_group_accuracy", min_worst_group_accuracy
    )
    max_accuracy_gap = _validate_rate("max_accuracy_gap", max_accuracy_gap)

    grouped: dict[str, list[bool]] = defaultdict(list)
    for index, record in enumerate(records):
        group = record.get(group_field)
        expected = record.get(expected_field)
        predicted = record.get(predicted_field)
        if not isinstance(group, str) or not group:
            raise ValueError(
                f"record {index} {group_field} must be a non-empty string group"
            )
        if not isinstance(expected, str):
            raise ValueError(f"record {index} {expected_field} must be a string")
        if not isinstance(predicted, str):
            raise ValueError(f"record {index} {predicted_field} must be a string")
        correct = bool(exact_match(expected, predicted))
        grouped[group].append(correct)

    groups = []
    for value in sorted(grouped):
        outcomes = grouped[value]
        correct_count = sum(outcomes)
        groups.append(
            {
                "value": value,
                "count": len(outcomes),
                "correct_count": correct_count,
                "accuracy": correct_count / len(outcomes),
                "accuracy_interval": _wilson_interval(
                    correct_count, len(outcomes), confidence
                ),
            }
        )

    accuracies = [group["accuracy"] for group in groups]
    best_accuracy = max(accuracies)
    worst_accuracy = min(accuracies)
    minimum_group_count = min(group["count"] for group in groups)
    count = sum(group["count"] for group in groups)
    correct_count = sum(group["correct_count"] for group in groups)
    accuracy_gap = best_accuracy - worst_accuracy
    failures = []
    undersized_groups = [
        group["value"] for group in groups if group["count"] < min_group_count
    ]
    if undersized_groups:
        failures.append(
            {
                "metric": "minimum_group_count",
                "actual": minimum_group_count,
                "minimum": min_group_count,
                "groups": undersized_groups,
            }
        )
    if (
        min_worst_group_accuracy is not None
        and worst_accuracy < min_worst_group_accuracy
    ):
        failures.append(
            {
                "metric": "worst_group_accuracy",
                "actual": worst_accuracy,
                "minimum": min_worst_group_accuracy,
                "shortfall": min_worst_group_accuracy - worst_accuracy,
                "groups": [
                    group["value"]
                    for group in groups
                    if group["accuracy"] == worst_accuracy
                ],
            }
        )
    if max_accuracy_gap is not None and accuracy_gap > max_accuracy_gap:
        failures.append(
            {
                "metric": "accuracy_gap",
                "actual": accuracy_gap,
                "maximum": max_accuracy_gap,
                "excess": accuracy_gap - max_accuracy_gap,
            }
        )
    return {
        "passed": not failures,
        "groups": groups,
        "metrics": {
            "count": count,
            "correct_count": correct_count,
            "accuracy": correct_count / count,
            "best_group_accuracy": best_accuracy,
            "worst_group_accuracy": worst_accuracy,
            "accuracy_gap": accuracy_gap,
            "minimum_group_count": minimum_group_count,
            "best_groups": [
                group["value"] for group in groups if group["accuracy"] == best_accuracy
            ],
            "worst_groups": [
                group["value"] for group in groups if group["accuracy"] == worst_accuracy
            ],
        },
        "thresholds": {
            "minimum_group_count": min_group_count,
            "worst_group_accuracy": min_worst_group_accuracy,
            "accuracy_gap": max_accuracy_gap,
        },
        "failures": failures,
        "settings": {
            **field_names,
            "confidence": float(confidence),
        },
    }


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError(f"dataset exceeds {MAX_INPUT_BYTES} bytes")
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
    parser.add_argument("--group-field", default="group")
    parser.add_argument("--expected-field", default="expected")
    parser.add_argument("--predicted-field", default="predicted")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--min-group-count", type=int, default=1)
    parser.add_argument("--min-worst-group-accuracy", type=float)
    parser.add_argument("--max-accuracy-gap", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.output is not None and _paths_alias(args.dataset, args.output):
            raise ValueError("output must not alias the source dataset")
        report = audit_group_disparity(
            _load_jsonl(args.dataset),
            group_field=args.group_field,
            expected_field=args.expected_field,
            predicted_field=args.predicted_field,
            confidence=args.confidence,
            min_group_count=args.min_group_count,
            min_worst_group_accuracy=args.min_worst_group_accuracy,
            max_accuracy_gap=args.max_accuracy_gap,
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
