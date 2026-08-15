"""Audit paired preference judgments for presentation-order effects."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_WINNERS = ("first", "second", "tie")
_MAX_INPUT_BYTES = 8 * 1024 * 1024


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


def _validate_count(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _chosen_candidate(first: str, second: str, winner: str) -> str | None:
    if winner == "first":
        return first
    if winner == "second":
        return second
    return None


def audit_position_bias(
    records: Sequence[Mapping[str, Any]],
    *,
    pair_field: str = "pair_id",
    first_field: str = "first",
    second_field: str = "second",
    winner_field: str = "winner",
    min_complete_pairs: int | None = None,
    min_robust_preference_rate: float | None = None,
    max_position_flip_rate: float | None = None,
    max_tie_instability_rate: float | None = None,
    max_incomplete_pairs: int | None = None,
    max_details: int = 20,
) -> dict[str, Any]:
    """Pair reversed presentations and report order-sensitive outcomes."""

    if not records:
        raise ValueError("at least one record is required")
    field_names = {
        "pair_field": pair_field,
        "first_field": first_field,
        "second_field": second_field,
        "winner_field": winner_field,
    }
    for name, value in field_names.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if len(set(field_names.values())) != len(field_names):
        raise ValueError("field names must be distinct")

    thresholds = {
        "complete_pairs": _validate_count("min_complete_pairs", min_complete_pairs),
        "robust_preference_rate": _validate_rate(
            "min_robust_preference_rate", min_robust_preference_rate
        ),
        "position_flip_rate": _validate_rate(
            "max_position_flip_rate", max_position_flip_rate
        ),
        "tie_instability_rate": _validate_rate(
            "max_tie_instability_rate", max_tie_instability_rate
        ),
        "incomplete_pairs": _validate_count(
            "max_incomplete_pairs", max_incomplete_pairs
        ),
    }
    if isinstance(max_details, bool) or not isinstance(max_details, int) or max_details < 0:
        raise ValueError("max_details must be a non-negative integer")

    grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    presentations: set[tuple[str, str, str]] = set()
    for index, record in enumerate(records):
        pair_id = record.get(pair_field)
        first = record.get(first_field)
        second = record.get(second_field)
        winner = record.get(winner_field)
        for field, value in (
            (pair_field, pair_id),
            (first_field, first),
            (second_field, second),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"record {index} {field} must be a non-empty string")
        if first == second:
            raise ValueError(f"record {index} candidates must be distinct")
        if winner not in _WINNERS:
            raise ValueError(
                f"record {index} {winner_field} must be first, second, or tie"
            )
        presentation = (pair_id, first, second)
        if presentation in presentations:
            raise ValueError(f"record {index} duplicates a presentation for {pair_id!r}")
        presentations.add(presentation)
        grouped[pair_id].append((first, second, winner))

    classifications = {
        "robust_preference": 0,
        "stable_tie": 0,
        "first_position_flip": 0,
        "second_position_flip": 0,
        "tie_instability": 0,
    }
    unstable_details: list[dict[str, str]] = []
    incomplete_details: list[dict[str, Any]] = []
    complete_pairs = 0
    for pair_id in sorted(grouped):
        pair_records = grouped[pair_id]
        if len(pair_records) == 1:
            incomplete_details.append({"pair_id": pair_id, "presentations": 1})
            continue
        if len(pair_records) != 2:
            raise ValueError(f"pair {pair_id!r} must contain one or two presentations")
        first_record, second_record = pair_records
        if first_record[:2] != tuple(reversed(second_record[:2])):
            raise ValueError(f"pair {pair_id!r} presentations must exactly reverse candidates")

        complete_pairs += 1
        first_choice = _chosen_candidate(*first_record)
        second_choice = _chosen_candidate(*second_record)
        if first_choice is not None and first_choice == second_choice:
            classification = "robust_preference"
        elif first_choice is None and second_choice is None:
            classification = "stable_tie"
        elif first_choice is None or second_choice is None:
            classification = "tie_instability"
        elif first_record[2] == second_record[2] == "first":
            classification = "first_position_flip"
        else:
            classification = "second_position_flip"
        classifications[classification] += 1
        if classification not in ("robust_preference", "stable_tie"):
            unstable_details.append(
                {"pair_id": pair_id, "classification": classification}
            )

    incomplete_pairs = len(incomplete_details)
    position_flips = (
        classifications["first_position_flip"]
        + classifications["second_position_flip"]
    )
    robust_rate = (
        classifications["robust_preference"] / complete_pairs
        if complete_pairs
        else None
    )
    position_flip_rate = position_flips / complete_pairs if complete_pairs else None
    tie_instability_rate = (
        classifications["tie_instability"] / complete_pairs
        if complete_pairs
        else None
    )

    failures: list[dict[str, Any]] = []
    minimum_pairs = thresholds["complete_pairs"]
    if minimum_pairs is not None and complete_pairs < minimum_pairs:
        failures.append(
            {
                "metric": "complete_pairs",
                "actual": complete_pairs,
                "minimum": minimum_pairs,
                "shortfall": minimum_pairs - complete_pairs,
            }
        )
    minimum_robust = thresholds["robust_preference_rate"]
    if minimum_robust is not None:
        if robust_rate is None:
            failures.append(
                {
                    "metric": "robust_preference_rate",
                    "actual": None,
                    "minimum": minimum_robust,
                    "reason": "no_complete_pairs",
                }
            )
        elif robust_rate < minimum_robust:
            failures.append(
                {
                    "metric": "robust_preference_rate",
                    "actual": robust_rate,
                    "minimum": minimum_robust,
                    "shortfall": minimum_robust - robust_rate,
                }
            )
    for metric, actual in (
        ("position_flip_rate", position_flip_rate),
        ("tie_instability_rate", tie_instability_rate),
    ):
        maximum = thresholds[metric]
        if maximum is None:
            continue
        if actual is None:
            failures.append(
                {
                    "metric": metric,
                    "actual": None,
                    "maximum": maximum,
                    "reason": "no_complete_pairs",
                }
            )
        elif actual > maximum:
            failures.append(
                {
                    "metric": metric,
                    "actual": actual,
                    "maximum": maximum,
                    "excess": actual - maximum,
                }
            )
    maximum_incomplete = thresholds["incomplete_pairs"]
    if maximum_incomplete is not None and incomplete_pairs > maximum_incomplete:
        failures.append(
            {
                "metric": "incomplete_pairs",
                "actual": incomplete_pairs,
                "maximum": maximum_incomplete,
                "excess": incomplete_pairs - maximum_incomplete,
            }
        )

    return {
        "passed": not failures,
        "metrics": {
            "record_count": len(records),
            "pair_count": len(grouped),
            "complete_pairs": complete_pairs,
            "incomplete_pairs": incomplete_pairs,
            **{f"{name}_pairs": count for name, count in classifications.items()},
            "position_flip_pairs": position_flips,
            "robust_preference_rate": robust_rate,
            "stable_tie_rate": (
                classifications["stable_tie"] / complete_pairs
                if complete_pairs
                else None
            ),
            "position_flip_rate": position_flip_rate,
            "tie_instability_rate": tie_instability_rate,
        },
        "thresholds": thresholds,
        "failures": failures,
        "unstable_details": unstable_details[:max_details],
        "incomplete_details": incomplete_details[:max_details],
        "details_truncated": (
            len(unstable_details) > max_details
            or len(incomplete_details) > max_details
        ),
        "settings": {**field_names, "max_details": max_details},
    }


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if path.stat().st_size > _MAX_INPUT_BYTES:
        raise ValueError(f"dataset exceeds {_MAX_INPUT_BYTES} bytes")
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
    parser.add_argument("--pair-field", default="pair_id")
    parser.add_argument("--first-field", default="first")
    parser.add_argument("--second-field", default="second")
    parser.add_argument("--winner-field", default="winner")
    parser.add_argument("--min-complete-pairs", type=int)
    parser.add_argument("--min-robust-preference-rate", type=float)
    parser.add_argument("--max-position-flip-rate", type=float)
    parser.add_argument("--max-tie-instability-rate", type=float)
    parser.add_argument("--max-incomplete-pairs", type=int)
    parser.add_argument("--max-details", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.output is not None and _paths_alias(args.dataset, args.output):
            raise ValueError("output must not alias the source dataset")
        report = audit_position_bias(
            _load_jsonl(args.dataset),
            pair_field=args.pair_field,
            first_field=args.first_field,
            second_field=args.second_field,
            winner_field=args.winner_field,
            min_complete_pairs=args.min_complete_pairs,
            min_robust_preference_rate=args.min_robust_preference_rate,
            max_position_flip_rate=args.max_position_flip_rate,
            max_tie_instability_rate=args.max_tie_instability_rate,
            max_incomplete_pairs=args.max_incomplete_pairs,
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
