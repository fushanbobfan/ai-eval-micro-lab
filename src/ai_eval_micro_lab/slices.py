"""Evaluate text predictions across named dataset slices."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .metrics import evaluate_records


def evaluate_slices(
    records: Sequence[Mapping[str, Any]],
    *,
    slice_by: str,
    min_count: int = 1,
) -> dict[str, Any]:
    """Report overall metrics and deterministic per-slice metrics."""

    if not slice_by:
        raise ValueError("slice_by must be a non-empty field name")
    if min_count < 1:
        raise ValueError("min_count must be positive")

    validated: list[dict[str, str]] = []
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for index, record in enumerate(records):
        expected = record.get("expected")
        predicted = record.get("predicted")
        label = record.get(slice_by)
        if not isinstance(expected, str) or not isinstance(predicted, str):
            raise ValueError(
                f"record {index} must contain string expected and predicted fields"
            )
        if not isinstance(label, str) or not label:
            raise ValueError(
                f"record {index} must contain a non-empty string {slice_by} field"
            )

        pair = {"expected": expected, "predicted": predicted}
        validated.append(pair)
        grouped[label].append(pair)

    included = []
    excluded_slice_count = 0
    excluded_record_count = 0
    for label in sorted(grouped):
        members = grouped[label]
        if len(members) < min_count:
            excluded_slice_count += 1
            excluded_record_count += len(members)
            continue
        included.append({"value": label, **evaluate_records(members)})

    return {
        "slice_by": slice_by,
        "min_count": min_count,
        "overall": evaluate_records(validated),
        "slices": included,
        "excluded": {
            "slices": excluded_slice_count,
            "records": excluded_record_count,
        },
    }


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--slice-by", required=True, help="string field used to group records")
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="omit slices with fewer records while preserving exclusion counts",
    )
    args = parser.parse_args(argv)

    try:
        report = evaluate_slices(
            _load_jsonl(args.dataset),
            slice_by=args.slice_by,
            min_count=args.min_count,
        )
    except (OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
