"""Apply deterministic quality thresholds to text evaluation records."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .metrics import evaluate_records


def evaluate_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    min_exact_match: float = 0.0,
    min_token_f1: float = 0.0,
) -> dict[str, Any]:
    """Evaluate records and report whether every metric threshold is met."""

    thresholds = {
        "exact_match": min_exact_match,
        "token_f1": min_token_f1,
    }
    for name, minimum in thresholds.items():
        if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
            raise ValueError(f"{name} threshold must be a number")
        if not 0.0 <= minimum <= 1.0:
            raise ValueError(f"{name} threshold must be between 0 and 1")
    if not records:
        raise ValueError("at least one record is required")

    validated = []
    for index, record in enumerate(records):
        expected = record.get("expected")
        predicted = record.get("predicted")
        if not isinstance(expected, str) or not isinstance(predicted, str):
            raise ValueError(
                f"record {index} must contain string expected and predicted fields"
            )
        validated.append({"expected": expected, "predicted": predicted})

    metrics = evaluate_records(validated)
    failures = []
    for name, minimum in thresholds.items():
        actual = metrics[name]
        if actual < minimum:
            failures.append(
                {
                    "metric": name,
                    "actual": actual,
                    "minimum": minimum,
                    "shortfall": minimum - actual,
                }
            )

    return {
        "passed": not failures,
        "metrics": metrics,
        "thresholds": thresholds,
        "failures": failures,
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
    parser.add_argument("--min-exact-match", type=float, default=0.0)
    parser.add_argument("--min-token-f1", type=float, default=0.0)
    args = parser.parse_args(argv)

    try:
        report = evaluate_gate(
            _load_jsonl(args.dataset),
            min_exact_match=args.min_exact_match,
            min_token_f1=args.min_token_f1,
        )
    except (OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
