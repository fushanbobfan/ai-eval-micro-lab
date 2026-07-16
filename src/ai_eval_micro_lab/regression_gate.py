"""Gate paired model comparisons on bootstrap lower confidence bounds."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .comparison import compare_records


def evaluate_regression_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    min_exact_match_difference: float = 0.0,
    min_token_f1_difference: float = 0.0,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Require each paired metric's lower confidence bound to clear a floor."""

    minimums = {
        "exact_match": min_exact_match_difference,
        "token_f1": min_token_f1_difference,
    }
    for name, minimum in minimums.items():
        if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
            raise ValueError(f"{name} minimum difference must be a number")
        if not -1.0 <= minimum <= 1.0:
            raise ValueError(f"{name} minimum difference must be between -1 and 1")

    comparison = compare_records(
        records,
        samples=samples,
        confidence=confidence,
        seed=seed,
    )
    failures = []
    for name, minimum in minimums.items():
        lower_bound = comparison["difference"][name]["lower"]
        if lower_bound < minimum:
            failures.append(
                {
                    "metric": name,
                    "lower_bound": lower_bound,
                    "minimum_difference": minimum,
                    "shortfall": minimum - lower_bound,
                }
            )

    return {
        "passed": not failures,
        "minimum_difference": minimums,
        "comparison": comparison,
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
    parser.add_argument("--min-exact-match-difference", type=float, default=0.0)
    parser.add_argument("--min-token-f1-difference", type=float, default=0.0)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        report = evaluate_regression_gate(
            _load_jsonl(args.dataset),
            min_exact_match_difference=args.min_exact_match_difference,
            min_token_f1_difference=args.min_token_f1_difference,
            samples=args.samples,
            confidence=args.confidence,
            seed=args.seed,
        )
    except (OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
