"""Measure agreement across repeated model outputs for the same cases."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from .metrics import exact_match, normalize, token_f1


def _validate_unit_interval(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    return float(value)


def _validate_field_name(name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def evaluate_consistency(
    records: Sequence[Mapping[str, Any]],
    *,
    case_field: str = "case_id",
    prediction_field: str = "predicted",
    min_exact_agreement: float = 0.0,
    min_token_f1_agreement: float = 0.0,
) -> dict[str, Any]:
    """Return deterministic pairwise agreement metrics for repeated cases."""

    case_key = _validate_field_name("case_field", case_field)
    prediction_key = _validate_field_name("prediction_field", prediction_field)
    minimum_exact = _validate_unit_interval(
        "min_exact_agreement", min_exact_agreement
    )
    minimum_token_f1 = _validate_unit_interval(
        "min_token_f1_agreement", min_token_f1_agreement
    )
    if not records:
        raise ValueError("at least one record is required")

    grouped: dict[str, list[str]] = defaultdict(list)
    for index, record in enumerate(records):
        case_id = record.get(case_key)
        prediction = record.get(prediction_key)
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(
                f"record {index} field {case_key!r} must be a non-empty string"
            )
        if not isinstance(prediction, str):
            raise ValueError(
                f"record {index} field {prediction_key!r} must be a string"
            )
        grouped[case_id].append(prediction)

    repeated = {case_id: values for case_id, values in grouped.items() if len(values) > 1}
    if not repeated:
        raise ValueError("at least one case must have two or more predictions")

    case_reports = []
    exact_total = 0.0
    token_f1_total = 0.0
    pair_total = 0
    for case_id in sorted(repeated):
        predictions = repeated[case_id]
        normalized_counts = Counter(normalize(value) for value in predictions)
        majority_count = max(normalized_counts.values())
        modal_prediction = min(
            value
            for value, count in normalized_counts.items()
            if count == majority_count
        )

        pairs = list(combinations(predictions, 2))
        exact_scores = [exact_match(left, right) for left, right in pairs]
        token_f1_scores = [token_f1(left, right) for left, right in pairs]
        exact_agreement = sum(exact_scores) / len(pairs)
        token_f1_agreement = sum(token_f1_scores) / len(pairs)
        exact_total += sum(exact_scores)
        token_f1_total += sum(token_f1_scores)
        pair_total += len(pairs)

        case_reports.append(
            {
                "case_id": case_id,
                "prediction_count": len(predictions),
                "pair_count": len(pairs),
                "unique_normalized_predictions": len(normalized_counts),
                "modal_normalized_prediction": modal_prediction,
                "majority_fraction": majority_count / len(predictions),
                "exact_agreement": exact_agreement,
                "token_f1_agreement": token_f1_agreement,
            }
        )

    exact_agreement = exact_total / pair_total
    token_f1_agreement = token_f1_total / pair_total
    failures = []
    if exact_agreement < minimum_exact:
        failures.append(
            {
                "metric": "exact_agreement",
                "actual": exact_agreement,
                "minimum": minimum_exact,
                "shortfall": minimum_exact - exact_agreement,
            }
        )
    if token_f1_agreement < minimum_token_f1:
        failures.append(
            {
                "metric": "token_f1_agreement",
                "actual": token_f1_agreement,
                "minimum": minimum_token_f1,
                "shortfall": minimum_token_f1 - token_f1_agreement,
            }
        )

    return {
        "passed": not failures,
        "summary": {
            "record_count": len(records),
            "case_count": len(grouped),
            "repeated_case_count": len(repeated),
            "singleton_case_count": len(grouped) - len(repeated),
            "pair_count": pair_total,
            "exact_agreement": exact_agreement,
            "token_f1_agreement": token_f1_agreement,
        },
        "thresholds": {
            "min_exact_agreement": minimum_exact,
            "min_token_f1_agreement": minimum_token_f1,
        },
        "failures": failures,
        "cases": case_reports,
    }


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records = []
    with path.open(encoding="utf-8-sig") as handle:
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
    parser.add_argument("--case-field", default="case_id")
    parser.add_argument("--prediction-field", default="predicted")
    parser.add_argument("--min-exact-agreement", type=float, default=0.0)
    parser.add_argument("--min-token-f1-agreement", type=float, default=0.0)
    args = parser.parse_args(argv)

    try:
        report = evaluate_consistency(
            _load_jsonl(args.dataset),
            case_field=args.case_field,
            prediction_field=args.prediction_field,
            min_exact_agreement=args.min_exact_agreement,
            min_token_f1_agreement=args.min_token_f1_agreement,
        )
    except (OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
