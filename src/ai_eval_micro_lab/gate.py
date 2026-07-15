"""Apply deterministic quality thresholds to text evaluation records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
