"""Gate paired model comparisons on bootstrap lower confidence bounds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
