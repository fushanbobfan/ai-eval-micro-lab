"""Evaluate confidence-based selective prediction behavior."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .metrics import exact_match


def _validate_unit_interval(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    return float(value)


def evaluate_selective_prediction(
    records: Sequence[Mapping[str, Any]],
    *,
    confidence_threshold: float = 0.0,
    min_coverage: float = 0.0,
    max_risk: float = 1.0,
) -> dict[str, Any]:
    """Return a selective-prediction operating point and risk-coverage curve."""

    threshold = _validate_unit_interval(
        "confidence_threshold", confidence_threshold
    )
    minimum_coverage = _validate_unit_interval("min_coverage", min_coverage)
    maximum_risk = _validate_unit_interval("max_risk", max_risk)
    if not records:
        raise ValueError("at least one record is required")

    validated: list[tuple[float, float]] = []
    for index, record in enumerate(records):
        expected = record.get("expected")
        predicted = record.get("predicted")
        confidence = record.get("confidence")
        if not isinstance(expected, str) or not isinstance(predicted, str):
            raise ValueError(
                f"record {index} must contain string expected and predicted fields"
            )
        validated_confidence = _validate_unit_interval(
            f"record {index} confidence", confidence
        )
        validated.append(
            (validated_confidence, exact_match(expected, predicted))
        )

    ordered = sorted(validated, key=lambda item: item[0], reverse=True)
    curve = []
    accepted_count = 0
    accepted_correct = 0.0
    previous_coverage = 0.0
    risk_coverage_area = 0.0
    index = 0
    while index < len(ordered):
        group_confidence = ordered[index][0]
        group_end = index
        while (
            group_end < len(ordered)
            and ordered[group_end][0] == group_confidence
        ):
            accepted_correct += ordered[group_end][1]
            group_end += 1
        accepted_count = group_end
        coverage = accepted_count / len(ordered)
        risk = 1.0 - accepted_correct / accepted_count
        risk_coverage_area += risk * (coverage - previous_coverage)
        curve.append(
            {
                "confidence_threshold": group_confidence,
                "accepted_count": accepted_count,
                "coverage": coverage,
                "selective_risk": risk,
            }
        )
        previous_coverage = coverage
        index = group_end

    accepted = [item for item in validated if item[0] >= threshold]
    selected_count = len(accepted)
    coverage = selected_count / len(validated)
    selective_accuracy = (
        sum(item[1] for item in accepted) / selected_count
        if selected_count
        else None
    )
    selective_risk = (
        1.0 - selective_accuracy if selective_accuracy is not None else None
    )

    failures = []
    if coverage < minimum_coverage:
        failures.append(
            {
                "metric": "coverage",
                "actual": coverage,
                "minimum": minimum_coverage,
                "shortfall": minimum_coverage - coverage,
            }
        )
    if selective_risk is None or selective_risk > maximum_risk:
        failures.append(
            {
                "metric": "selective_risk",
                "actual": selective_risk,
                "maximum": maximum_risk,
                "excess": (
                    selective_risk - maximum_risk
                    if selective_risk is not None
                    else None
                ),
                "reason": (
                    "no predictions met the confidence threshold"
                    if selective_risk is None
                    else "maximum risk exceeded"
                ),
            }
        )

    return {
        "passed": not failures,
        "operating_point": {
            "confidence_threshold": threshold,
            "count": len(validated),
            "accepted_count": selected_count,
            "abstained_count": len(validated) - selected_count,
            "coverage": coverage,
            "selective_accuracy": selective_accuracy,
            "selective_risk": selective_risk,
        },
        "risk_coverage_area": risk_coverage_area,
        "risk_coverage_curve": curve,
        "thresholds": {
            "min_coverage": minimum_coverage,
            "max_risk": maximum_risk,
        },
        "failures": failures,
    }
