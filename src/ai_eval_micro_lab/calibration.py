"""Measure confidence calibration for exact-match predictions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .metrics import exact_match


def evaluate_calibration(
    records: Sequence[Mapping[str, Any]],
    *,
    bins: int = 10,
    max_ece: float = 1.0,
    max_brier: float = 1.0,
) -> dict[str, Any]:
    """Return calibration metrics and threshold failures for model confidences."""

    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    thresholds = {
        "expected_calibration_error": max_ece,
        "brier_score": max_brier,
    }
    for name, maximum in thresholds.items():
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, (int, float))
            or not math.isfinite(maximum)
        ):
            raise ValueError(f"{name} threshold must be a finite number")
        if not 0.0 <= maximum <= 1.0:
            raise ValueError(f"{name} threshold must be between 0 and 1")
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
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError(f"record {index} confidence must be between 0 and 1")
        validated.append((float(confidence), exact_match(expected, predicted)))

    grouped: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for confidence, correct in validated:
        bin_index = min(int(confidence * bins), bins - 1)
        grouped[bin_index].append((confidence, correct))

    bin_reports = []
    weighted_gap = 0.0
    for index, values in enumerate(grouped):
        if not values:
            continue
        count = len(values)
        mean_confidence = sum(value[0] for value in values) / count
        accuracy = sum(value[1] for value in values) / count
        gap = abs(mean_confidence - accuracy)
        weighted_gap += count * gap
        bin_reports.append(
            {
                "index": index,
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "gap": gap,
            }
        )

    count = len(validated)
    metrics = {
        "count": count,
        "accuracy": sum(value[1] for value in validated) / count,
        "mean_confidence": sum(value[0] for value in validated) / count,
        "brier_score": sum(
            (confidence - correct) ** 2 for confidence, correct in validated
        )
        / count,
        "expected_calibration_error": weighted_gap / count,
    }
    failures = []
    for name, maximum in thresholds.items():
        actual = metrics[name]
        if actual > maximum:
            failures.append(
                {
                    "metric": name,
                    "actual": actual,
                    "maximum": maximum,
                    "excess": actual - maximum,
                }
            )

    return {
        "passed": not failures,
        "bins": bin_reports,
        "metrics": metrics,
        "thresholds": thresholds,
        "failures": failures,
    }
