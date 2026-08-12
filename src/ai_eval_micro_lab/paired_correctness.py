"""Compare paired baseline and candidate correctness outcomes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any

from .metrics import exact_match


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


def _validate_difference(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not -1.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be between -1 and 1")
    return float(value)


def _validate_count(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


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


def _mcnemar_exact_p_value(baseline_only: int, candidate_only: int) -> float:
    discordant_count = baseline_only + candidate_only
    if discordant_count == 0:
        return 1.0
    smaller_count = min(baseline_only, candidate_only)
    tail_numerator = sum(
        math.comb(discordant_count, count)
        for count in range(smaller_count + 1)
    )
    return min(1.0, 2.0 * tail_numerator / (1 << discordant_count))


def evaluate_paired_correctness(
    records: Sequence[Mapping[str, Any]],
    *,
    id_field: str = "id",
    expected_field: str = "expected",
    baseline_field: str = "baseline",
    candidate_field: str = "candidate",
    confidence: float = 0.95,
    min_accuracy_difference: float | None = None,
    max_regressions: int | None = None,
    max_exact_p_value: float | None = None,
    max_details: int = 20,
) -> dict[str, Any]:
    """Return paired correctness transitions, uncertainty, and gate results."""

    if not records:
        raise ValueError("at least one record is required")
    field_names = {
        "id_field": id_field,
        "expected_field": expected_field,
        "baseline_field": baseline_field,
        "candidate_field": candidate_field,
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
    if isinstance(max_details, bool) or not isinstance(max_details, int) or max_details < 0:
        raise ValueError("max_details must be a non-negative integer")

    thresholds = {
        "accuracy_difference": _validate_difference(
            "min_accuracy_difference", min_accuracy_difference
        ),
        "regressions": _validate_count("max_regressions", max_regressions),
        "mcnemar_exact_p_value": _validate_rate(
            "max_exact_p_value", max_exact_p_value
        ),
    }

    seen_ids: set[str] = set()
    transitions = {
        "both_correct": 0,
        "both_incorrect": 0,
        "baseline_only_correct": 0,
        "candidate_only_correct": 0,
    }
    discordant_details = []
    for index, record in enumerate(records):
        case_id = record.get(id_field)
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"record {index} {id_field} must be a non-empty string")
        if case_id in seen_ids:
            raise ValueError(f"record {index} has duplicate {id_field} {case_id!r}")
        seen_ids.add(case_id)

        values = []
        for field in (expected_field, baseline_field, candidate_field):
            value = record.get(field)
            if not isinstance(value, str):
                raise ValueError(f"record {index} {field} must be a string")
            values.append(value)
        expected, baseline, candidate = values
        baseline_correct = bool(exact_match(expected, baseline))
        candidate_correct = bool(exact_match(expected, candidate))

        if baseline_correct and candidate_correct:
            transition = "both_correct"
        elif baseline_correct:
            transition = "baseline_only_correct"
        elif candidate_correct:
            transition = "candidate_only_correct"
        else:
            transition = "both_incorrect"
        transitions[transition] += 1
        if baseline_correct != candidate_correct:
            discordant_details.append({"id": case_id, "transition": transition})

    count = len(records)
    baseline_correct_count = (
        transitions["both_correct"] + transitions["baseline_only_correct"]
    )
    candidate_correct_count = (
        transitions["both_correct"] + transitions["candidate_only_correct"]
    )
    baseline_only = transitions["baseline_only_correct"]
    candidate_only = transitions["candidate_only_correct"]
    discordant_count = baseline_only + candidate_only
    accuracy_difference = (candidate_correct_count - baseline_correct_count) / count
    candidate_discordant_win_rate = (
        candidate_only / discordant_count if discordant_count else None
    )
    interval = (
        _wilson_interval(candidate_only, discordant_count, float(confidence))
        if discordant_count
        else None
    )
    p_value = _mcnemar_exact_p_value(baseline_only, candidate_only)

    failures = []
    minimum_difference = thresholds["accuracy_difference"]
    if minimum_difference is not None and accuracy_difference < minimum_difference:
        failures.append(
            {
                "metric": "accuracy_difference",
                "actual": accuracy_difference,
                "minimum": minimum_difference,
                "shortfall": minimum_difference - accuracy_difference,
            }
        )
    maximum_regressions = thresholds["regressions"]
    if maximum_regressions is not None and baseline_only > maximum_regressions:
        failures.append(
            {
                "metric": "regressions",
                "actual": baseline_only,
                "maximum": maximum_regressions,
                "excess": baseline_only - maximum_regressions,
            }
        )
    maximum_p_value = thresholds["mcnemar_exact_p_value"]
    if maximum_p_value is not None:
        if candidate_only <= baseline_only:
            failures.append(
                {
                    "metric": "mcnemar_exact_p_value",
                    "actual": p_value,
                    "maximum": maximum_p_value,
                    "reason": "candidate_not_better_on_discordant_cases",
                }
            )
        elif p_value > maximum_p_value:
            failures.append(
                {
                    "metric": "mcnemar_exact_p_value",
                    "actual": p_value,
                    "maximum": maximum_p_value,
                    "excess": p_value - maximum_p_value,
                }
            )

    return {
        "passed": not failures,
        "metrics": {
            "count": count,
            "baseline_correct": baseline_correct_count,
            "candidate_correct": candidate_correct_count,
            "baseline_accuracy": baseline_correct_count / count,
            "candidate_accuracy": candidate_correct_count / count,
            "accuracy_difference": accuracy_difference,
            "transitions": transitions,
            "discordant_count": discordant_count,
            "candidate_discordant_win_rate": candidate_discordant_win_rate,
            "candidate_discordant_win_rate_interval": interval,
            "mcnemar_exact_p_value": p_value,
        },
        "thresholds": thresholds,
        "failures": failures,
        "discordant_details": discordant_details[:max_details],
        "details_truncated": len(discordant_details) > max_details,
        "settings": {
            **field_names,
            "confidence": float(confidence),
            "max_details": max_details,
        },
    }
