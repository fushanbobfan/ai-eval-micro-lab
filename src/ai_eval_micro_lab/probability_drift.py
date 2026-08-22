"""Audit paired categorical probability distributions for bounded drift."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _unit_interval(name: str, value: float | None) -> float | None:
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


def _probabilities(
    value: Any,
    *,
    field: str,
    record_index: int,
    probability_tolerance: float,
) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"record {record_index} field {field!r} must be a non-empty object")

    scores: dict[str, float] = {}
    for label, score in value.items():
        if not isinstance(label, str) or not label:
            raise ValueError(
                f"record {record_index} field {field!r} labels must be non-empty strings"
            )
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0.0 <= score <= 1.0
        ):
            raise ValueError(
                f"record {record_index} probability for {label!r} must be between 0 and 1"
            )
        scores[label] = float(score)

    if abs(sum(scores.values()) - 1.0) > probability_tolerance:
        raise ValueError(
            f"record {record_index} field {field!r} probabilities must sum to 1 within tolerance"
        )
    return scores


def _jensen_shannon_bits(
    baseline: Mapping[str, float], candidate: Mapping[str, float]
) -> float:
    divergence = 0.0
    for label in baseline:
        left = baseline[label]
        right = candidate[label]
        midpoint = (left + right) / 2.0
        if left > 0.0:
            divergence += 0.5 * left * math.log2(left / midpoint)
        if right > 0.0:
            divergence += 0.5 * right * math.log2(right / midpoint)
    return divergence


def audit_probability_drift(
    records: Sequence[Mapping[str, Any]],
    *,
    max_mean_total_variation: float | None = None,
    max_case_total_variation: float | None = None,
    max_mean_js_divergence: float | None = None,
    max_top_label_change_rate: float | None = None,
    probability_tolerance: float = 1e-6,
    max_details: int = 20,
    id_field: str = "case_id",
    baseline_field: str = "baseline_scores",
    candidate_field: str = "candidate_scores",
) -> dict[str, Any]:
    """Return deterministic paired probability-drift metrics and gate failures."""
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("records must be a sequence")
    if not records:
        raise ValueError("at least one record is required")
    fields = (id_field, baseline_field, candidate_field)
    if any(not isinstance(field, str) or not field for field in fields):
        raise ValueError("field names must be non-empty strings")
    if len(set(fields)) != len(fields):
        raise ValueError("field names must be distinct")
    if (
        isinstance(probability_tolerance, bool)
        or not isinstance(probability_tolerance, (int, float))
        or not math.isfinite(probability_tolerance)
        or not 0.0 < probability_tolerance <= 0.1
    ):
        raise ValueError("probability_tolerance must be greater than 0 and at most 0.1")
    if isinstance(max_details, bool) or not isinstance(max_details, int) or max_details < 0:
        raise ValueError("max_details must be a non-negative integer")

    thresholds = {
        "mean_total_variation": _unit_interval(
            "max_mean_total_variation", max_mean_total_variation
        ),
        "case_total_variation": _unit_interval(
            "max_case_total_variation", max_case_total_variation
        ),
        "mean_js_divergence": _unit_interval(
            "max_mean_js_divergence", max_mean_js_divergence
        ),
        "top_label_change_rate": _unit_interval(
            "max_top_label_change_rate", max_top_label_change_rate
        ),
    }

    seen_ids: set[str] = set()
    labels: list[str] | None = None
    validated: list[tuple[str, dict[str, float], dict[str, float]]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {index} must be an object")
        case_id = record.get(id_field)
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(
                f"record {index} field {id_field!r} must be a non-empty string"
            )
        if case_id in seen_ids:
            raise ValueError(f"record {index} repeats case id {case_id!r}")
        seen_ids.add(case_id)

        baseline = _probabilities(
            record.get(baseline_field),
            field=baseline_field,
            record_index=index,
            probability_tolerance=probability_tolerance,
        )
        candidate = _probabilities(
            record.get(candidate_field),
            field=candidate_field,
            record_index=index,
            probability_tolerance=probability_tolerance,
        )
        baseline_labels = sorted(baseline)
        if sorted(candidate) != baseline_labels:
            raise ValueError(
                f"record {index} baseline and candidate must use the same labels"
            )
        if labels is None:
            labels = baseline_labels
        elif baseline_labels != labels:
            raise ValueError(f"record {index} must use the same labels as earlier records")
        validated.append((case_id, baseline, candidate))

    assert labels is not None
    probability_totals = {
        label: {"baseline": 0.0, "candidate": 0.0, "absolute_change": 0.0}
        for label in labels
    }
    case_details = []
    top_label_change_count = 0
    for case_id, baseline, candidate in validated:
        baseline_top = min(labels, key=lambda label: (-baseline[label], label))
        candidate_top = min(labels, key=lambda label: (-candidate[label], label))
        top_label_changed = baseline_top != candidate_top
        top_label_change_count += int(top_label_changed)
        total_variation = 0.5 * sum(
            abs(candidate[label] - baseline[label]) for label in labels
        )
        js_divergence = _jensen_shannon_bits(baseline, candidate)
        case_details.append(
            {
                "case_id": case_id,
                "baseline_top_label": baseline_top,
                "candidate_top_label": candidate_top,
                "top_label_changed": top_label_changed,
                "total_variation": total_variation,
                "js_divergence_bits": js_divergence,
            }
        )
        for label in labels:
            probability_totals[label]["baseline"] += baseline[label]
            probability_totals[label]["candidate"] += candidate[label]
            probability_totals[label]["absolute_change"] += abs(
                candidate[label] - baseline[label]
            )

    case_count = len(case_details)
    mean_total_variation = sum(
        detail["total_variation"] for detail in case_details
    ) / case_count
    mean_js_divergence = sum(
        detail["js_divergence_bits"] for detail in case_details
    ) / case_count
    maximum_tv = min(
        case_details,
        key=lambda detail: (-detail["total_variation"], detail["case_id"]),
    )
    maximum_js = min(
        case_details,
        key=lambda detail: (-detail["js_divergence_bits"], detail["case_id"]),
    )
    top_label_change_rate = top_label_change_count / case_count

    failures = []
    checks = (
        ("mean_total_variation", mean_total_variation),
        ("case_total_variation", maximum_tv["total_variation"]),
        ("mean_js_divergence", mean_js_divergence),
        ("top_label_change_rate", top_label_change_rate),
    )
    for metric, actual in checks:
        maximum = thresholds[metric]
        if maximum is not None and actual > maximum:
            failures.append(
                {
                    "metric": metric,
                    "actual": actual,
                    "maximum": maximum,
                    "excess": actual - maximum,
                }
            )

    per_label = []
    for label in labels:
        baseline_mean = probability_totals[label]["baseline"] / case_count
        candidate_mean = probability_totals[label]["candidate"] / case_count
        per_label.append(
            {
                "label": label,
                "baseline_mean_probability": baseline_mean,
                "candidate_mean_probability": candidate_mean,
                "mean_probability_shift": candidate_mean - baseline_mean,
                "mean_absolute_probability_change": (
                    probability_totals[label]["absolute_change"] / case_count
                ),
            }
        )

    case_details.sort(
        key=lambda detail: (
            -detail["js_divergence_bits"],
            -detail["total_variation"],
            detail["case_id"],
        )
    )
    return {
        "passed": not failures,
        "case_count": case_count,
        "labels": labels,
        "metrics": {
            "mean_total_variation": mean_total_variation,
            "maximum_total_variation": maximum_tv["total_variation"],
            "maximum_total_variation_case_id": maximum_tv["case_id"],
            "mean_js_divergence_bits": mean_js_divergence,
            "maximum_js_divergence_bits": maximum_js["js_divergence_bits"],
            "maximum_js_divergence_case_id": maximum_js["case_id"],
            "top_label_change_count": top_label_change_count,
            "top_label_change_rate": top_label_change_rate,
        },
        "per_label": per_label,
        "thresholds": thresholds,
        "failures": failures,
        "case_details": case_details[:max_details],
        "details_truncated": case_count > max_details,
        "omitted_case_count": max(0, case_count - max_details),
        "settings": {
            "probability_tolerance": float(probability_tolerance),
            "max_details": max_details,
            "id_field": id_field,
            "baseline_field": baseline_field,
            "candidate_field": candidate_field,
        },
    }
