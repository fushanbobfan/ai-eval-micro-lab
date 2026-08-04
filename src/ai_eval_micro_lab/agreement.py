"""Audit agreement between a reference rater and an evaluated rater."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def _validate_minimum(name: str, value: float | None, *, lower: float) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not lower <= value <= 1.0
    ):
        raise ValueError(f"{name} threshold must be between {lower} and 1")


def evaluate_agreement(
    records: Sequence[Mapping[str, Any]],
    *,
    reference_field: str = "reference",
    rater_field: str = "rater",
    id_field: str = "id",
    min_agreement: float | None = None,
    min_kappa: float | None = None,
    max_disagreements: int | None = None,
    max_details: int = 20,
) -> dict[str, Any]:
    """Return observed agreement, Cohen's kappa, and bounded disagreements."""

    if not records:
        raise ValueError("at least one record is required")
    field_names = {
        "reference_field": reference_field,
        "rater_field": rater_field,
        "id_field": id_field,
    }
    for name, value in field_names.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if len(set(field_names.values())) != len(field_names):
        raise ValueError("reference, rater, and ID fields must be distinct")
    _validate_minimum("agreement", min_agreement, lower=0.0)
    _validate_minimum("kappa", min_kappa, lower=-1.0)
    if (
        max_disagreements is not None
        and (
            isinstance(max_disagreements, bool)
            or not isinstance(max_disagreements, int)
            or max_disagreements < 0
        )
    ):
        raise ValueError("max_disagreements must be a non-negative integer")
    if (
        isinstance(max_details, bool)
        or not isinstance(max_details, int)
        or max_details < 0
    ):
        raise ValueError("max_details must be a non-negative integer")

    validated: list[tuple[str, str, str]] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        case_id = record.get(id_field)
        reference = record.get(reference_field)
        rater = record.get(rater_field)
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"record {index} {id_field} must be a non-empty string")
        if case_id in seen_ids:
            raise ValueError(f"record {index} has duplicate {id_field} {case_id!r}")
        if not isinstance(reference, str) or not reference:
            raise ValueError(
                f"record {index} {reference_field} must be a non-empty string"
            )
        if not isinstance(rater, str) or not rater:
            raise ValueError(
                f"record {index} {rater_field} must be a non-empty string"
            )
        seen_ids.add(case_id)
        validated.append((case_id, reference, rater))

    labels = sorted({label for _, reference, rater in validated for label in (reference, rater)})
    label_indexes = {label: index for index, label in enumerate(labels)}
    confusion_matrix = [[0 for _ in labels] for _ in labels]
    reference_counts: Counter[str] = Counter()
    rater_counts: Counter[str] = Counter()
    disagreements: list[dict[str, str]] = []
    pair_counts: Counter[tuple[str, str]] = Counter()

    for case_id, reference, rater in validated:
        reference_counts[reference] += 1
        rater_counts[rater] += 1
        confusion_matrix[label_indexes[reference]][label_indexes[rater]] += 1
        if reference != rater:
            pair_counts[(reference, rater)] += 1
            if len(disagreements) < max_details:
                disagreements.append(
                    {"id": case_id, "reference": reference, "rater": rater}
                )

    count = len(validated)
    agreement_count = sum(
        confusion_matrix[index][index] for index in range(len(labels))
    )
    disagreement_count = count - agreement_count
    observed_agreement = agreement_count / count
    expected_agreement = sum(
        reference_counts[label] * rater_counts[label] for label in labels
    ) / (count * count)
    kappa = (
        (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
        if expected_agreement < 1.0
        else None
    )

    metrics = {
        "count": count,
        "agreement_count": agreement_count,
        "disagreement_count": disagreement_count,
        "observed_agreement": observed_agreement,
        "expected_agreement": expected_agreement,
        "cohen_kappa": kappa,
        "kappa_status": (
            "defined" if kappa is not None else "undefined_expected_agreement_one"
        ),
    }
    per_label = [
        {
            "label": label,
            "reference_count": reference_counts[label],
            "rater_count": rater_counts[label],
            "agreement_count": confusion_matrix[index][index],
        }
        for index, label in enumerate(labels)
    ]
    disagreement_pairs = [
        {"reference": reference, "rater": rater, "count": pair_count}
        for (reference, rater), pair_count in sorted(
            pair_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )
    ]

    thresholds = {
        "observed_agreement": min_agreement,
        "cohen_kappa": min_kappa,
        "disagreement_count": max_disagreements,
    }
    failures = []
    if min_agreement is not None and observed_agreement < min_agreement:
        failures.append(
            {
                "metric": "observed_agreement",
                "actual": observed_agreement,
                "minimum": min_agreement,
                "shortfall": min_agreement - observed_agreement,
            }
        )
    if min_kappa is not None:
        if kappa is None:
            failures.append(
                {
                    "metric": "cohen_kappa",
                    "actual": None,
                    "minimum": min_kappa,
                    "reason": "undefined_expected_agreement_one",
                }
            )
        elif kappa < min_kappa:
            failures.append(
                {
                    "metric": "cohen_kappa",
                    "actual": kappa,
                    "minimum": min_kappa,
                    "shortfall": min_kappa - kappa,
                }
            )
    if max_disagreements is not None and disagreement_count > max_disagreements:
        failures.append(
            {
                "metric": "disagreement_count",
                "actual": disagreement_count,
                "maximum": max_disagreements,
                "excess": disagreement_count - max_disagreements,
            }
        )

    return {
        "passed": not failures,
        "labels": labels,
        "confusion_matrix": confusion_matrix,
        "per_label": per_label,
        "metrics": metrics,
        "disagreement_pairs": disagreement_pairs,
        "disagreement_details": disagreements,
        "details_truncated": disagreement_count > len(disagreements),
        "thresholds": thresholds,
        "failures": failures,
        "settings": {
            "reference_field": reference_field,
            "rater_field": rater_field,
            "id_field": id_field,
            "max_details": max_details,
        },
    }
