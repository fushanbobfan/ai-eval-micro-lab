"""Evaluate deterministic multi-label classification predictions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _validate_unit_interval(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} threshold must be between 0 and 1")
    return float(value)


def _validate_labels(
    record: Mapping[str, Any],
    *,
    index: int,
    field: str,
) -> frozenset[str]:
    values = record.get(field)
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"record {index} {field} must be an array of labels")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(
            f"record {index} {field} must contain non-empty string labels"
        )
    if len(set(values)) != len(values):
        raise ValueError(f"record {index} {field} must not contain duplicate labels")
    return frozenset(values)


def _precision_recall_f1(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> tuple[float, float, float]:
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def evaluate_multilabel(
    records: Sequence[Mapping[str, Any]],
    *,
    min_micro_f1: float = 0.0,
    min_macro_f1: float = 0.0,
    max_hamming_loss: float = 1.0,
) -> dict[str, Any]:
    """Return per-label and aggregate metrics for multi-label predictions."""

    thresholds = {
        "micro_f1": _validate_unit_interval("micro_f1", min_micro_f1),
        "macro_f1": _validate_unit_interval("macro_f1", min_macro_f1),
        "hamming_loss": _validate_unit_interval(
            "hamming_loss", max_hamming_loss
        ),
    }
    if not records:
        raise ValueError("at least one record is required")

    validated = []
    all_labels: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {index} must be an object")
        expected = _validate_labels(record, index=index, field="expected")
        predicted = _validate_labels(record, index=index, field="predicted")
        validated.append((expected, predicted))
        all_labels.update(expected)
        all_labels.update(predicted)
    if not all_labels:
        raise ValueError("at least one label must appear in expected or predicted")

    labels = sorted(all_labels)
    per_label = []
    total_true_positive = 0
    total_false_positive = 0
    total_false_negative = 0
    subset_matches = 0
    hamming_errors = 0
    sample_jaccard = 0.0
    expected_assignments = 0
    predicted_assignments = 0

    for expected, predicted in validated:
        subset_matches += expected == predicted
        hamming_errors += len(expected ^ predicted)
        expected_assignments += len(expected)
        predicted_assignments += len(predicted)
        union = expected | predicted
        sample_jaccard += len(expected & predicted) / len(union) if union else 1.0

    for label in labels:
        true_positive = sum(
            label in expected and label in predicted
            for expected, predicted in validated
        )
        false_positive = sum(
            label not in expected and label in predicted
            for expected, predicted in validated
        )
        false_negative = sum(
            label in expected and label not in predicted
            for expected, predicted in validated
        )
        true_negative = len(validated) - (
            true_positive + false_positive + false_negative
        )
        precision, recall, f1 = _precision_recall_f1(
            true_positive,
            false_positive,
            false_negative,
        )
        total_true_positive += true_positive
        total_false_positive += false_positive
        total_false_negative += false_negative
        per_label.append(
            {
                "label": label,
                "support": true_positive + false_negative,
                "predicted_count": true_positive + false_positive,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    micro_precision, micro_recall, micro_f1 = _precision_recall_f1(
        total_true_positive,
        total_false_positive,
        total_false_negative,
    )
    count = len(validated)
    metrics = {
        "count": count,
        "label_count": len(labels),
        "subset_accuracy": subset_matches / count,
        "hamming_loss": hamming_errors / (count * len(labels)),
        "samples_jaccard": sample_jaccard / count,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": sum(item["precision"] for item in per_label)
        / len(labels),
        "macro_recall": sum(item["recall"] for item in per_label) / len(labels),
        "macro_f1": sum(item["f1"] for item in per_label) / len(labels),
        "weighted_f1": (
            sum(item["f1"] * item["support"] for item in per_label)
            / expected_assignments
            if expected_assignments
            else 0.0
        ),
        "average_expected_labels": expected_assignments / count,
        "average_predicted_labels": predicted_assignments / count,
    }

    failures = []
    for name in ("micro_f1", "macro_f1"):
        if metrics[name] < thresholds[name]:
            failures.append(
                {
                    "metric": name,
                    "actual": metrics[name],
                    "minimum": thresholds[name],
                    "shortfall": thresholds[name] - metrics[name],
                }
            )
    if metrics["hamming_loss"] > thresholds["hamming_loss"]:
        failures.append(
            {
                "metric": "hamming_loss",
                "actual": metrics["hamming_loss"],
                "maximum": thresholds["hamming_loss"],
                "excess": metrics["hamming_loss"] - thresholds["hamming_loss"],
            }
        )

    return {
        "passed": not failures,
        "labels": labels,
        "per_label": per_label,
        "metrics": metrics,
        "thresholds": thresholds,
        "failures": failures,
    }
