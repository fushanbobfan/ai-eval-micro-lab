"""Evaluate deterministic single-label classification predictions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def evaluate_classification(
    records: Sequence[Mapping[str, Any]],
    *,
    min_accuracy: float = 0.0,
    min_macro_f1: float = 0.0,
) -> dict[str, Any]:
    """Return confusion, per-class, and aggregate classification metrics."""

    thresholds = {
        "accuracy": min_accuracy,
        "macro_f1": min_macro_f1,
    }
    for name, minimum in thresholds.items():
        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, (int, float))
            or not math.isfinite(minimum)
            or not 0.0 <= minimum <= 1.0
        ):
            raise ValueError(f"{name} threshold must be between 0 and 1")
    if not records:
        raise ValueError("at least one record is required")

    validated: list[tuple[str, str]] = []
    for index, record in enumerate(records):
        expected = record.get("expected")
        predicted = record.get("predicted")
        if not isinstance(expected, str) or not isinstance(predicted, str):
            raise ValueError(
                f"record {index} must contain string expected and predicted fields"
            )
        if not expected or not predicted:
            raise ValueError(
                f"record {index} must contain non-empty expected and predicted labels"
            )
        validated.append((expected, predicted))

    labels = sorted({label for pair in validated for label in pair})
    label_indexes = {label: index for index, label in enumerate(labels)}
    confusion_matrix = [[0 for _ in labels] for _ in labels]
    for expected, predicted in validated:
        confusion_matrix[label_indexes[expected]][label_indexes[predicted]] += 1

    per_class = []
    for index, label in enumerate(labels):
        true_positive = confusion_matrix[index][index]
        support = sum(confusion_matrix[index])
        predicted_count = sum(row[index] for row in confusion_matrix)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class.append(
            {
                "label": label,
                "support": support,
                "predicted_count": predicted_count,
                "true_positive": true_positive,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    count = len(validated)
    metrics = {
        "count": count,
        "accuracy": sum(confusion_matrix[i][i] for i in range(len(labels))) / count,
        "macro_precision": sum(item["precision"] for item in per_class)
        / len(labels),
        "macro_recall": sum(item["recall"] for item in per_class) / len(labels),
        "macro_f1": sum(item["f1"] for item in per_class) / len(labels),
        "weighted_f1": sum(item["f1"] * item["support"] for item in per_class)
        / count,
    }
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
        "labels": labels,
        "confusion_matrix": confusion_matrix,
        "per_class": per_class,
        "metrics": metrics,
        "thresholds": thresholds,
        "failures": failures,
    }
