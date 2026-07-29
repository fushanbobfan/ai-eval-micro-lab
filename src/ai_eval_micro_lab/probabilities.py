"""Evaluate multiclass probability distributions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _validate_optional_threshold(
    name: str,
    value: float | None,
    *,
    lower: float,
    upper: float | None,
) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < lower
        or (upper is not None and value > upper)
    ):
        range_text = (
            f"at least {lower}" if upper is None else f"between {lower} and {upper}"
        )
        raise ValueError(f"{name} threshold must be {range_text}")


def evaluate_probabilistic_classification(
    records: Sequence[Mapping[str, Any]],
    *,
    bins: int = 10,
    top_k: Sequence[int] = (1,),
    gate_top_k: int = 1,
    min_accuracy: float | None = None,
    min_top_k_accuracy: float | None = None,
    max_log_loss: float | None = None,
    max_brier_score: float | None = None,
    max_ece: float | None = None,
    probability_tolerance: float = 1e-6,
    log_floor: float = 1e-15,
) -> dict[str, Any]:
    """Return proper scores, top-k accuracy, and confidence calibration."""

    if not records:
        raise ValueError("at least one record is required")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    if (
        isinstance(probability_tolerance, bool)
        or not isinstance(probability_tolerance, (int, float))
        or not math.isfinite(probability_tolerance)
        or not 0.0 <= probability_tolerance <= 0.1
    ):
        raise ValueError("probability_tolerance must be between 0 and 0.1")
    if (
        isinstance(log_floor, bool)
        or not isinstance(log_floor, (int, float))
        or not math.isfinite(log_floor)
        or not 0.0 < log_floor <= 1.0
    ):
        raise ValueError("log_floor must be greater than 0 and at most 1")

    cutoffs = []
    for cutoff in top_k:
        if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff <= 0:
            raise ValueError("top_k values must be positive integers")
        if cutoff not in cutoffs:
            cutoffs.append(cutoff)
    cutoffs.sort()
    if not cutoffs:
        raise ValueError("at least one top_k value is required")
    if gate_top_k not in cutoffs:
        raise ValueError("gate_top_k must be one of the requested top_k values")

    _validate_optional_threshold(
        "accuracy", min_accuracy, lower=0.0, upper=1.0
    )
    _validate_optional_threshold(
        "top_k_accuracy", min_top_k_accuracy, lower=0.0, upper=1.0
    )
    _validate_optional_threshold(
        "log_loss", max_log_loss, lower=0.0, upper=None
    )
    _validate_optional_threshold(
        "brier_score", max_brier_score, lower=0.0, upper=2.0
    )
    _validate_optional_threshold(
        "expected_calibration_error", max_ece, lower=0.0, upper=1.0
    )

    validated: list[tuple[str, dict[str, float]]] = []
    labels: list[str] | None = None
    for index, record in enumerate(records):
        expected = record.get("expected")
        scores = record.get("scores")
        if not isinstance(expected, str) or not expected:
            raise ValueError(f"record {index} expected must be a non-empty string")
        if not isinstance(scores, Mapping) or not scores:
            raise ValueError(f"record {index} scores must be a non-empty object")

        normalized_scores: dict[str, float] = {}
        for label, score in scores.items():
            if not isinstance(label, str) or not label:
                raise ValueError(
                    f"record {index} score labels must be non-empty strings"
                )
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not 0.0 <= score <= 1.0
            ):
                raise ValueError(
                    f"record {index} score for {label!r} must be between 0 and 1"
                )
            normalized_scores[label] = float(score)

        record_labels = sorted(normalized_scores)
        if labels is None:
            labels = record_labels
        elif record_labels != labels:
            raise ValueError(f"record {index} must use the same score labels")
        if expected not in normalized_scores:
            raise ValueError(f"record {index} expected label is missing from scores")
        probability_sum = sum(normalized_scores.values())
        if abs(probability_sum - 1.0) > probability_tolerance:
            raise ValueError(
                f"record {index} probabilities must sum to 1 within tolerance"
            )
        validated.append((expected, normalized_scores))

    assert labels is not None
    if cutoffs[-1] > len(labels):
        raise ValueError("top_k values cannot exceed the number of score labels")

    top_k_hits = {cutoff: 0 for cutoff in cutoffs}
    expected_counts = {label: 0 for label in labels}
    predicted_counts = {label: 0 for label in labels}
    probability_totals = {label: 0.0 for label in labels}
    confidences: list[tuple[float, float]] = []
    log_losses = []
    brier_scores = []
    zero_expected_probability_count = 0

    for expected, scores in validated:
        ranked = sorted(labels, key=lambda label: (-scores[label], label))
        predicted = ranked[0]
        correct = float(predicted == expected)
        confidence = scores[predicted]
        confidences.append((confidence, correct))
        expected_counts[expected] += 1
        predicted_counts[predicted] += 1
        for label in labels:
            probability_totals[label] += scores[label]
        for cutoff in cutoffs:
            top_k_hits[cutoff] += int(expected in ranked[:cutoff])

        expected_probability = scores[expected]
        if expected_probability == 0.0:
            zero_expected_probability_count += 1
        log_losses.append(-math.log(max(expected_probability, log_floor)))
        brier_scores.append(
            sum(
                (scores[label] - float(label == expected)) ** 2 for label in labels
            )
        )

    grouped: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for confidence, correct in confidences:
        bin_index = min(int(confidence * bins), bins - 1)
        grouped[bin_index].append((confidence, correct))

    calibration_bins = []
    weighted_gap = 0.0
    for index, values in enumerate(grouped):
        if not values:
            continue
        count = len(values)
        mean_confidence = sum(value[0] for value in values) / count
        accuracy = sum(value[1] for value in values) / count
        gap = abs(mean_confidence - accuracy)
        weighted_gap += count * gap
        calibration_bins.append(
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
    accuracy = sum(value[1] for value in confidences) / count
    top_k_accuracy = {
        str(cutoff): top_k_hits[cutoff] / count for cutoff in cutoffs
    }
    metrics = {
        "count": count,
        "accuracy": accuracy,
        "top_k_accuracy": top_k_accuracy,
        "mean_confidence": sum(value[0] for value in confidences) / count,
        "log_loss": sum(log_losses) / count,
        "brier_score": sum(brier_scores) / count,
        "expected_calibration_error": weighted_gap / count,
        "zero_expected_probability_count": zero_expected_probability_count,
    }
    per_class = [
        {
            "label": label,
            "expected_count": expected_counts[label],
            "predicted_count": predicted_counts[label],
            "mean_probability": probability_totals[label] / count,
        }
        for label in labels
    ]

    thresholds = {
        "accuracy": min_accuracy,
        f"top_{gate_top_k}_accuracy": min_top_k_accuracy,
        "log_loss": max_log_loss,
        "brier_score": max_brier_score,
        "expected_calibration_error": max_ece,
    }
    failures = []
    minimum_checks = [
        ("accuracy", accuracy, min_accuracy),
        (
            f"top_{gate_top_k}_accuracy",
            top_k_accuracy[str(gate_top_k)],
            min_top_k_accuracy,
        ),
    ]
    for name, actual, minimum in minimum_checks:
        if minimum is not None and actual < minimum:
            failures.append(
                {
                    "metric": name,
                    "actual": actual,
                    "minimum": minimum,
                    "shortfall": minimum - actual,
                }
            )
    maximum_checks = [
        ("log_loss", metrics["log_loss"], max_log_loss),
        ("brier_score", metrics["brier_score"], max_brier_score),
        (
            "expected_calibration_error",
            metrics["expected_calibration_error"],
            max_ece,
        ),
    ]
    for name, actual, maximum in maximum_checks:
        if maximum is not None and actual > maximum:
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
        "labels": labels,
        "metrics": metrics,
        "per_class": per_class,
        "calibration_bins": calibration_bins,
        "thresholds": thresholds,
        "failures": failures,
        "settings": {
            "bins": bins,
            "top_k": cutoffs,
            "gate_top_k": gate_top_k,
            "probability_tolerance": probability_tolerance,
            "log_floor": log_floor,
        },
    }


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
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
    parser.add_argument(
        "--top-k",
        action="append",
        type=int,
        default=[1],
        help="repeat to report another cutoff; top-1 is always included",
    )
    parser.add_argument("--gate-top-k", type=int, default=1)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--min-accuracy", type=float)
    parser.add_argument("--min-top-k-accuracy", type=float)
    parser.add_argument("--max-log-loss", type=float)
    parser.add_argument("--max-brier-score", type=float)
    parser.add_argument("--max-ece", type=float)
    parser.add_argument("--probability-tolerance", type=float, default=1e-6)
    parser.add_argument("--log-floor", type=float, default=1e-15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        report = evaluate_probabilistic_classification(
            _load_jsonl(args.dataset),
            bins=args.bins,
            top_k=args.top_k,
            gate_top_k=args.gate_top_k,
            min_accuracy=args.min_accuracy,
            min_top_k_accuracy=args.min_top_k_accuracy,
            max_log_loss=args.max_log_loss,
            max_brier_score=args.max_brier_score,
            max_ece=args.max_ece,
            probability_tolerance=args.probability_tolerance,
            log_floor=args.log_floor,
        )
        rendered = json.dumps(report, indent=2) + "\n"
        if args.output is None:
            print(rendered, end="")
        else:
            args.output.write_text(rendered, encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
