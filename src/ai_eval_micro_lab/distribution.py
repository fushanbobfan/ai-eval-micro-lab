"""Audit categorical label-distribution drift between two datasets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _validate_threshold(name: str, value: float | None) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} threshold must be between 0 and 1")


def _label_counts(
    records: Sequence[Mapping[str, Any]], *, label_field: str, dataset_name: str
) -> Counter[str]:
    if not records:
        raise ValueError(f"{dataset_name} dataset must contain at least one record")

    counts: Counter[str] = Counter()
    for index, record in enumerate(records):
        label = record.get(label_field)
        if not isinstance(label, str) or not label:
            raise ValueError(
                f"{dataset_name} record {index} {label_field} "
                "must be a non-empty string"
            )
        counts[label] += 1
    return counts


def _kl_term(probability: float, midpoint: float) -> float:
    return probability * math.log2(probability / midpoint) if probability else 0.0


def audit_label_distribution(
    reference_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    *,
    label_field: str = "label",
    max_total_variation: float | None = None,
    max_js_divergence: float | None = None,
    max_label_delta: float | None = None,
    max_details: int = 20,
) -> dict[str, Any]:
    """Return bounded categorical prevalence-shift diagnostics and gates."""

    if not isinstance(label_field, str) or not label_field:
        raise ValueError("label_field must be a non-empty string")
    _validate_threshold("total_variation", max_total_variation)
    _validate_threshold("jensen_shannon_divergence", max_js_divergence)
    _validate_threshold("label_delta", max_label_delta)
    if (
        isinstance(max_details, bool)
        or not isinstance(max_details, int)
        or max_details < 0
    ):
        raise ValueError("max_details must be a non-negative integer")

    reference_counts = _label_counts(
        reference_records, label_field=label_field, dataset_name="reference"
    )
    candidate_counts = _label_counts(
        candidate_records, label_field=label_field, dataset_name="candidate"
    )
    reference_count = sum(reference_counts.values())
    candidate_count = sum(candidate_counts.values())
    labels = sorted(reference_counts.keys() | candidate_counts.keys())

    shifts = []
    absolute_deltas = []
    js_divergence = 0.0
    for label in labels:
        reference_prevalence = reference_counts[label] / reference_count
        candidate_prevalence = candidate_counts[label] / candidate_count
        delta = candidate_prevalence - reference_prevalence
        absolute_delta = abs(delta)
        midpoint = (reference_prevalence + candidate_prevalence) / 2.0
        js_divergence += 0.5 * (
            _kl_term(reference_prevalence, midpoint)
            + _kl_term(candidate_prevalence, midpoint)
        )
        absolute_deltas.append(absolute_delta)
        shifts.append(
            {
                "label": label,
                "reference_count": reference_counts[label],
                "candidate_count": candidate_counts[label],
                "reference_prevalence": reference_prevalence,
                "candidate_prevalence": candidate_prevalence,
                "prevalence_delta": delta,
                "absolute_prevalence_delta": absolute_delta,
            }
        )

    shifts.sort(key=lambda item: (-item["absolute_prevalence_delta"], item["label"]))
    total_variation = 0.5 * sum(absolute_deltas)
    max_absolute_delta = max(absolute_deltas)
    metrics = {
        "reference_count": reference_count,
        "candidate_count": candidate_count,
        "label_count": len(labels),
        "reference_only_label_count": sum(
            label not in candidate_counts for label in labels
        ),
        "candidate_only_label_count": sum(
            label not in reference_counts for label in labels
        ),
        "total_variation": total_variation,
        "jensen_shannon_divergence_bits": js_divergence,
        "max_absolute_prevalence_delta": max_absolute_delta,
    }
    thresholds = {
        "total_variation": max_total_variation,
        "jensen_shannon_divergence_bits": max_js_divergence,
        "max_absolute_prevalence_delta": max_label_delta,
    }
    failures = []
    for metric, actual, maximum in (
        ("total_variation", total_variation, max_total_variation),
        ("jensen_shannon_divergence_bits", js_divergence, max_js_divergence),
        ("max_absolute_prevalence_delta", max_absolute_delta, max_label_delta),
    ):
        if maximum is not None and actual > maximum:
            failures.append(
                {
                    "metric": metric,
                    "actual": actual,
                    "maximum": maximum,
                    "excess": actual - maximum,
                }
            )

    return {
        "passed": not failures,
        "metrics": metrics,
        "thresholds": thresholds,
        "failures": failures,
        "label_shifts": shifts[:max_details],
        "details_truncated": len(shifts) > max_details,
        "omitted_label_count": max(0, len(shifts) - max_details),
        "settings": {"label_field": label_field, "max_details": max_details},
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


def _paths_alias(source: Path, output: Path) -> bool:
    if source.resolve() == output.resolve():
        return True
    try:
        return source.samefile(output)
    except (FileNotFoundError, OSError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--max-total-variation", type=float)
    parser.add_argument("--max-js-divergence", type=float)
    parser.add_argument("--max-label-delta", type=float)
    parser.add_argument("--max-details", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.output is not None and (
            _paths_alias(args.reference, args.output)
            or _paths_alias(args.candidate, args.output)
        ):
            raise ValueError("output must not alias an input dataset")
        report = audit_label_distribution(
            _load_jsonl(args.reference),
            _load_jsonl(args.candidate),
            label_field=args.label_field,
            max_total_variation=args.max_total_variation,
            max_js_divergence=args.max_js_divergence,
            max_label_delta=args.max_label_delta,
            max_details=args.max_details,
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
