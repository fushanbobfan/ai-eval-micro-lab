"""Evaluate text predictions across named dataset slices."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .metrics import evaluate_records


def evaluate_slices(
    records: Sequence[Mapping[str, Any]],
    *,
    slice_by: str,
    min_count: int = 1,
) -> dict[str, Any]:
    """Report overall metrics and deterministic per-slice metrics."""

    if not slice_by:
        raise ValueError("slice_by must be a non-empty field name")
    if min_count < 1:
        raise ValueError("min_count must be positive")

    validated: list[dict[str, str]] = []
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for index, record in enumerate(records):
        expected = record.get("expected")
        predicted = record.get("predicted")
        label = record.get(slice_by)
        if not isinstance(expected, str) or not isinstance(predicted, str):
            raise ValueError(
                f"record {index} must contain string expected and predicted fields"
            )
        if not isinstance(label, str) or not label:
            raise ValueError(
                f"record {index} must contain a non-empty string {slice_by} field"
            )

        pair = {"expected": expected, "predicted": predicted}
        validated.append(pair)
        grouped[label].append(pair)

    included = []
    excluded_slice_count = 0
    excluded_record_count = 0
    for label in sorted(grouped):
        members = grouped[label]
        if len(members) < min_count:
            excluded_slice_count += 1
            excluded_record_count += len(members)
            continue
        included.append({"value": label, **evaluate_records(members)})

    return {
        "slice_by": slice_by,
        "min_count": min_count,
        "overall": evaluate_records(validated),
        "slices": included,
        "excluded": {
            "slices": excluded_slice_count,
            "records": excluded_record_count,
        },
    }
