"""Audit exact and normalized text overlap between two datasets."""

from __future__ import annotations

import argparse
import json
import math
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def normalize_overlap_text(value: str) -> str:
    """Return a conservative key for case and whitespace-insensitive matching."""

    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _validate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    dataset_name: str,
    id_field: str,
    text_field: str,
) -> list[tuple[str, str, str]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError(f"{dataset_name} records must be a sequence")
    if not records:
        raise ValueError(f"{dataset_name} dataset must contain at least one record")

    validated = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"{dataset_name} record {index} must be an object")
        record_id = record.get(id_field)
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(
                f"{dataset_name} record {index} field {id_field!r} "
                "must be a non-empty string"
            )
        if record_id in seen_ids:
            raise ValueError(
                f"{dataset_name} record {index} repeats id {record_id!r}"
            )
        text = record.get(text_field)
        if not isinstance(text, str) or not text:
            raise ValueError(
                f"{dataset_name} record {index} field {text_field!r} "
                "must be a non-empty string"
            )
        normalized = normalize_overlap_text(text)
        if not normalized:
            raise ValueError(
                f"{dataset_name} record {index} field {text_field!r} "
                "must contain non-whitespace text"
            )
        seen_ids.add(record_id)
        validated.append((record_id, text, normalized))
    return validated


def _validate_rate(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError("max_overlap_rate must be between 0 and 1")
    return float(value)


def audit_dataset_overlap(
    reference_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    *,
    id_field: str = "id",
    text_field: str = "text",
    max_overlap_rate: float = 0.0,
    max_details: int = 100,
) -> dict[str, Any]:
    """Return cross-dataset overlap counts and an optional threshold failure."""

    if any(
        not isinstance(field, str) or not field
        for field in (id_field, text_field)
    ):
        raise ValueError("field names must be non-empty strings")
    if id_field == text_field:
        raise ValueError("id_field and text_field must be distinct")
    maximum = _validate_rate(max_overlap_rate)
    if isinstance(max_details, bool) or not isinstance(max_details, int) or max_details < 0:
        raise ValueError("max_details must be a non-negative integer")

    reference = _validate_records(
        reference_records,
        dataset_name="reference",
        id_field=id_field,
        text_field=text_field,
    )
    candidate = _validate_records(
        candidate_records,
        dataset_name="candidate",
        id_field=id_field,
        text_field=text_field,
    )

    reference_by_key: dict[str, list[tuple[str, str]]] = defaultdict(list)
    candidate_by_key: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for record_id, text, normalized in reference:
        reference_by_key[normalized].append((record_id, text))
    for record_id, text, normalized in candidate:
        candidate_by_key[normalized].append((record_id, text))

    matching_keys = sorted(reference_by_key.keys() & candidate_by_key.keys())
    overlapping_reference_ids: set[str] = set()
    overlapping_candidate_ids: set[str] = set()
    exact_pair_count = 0
    normalized_only_pair_count = 0
    matches = []

    for key in matching_keys:
        reference_group = sorted(reference_by_key[key])
        candidate_group = sorted(candidate_by_key[key])
        overlapping_reference_ids.update(item[0] for item in reference_group)
        overlapping_candidate_ids.update(item[0] for item in candidate_group)
        reference_text_counts = Counter(text for _, text in reference_group)
        candidate_text_counts = Counter(text for _, text in candidate_group)
        group_pair_count = len(reference_group) * len(candidate_group)
        group_exact_count = sum(
            count * candidate_text_counts[text]
            for text, count in reference_text_counts.items()
        )
        exact_pair_count += group_exact_count
        normalized_only_pair_count += group_pair_count - group_exact_count

        if len(matches) >= max_details:
            continue
        for reference_id, reference_text in reference_group:
            for candidate_id, candidate_text in candidate_group:
                matches.append(
                    {
                        "reference_id": reference_id,
                        "candidate_id": candidate_id,
                        "match_type": (
                            "exact"
                            if reference_text == candidate_text
                            else "normalized"
                        ),
                    }
                )
                if len(matches) >= max_details:
                    break
            if len(matches) >= max_details:
                break

    pair_count = exact_pair_count + normalized_only_pair_count
    candidate_overlap_rate = len(overlapping_candidate_ids) / len(candidate)
    failures = []
    if candidate_overlap_rate > maximum:
        failures.append(
            {
                "metric": "candidate_overlap_rate",
                "actual": candidate_overlap_rate,
                "maximum": maximum,
                "excess": candidate_overlap_rate - maximum,
            }
        )

    return {
        "passed": not failures,
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "overlap": {
            "matching_group_count": len(matching_keys),
            "overlapping_reference_count": len(overlapping_reference_ids),
            "overlapping_candidate_count": len(overlapping_candidate_ids),
            "candidate_overlap_rate": candidate_overlap_rate,
            "pair_count": pair_count,
            "exact_pair_count": exact_pair_count,
            "normalized_only_pair_count": normalized_only_pair_count,
            "matches": matches,
            "details_truncated": pair_count > len(matches),
        },
        "thresholds": {"max_overlap_rate": maximum},
        "failures": failures,
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
                raise ValueError(
                    f"invalid JSON in {path} on line {line_number}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path} line {line_number} must contain a JSON object"
                )
            records.append(record)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dataset", type=Path)
    parser.add_argument("candidate_dataset", type=Path)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-overlap-rate", type=float, default=0.0)
    parser.add_argument("--max-details", type=int, default=100)
    args = parser.parse_args(argv)

    try:
        report = audit_dataset_overlap(
            _load_jsonl(args.reference_dataset),
            _load_jsonl(args.candidate_dataset),
            id_field=args.id_field,
            text_field=args.text_field,
            max_overlap_rate=args.max_overlap_rate,
            max_details=args.max_details,
        )
    except (OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
