"""Audit stability drift between paired ranked retrieval outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_CUTOFFS = (1, 3, 5, 10)


def _validate_cutoffs(cutoffs: Sequence[int]) -> tuple[int, ...]:
    if isinstance(cutoffs, (str, bytes)) or not isinstance(cutoffs, Sequence):
        raise ValueError("cutoffs must be a sequence of positive integers")
    values = list(cutoffs)
    if not values:
        raise ValueError("cutoffs must not be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise ValueError("cutoffs must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError("cutoffs must not contain duplicates")
    return tuple(sorted(values))


def _validate_rate(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be between 0 and 1")
    return float(value)


def _validate_ranking(value: Any, *, record_index: int, field: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"record {record_index} field {field!r} must be a list")
    items = list(value)
    if any(not isinstance(item, str) or not item for item in items):
        raise ValueError(
            f"record {record_index} field {field!r} must contain non-empty strings"
        )
    if len(set(items)) != len(items):
        raise ValueError(
            f"record {record_index} field {field!r} must not contain duplicates"
        )
    return items


def audit_ranking_drift(
    records: Sequence[Mapping[str, Any]],
    *,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
    gate_cutoff: int | None = None,
    min_mean_jaccard: float = 0.0,
    max_top_item_change_rate: float = 1.0,
    max_details: int = 20,
    query_field: str = "query_id",
    baseline_field: str = "baseline",
    candidate_field: str = "candidate",
) -> dict[str, Any]:
    """Return deterministic top-k overlap, rank-churn, and gate diagnostics."""

    if not records:
        raise ValueError("at least one record is required")
    field_names = (query_field, baseline_field, candidate_field)
    if any(not isinstance(field, str) or not field for field in field_names):
        raise ValueError("field names must be non-empty strings")
    if len(set(field_names)) != len(field_names):
        raise ValueError("field names must be distinct")
    if (
        isinstance(max_details, bool)
        or not isinstance(max_details, int)
        or max_details < 0
    ):
        raise ValueError("max_details must be a non-negative integer")

    normalized_cutoffs = _validate_cutoffs(cutoffs)
    selected_gate_cutoff = max(normalized_cutoffs) if gate_cutoff is None else gate_cutoff
    if (
        isinstance(selected_gate_cutoff, bool)
        or not isinstance(selected_gate_cutoff, int)
        or selected_gate_cutoff not in normalized_cutoffs
    ):
        raise ValueError("gate_cutoff must be one of the configured cutoffs")
    minimum_jaccard = _validate_rate("min_mean_jaccard", min_mean_jaccard)
    maximum_top_item_change = _validate_rate(
        "max_top_item_change_rate", max_top_item_change_rate
    )

    validated = []
    seen_query_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {index} must be an object")
        query_id = record.get(query_field)
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(
                f"record {index} field {query_field!r} must be a non-empty string"
            )
        if query_id in seen_query_ids:
            raise ValueError(f"record {index} repeats query id {query_id!r}")
        seen_query_ids.add(query_id)
        baseline = _validate_ranking(
            record.get(baseline_field), record_index=index, field=baseline_field
        )
        candidate = _validate_ranking(
            record.get(candidate_field), record_index=index, field=candidate_field
        )
        validated.append((query_id, baseline, candidate))

    details_by_cutoff: dict[int, list[dict[str, Any]]] = {}
    metrics_at_cutoff = []
    for cutoff in normalized_cutoffs:
        query_details = []
        jaccard_total = 0.0
        top_item_changes = 0
        all_shared_displacements = []
        for query_id, baseline, candidate in validated:
            baseline_top = baseline[:cutoff]
            candidate_top = candidate[:cutoff]
            baseline_set = set(baseline_top)
            candidate_set = set(candidate_top)
            shared = baseline_set & candidate_set
            union = baseline_set | candidate_set
            jaccard = len(shared) / len(union) if union else 1.0
            baseline_ranks = {
                item: rank for rank, item in enumerate(baseline_top, start=1)
            }
            candidate_ranks = {
                item: rank for rank, item in enumerate(candidate_top, start=1)
            }
            displacements = [
                abs(baseline_ranks[item] - candidate_ranks[item])
                for item in shared
            ]
            all_shared_displacements.extend(displacements)
            baseline_first = baseline_top[0] if baseline_top else None
            candidate_first = candidate_top[0] if candidate_top else None
            top_item_changed = baseline_first != candidate_first
            top_item_changes += int(top_item_changed)
            jaccard_total += jaccard
            query_details.append(
                {
                    "query_id": query_id,
                    "baseline_top": baseline_top,
                    "candidate_top": candidate_top,
                    "overlap_count": len(shared),
                    "union_count": len(union),
                    "top_k_jaccard": jaccard,
                    "mean_shared_rank_displacement": (
                        sum(displacements) / len(displacements)
                        if displacements
                        else None
                    ),
                    "top_item_changed": top_item_changed,
                }
            )

        query_count = len(validated)
        metrics_at_cutoff.append(
            {
                "cutoff": cutoff,
                "mean_top_k_jaccard": jaccard_total / query_count,
                "top_item_change_rate": top_item_changes / query_count,
                "mean_shared_rank_displacement": (
                    sum(all_shared_displacements) / len(all_shared_displacements)
                    if all_shared_displacements
                    else None
                ),
                "shared_item_comparison_count": len(all_shared_displacements),
            }
        )
        details_by_cutoff[cutoff] = query_details

    gate_metrics = next(
        item for item in metrics_at_cutoff if item["cutoff"] == selected_gate_cutoff
    )
    failures = []
    actual_jaccard = gate_metrics["mean_top_k_jaccard"]
    if actual_jaccard < minimum_jaccard:
        failures.append(
            {
                "metric": "mean_top_k_jaccard",
                "cutoff": selected_gate_cutoff,
                "actual": actual_jaccard,
                "minimum": minimum_jaccard,
                "shortfall": minimum_jaccard - actual_jaccard,
            }
        )
    actual_top_item_change = gate_metrics["top_item_change_rate"]
    if actual_top_item_change > maximum_top_item_change:
        failures.append(
            {
                "metric": "top_item_change_rate",
                "cutoff": selected_gate_cutoff,
                "actual": actual_top_item_change,
                "maximum": maximum_top_item_change,
                "excess": actual_top_item_change - maximum_top_item_change,
            }
        )

    selected_details = details_by_cutoff[selected_gate_cutoff]
    selected_details.sort(
        key=lambda item: (
            item["top_k_jaccard"],
            not item["top_item_changed"],
            item["query_id"],
        )
    )
    return {
        "passed": not failures,
        "query_count": len(validated),
        "cutoffs": list(normalized_cutoffs),
        "metrics_at_cutoff": metrics_at_cutoff,
        "thresholds": {
            "gate_cutoff": selected_gate_cutoff,
            "min_mean_jaccard": minimum_jaccard,
            "max_top_item_change_rate": maximum_top_item_change,
        },
        "failures": failures,
        "query_details": selected_details[:max_details],
        "details_truncated": len(selected_details) > max_details,
        "omitted_query_count": max(0, len(selected_details) - max_details),
        "settings": {
            "query_field": query_field,
            "baseline_field": baseline_field,
            "candidate_field": candidate_field,
            "max_details": max_details,
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


def _paths_alias(source: Path, output: Path) -> bool:
    if source.resolve() == output.resolve():
        return True
    try:
        return source.samefile(output)
    except (FileNotFoundError, OSError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--cutoff", action="append", type=int, dest="cutoffs")
    parser.add_argument("--gate-cutoff", type=int)
    parser.add_argument("--min-mean-jaccard", type=float, default=0.0)
    parser.add_argument("--max-top-item-change-rate", type=float, default=1.0)
    parser.add_argument("--max-details", type=int, default=20)
    parser.add_argument("--query-field", default="query_id")
    parser.add_argument("--baseline-field", default="baseline")
    parser.add_argument("--candidate-field", default="candidate")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.output is not None and _paths_alias(args.dataset, args.output):
            raise ValueError("output must not alias the input dataset")
        report = audit_ranking_drift(
            _load_jsonl(args.dataset),
            cutoffs=args.cutoffs or DEFAULT_CUTOFFS,
            gate_cutoff=args.gate_cutoff,
            min_mean_jaccard=args.min_mean_jaccard,
            max_top_item_change_rate=args.max_top_item_change_rate,
            max_details=args.max_details,
            query_field=args.query_field,
            baseline_field=args.baseline_field,
            candidate_field=args.candidate_field,
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
