"""Evaluate ranked retrieval results at deterministic cutoffs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_CUTOFFS = (1, 3, 5, 10)


def _validate_string_list(
    value: Any,
    *,
    record_index: int,
    field: str,
    allow_empty: bool,
) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"record {record_index} field {field!r} must be a list")
    items = list(value)
    if not allow_empty and not items:
        raise ValueError(f"record {record_index} field {field!r} must not be empty")
    if any(not isinstance(item, str) or not item for item in items):
        raise ValueError(
            f"record {record_index} field {field!r} must contain non-empty strings"
        )
    if len(set(items)) != len(items):
        raise ValueError(
            f"record {record_index} field {field!r} must not contain duplicates"
        )
    return items


def _validate_cutoffs(cutoffs: Sequence[int]) -> tuple[int, ...]:
    if isinstance(cutoffs, (str, bytes)) or not isinstance(cutoffs, Sequence):
        raise ValueError("cutoffs must be a sequence of positive integers")
    values = list(cutoffs)
    if not values:
        raise ValueError("cutoffs must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("cutoffs must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError("cutoffs must not contain duplicates")
    return tuple(sorted(values))


def _validate_threshold(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be between 0 and 1")
    return float(value)


def evaluate_retrieval(
    records: Sequence[Mapping[str, Any]],
    *,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
    gate_cutoff: int | None = None,
    min_hit_rate: float = 0.0,
    min_mrr: float = 0.0,
    min_recall: float = 0.0,
    min_ndcg: float = 0.0,
    query_field: str = "query_id",
    relevant_field: str = "relevant",
    retrieved_field: str = "retrieved",
) -> dict[str, Any]:
    """Return binary-relevance retrieval metrics and threshold failures."""

    if not records:
        raise ValueError("at least one record is required")
    field_names = (query_field, relevant_field, retrieved_field)
    if any(not isinstance(field, str) or not field for field in field_names):
        raise ValueError("field names must be non-empty strings")
    if len(set(field_names)) != len(field_names):
        raise ValueError("field names must be distinct")

    normalized_cutoffs = _validate_cutoffs(cutoffs)
    selected_gate_cutoff = max(normalized_cutoffs) if gate_cutoff is None else gate_cutoff
    if (
        isinstance(selected_gate_cutoff, bool)
        or not isinstance(selected_gate_cutoff, int)
        or selected_gate_cutoff not in normalized_cutoffs
    ):
        raise ValueError("gate_cutoff must be one of the configured cutoffs")

    thresholds = {
        "hit_rate": _validate_threshold("min_hit_rate", min_hit_rate),
        "mean_reciprocal_rank": _validate_threshold("min_mrr", min_mrr),
        "mean_recall": _validate_threshold("min_recall", min_recall),
        "mean_ndcg": _validate_threshold("min_ndcg", min_ndcg),
    }

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
        relevant = _validate_string_list(
            record.get(relevant_field),
            record_index=index,
            field=relevant_field,
            allow_empty=False,
        )
        retrieved = _validate_string_list(
            record.get(retrieved_field),
            record_index=index,
            field=retrieved_field,
            allow_empty=True,
        )
        relevant_set = set(relevant)
        relevant_ranks = [
            rank
            for rank, item in enumerate(retrieved, start=1)
            if item in relevant_set
        ]
        validated.append((query_id, relevant, retrieved, relevant_ranks))

    query_reports = [
        {
            "query_id": query_id,
            "relevant_count": len(relevant),
            "retrieved_count": len(retrieved),
            "first_relevant_rank": ranks[0] if ranks else None,
        }
        for query_id, relevant, retrieved, ranks in validated
    ]

    metrics_at_cutoff = []
    for cutoff in normalized_cutoffs:
        hit_total = 0
        reciprocal_rank_total = 0.0
        recall_total = 0.0
        ndcg_total = 0.0
        for _, relevant, _, relevant_ranks in validated:
            ranks = [rank for rank in relevant_ranks if rank <= cutoff]
            hit_total += int(bool(ranks))
            reciprocal_rank_total += 1.0 / ranks[0] if ranks else 0.0
            recall_total += len(ranks) / len(relevant)
            dcg = sum(1.0 / math.log2(rank + 1) for rank in ranks)
            ideal_count = min(len(relevant), cutoff)
            ideal_dcg = sum(
                1.0 / math.log2(rank + 1)
                for rank in range(1, ideal_count + 1)
            )
            ndcg_total += dcg / ideal_dcg
        count = len(validated)
        metrics_at_cutoff.append(
            {
                "cutoff": cutoff,
                "hit_rate": hit_total / count,
                "mean_reciprocal_rank": reciprocal_rank_total / count,
                "mean_recall": recall_total / count,
                "mean_ndcg": ndcg_total / count,
            }
        )

    gate_metrics = next(
        metrics
        for metrics in metrics_at_cutoff
        if metrics["cutoff"] == selected_gate_cutoff
    )
    failures = []
    for metric, minimum in thresholds.items():
        actual = gate_metrics[metric]
        if actual < minimum:
            failures.append(
                {
                    "metric": metric,
                    "cutoff": selected_gate_cutoff,
                    "actual": actual,
                    "minimum": minimum,
                    "shortfall": minimum - actual,
                }
            )

    return {
        "passed": not failures,
        "query_count": len(validated),
        "cutoffs": list(normalized_cutoffs),
        "metrics_at_cutoff": metrics_at_cutoff,
        "queries": query_reports,
        "thresholds": {
            "gate_cutoff": selected_gate_cutoff,
            "minimums": thresholds,
        },
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
                raise ValueError(f"invalid JSON on line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            records.append(record)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--cutoff", action="append", type=int, dest="cutoffs")
    parser.add_argument("--gate-cutoff", type=int)
    parser.add_argument("--min-hit-rate", type=float, default=0.0)
    parser.add_argument("--min-mrr", type=float, default=0.0)
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--min-ndcg", type=float, default=0.0)
    parser.add_argument("--query-field", default="query_id")
    parser.add_argument("--relevant-field", default="relevant")
    parser.add_argument("--retrieved-field", default="retrieved")
    args = parser.parse_args(argv)

    try:
        report = evaluate_retrieval(
            _load_jsonl(args.dataset),
            cutoffs=args.cutoffs or DEFAULT_CUTOFFS,
            gate_cutoff=args.gate_cutoff,
            min_hit_rate=args.min_hit_rate,
            min_mrr=args.min_mrr,
            min_recall=args.min_recall,
            min_ndcg=args.min_ndcg,
            query_field=args.query_field,
            relevant_field=args.relevant_field,
            retrieved_field=args.retrieved_field,
        )
    except (OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
