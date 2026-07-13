"""Compare two prediction columns with paired bootstrap intervals."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .metrics import exact_match, token_f1


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int]:
    """Estimate a mean paired difference and percentile confidence interval."""

    if not baseline or len(baseline) != len(candidate):
        raise ValueError("baseline and candidate must have the same non-zero length")
    if samples < 1:
        raise ValueError("samples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    differences = [new - old for old, new in zip(baseline, candidate)]
    observed = sum(differences) / len(differences)
    generator = random.Random(seed)
    simulated = sorted(
        sum(generator.choice(differences) for _ in differences) / len(differences)
        for _ in range(samples)
    )
    tail = (1 - confidence) / 2
    return {
        "mean_difference": observed,
        "confidence": confidence,
        "lower": _quantile(simulated, tail),
        "upper": _quantile(simulated, 1 - tail),
        "bootstrap_samples": samples,
        "seed": seed,
    }


def compare_records(
    records: Sequence[Mapping[str, Any]],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare baseline and candidate predictions against the same references."""

    if not records:
        raise ValueError("at least one record is required")

    metric_scores: dict[str, tuple[list[float], list[float]]] = {
        "exact_match": ([], []),
        "token_f1": ([], []),
    }
    for index, record in enumerate(records):
        values = [record.get(field) for field in ("expected", "baseline", "candidate")]
        if not all(isinstance(value, str) for value in values):
            raise ValueError(
                f"record {index} must contain string expected, baseline, and candidate fields"
            )
        expected, baseline, candidate = values
        metric_scores["exact_match"][0].append(exact_match(expected, baseline))
        metric_scores["exact_match"][1].append(exact_match(expected, candidate))
        metric_scores["token_f1"][0].append(token_f1(expected, baseline))
        metric_scores["token_f1"][1].append(token_f1(expected, candidate))

    baseline_summary = {}
    candidate_summary = {}
    deltas = {}
    for offset, (name, (old_scores, new_scores)) in enumerate(metric_scores.items()):
        baseline_summary[name] = sum(old_scores) / len(old_scores)
        candidate_summary[name] = sum(new_scores) / len(new_scores)
        deltas[name] = paired_bootstrap(
            old_scores,
            new_scores,
            samples=samples,
            confidence=confidence,
            seed=seed + offset,
        )

    return {
        "count": len(records),
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "difference": deltas,
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
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        report = compare_records(
            _load_jsonl(args.dataset),
            samples=args.samples,
            confidence=args.confidence,
            seed=args.seed,
        )
    except (OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
