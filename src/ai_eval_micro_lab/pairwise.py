"""Audit paired preference outcomes between a baseline and candidate system."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any


_OUTCOMES = ("baseline", "candidate", "tie")


def _validate_rate(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be between 0 and 1")
    return float(value)


def _wilson_interval(successes: int, trials: int, confidence: float) -> dict[str, Any]:
    proportion = successes / trials
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )
    return {
        "method": "wilson",
        "confidence": confidence,
        "lower": max(0.0, center - half_width),
        "upper": min(1.0, center + half_width),
    }


def evaluate_pairwise_preferences(
    records: Sequence[Mapping[str, Any]],
    *,
    id_field: str = "id",
    outcome_field: str = "winner",
    confidence: float = 0.95,
    min_candidate_win_rate: float | None = None,
    min_candidate_win_rate_lower_bound: float | None = None,
    max_tie_rate: float | None = None,
    max_details: int = 20,
) -> dict[str, Any]:
    """Return win, tie, uncertainty, and gate diagnostics for paired judgments."""

    if not records:
        raise ValueError("at least one record is required")
    for name, value in (("id_field", id_field), ("outcome_field", outcome_field)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if id_field == outcome_field:
        raise ValueError("id_field and outcome_field must be distinct")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("confidence must be strictly between 0 and 1")
    thresholds = {
        "candidate_win_rate": _validate_rate(
            "min_candidate_win_rate", min_candidate_win_rate
        ),
        "candidate_win_rate_lower_bound": _validate_rate(
            "min_candidate_win_rate_lower_bound",
            min_candidate_win_rate_lower_bound,
        ),
        "tie_rate": _validate_rate("max_tie_rate", max_tie_rate),
    }
    if (
        isinstance(max_details, bool)
        or not isinstance(max_details, int)
        or max_details < 0
    ):
        raise ValueError("max_details must be a non-negative integer")

    validated: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        case_id = record.get(id_field)
        outcome = record.get(outcome_field)
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"record {index} {id_field} must be a non-empty string")
        if case_id in seen_ids:
            raise ValueError(f"record {index} has duplicate {id_field} {case_id!r}")
        if outcome not in _OUTCOMES:
            raise ValueError(
                f"record {index} {outcome_field} must be baseline, candidate, or tie"
            )
        seen_ids.add(case_id)
        validated.append((case_id, outcome))

    counts = Counter(outcome for _, outcome in validated)
    total = len(validated)
    decisive_count = counts["baseline"] + counts["candidate"]
    candidate_win_rate = (
        counts["candidate"] / decisive_count if decisive_count else None
    )
    interval = (
        _wilson_interval(counts["candidate"], decisive_count, float(confidence))
        if decisive_count
        else None
    )
    tie_rate = counts["tie"] / total
    non_candidate = [
        {"id": case_id, "outcome": outcome}
        for case_id, outcome in validated
        if outcome != "candidate"
    ]

    failures = []
    minimum = thresholds["candidate_win_rate"]
    if minimum is not None:
        if candidate_win_rate is None:
            failures.append(
                {
                    "metric": "candidate_win_rate",
                    "actual": None,
                    "minimum": minimum,
                    "reason": "no_decisive_comparisons",
                }
            )
        elif candidate_win_rate < minimum:
            failures.append(
                {
                    "metric": "candidate_win_rate",
                    "actual": candidate_win_rate,
                    "minimum": minimum,
                    "shortfall": minimum - candidate_win_rate,
                }
            )
    minimum_lower = thresholds["candidate_win_rate_lower_bound"]
    if minimum_lower is not None:
        if interval is None:
            failures.append(
                {
                    "metric": "candidate_win_rate_lower_bound",
                    "actual": None,
                    "minimum": minimum_lower,
                    "reason": "no_decisive_comparisons",
                }
            )
        elif interval["lower"] < minimum_lower:
            failures.append(
                {
                    "metric": "candidate_win_rate_lower_bound",
                    "actual": interval["lower"],
                    "minimum": minimum_lower,
                    "shortfall": minimum_lower - interval["lower"],
                }
            )
    maximum_ties = thresholds["tie_rate"]
    if maximum_ties is not None and tie_rate > maximum_ties:
        failures.append(
            {
                "metric": "tie_rate",
                "actual": tie_rate,
                "maximum": maximum_ties,
                "excess": tie_rate - maximum_ties,
            }
        )

    return {
        "passed": not failures,
        "metrics": {
            "count": total,
            "candidate_wins": counts["candidate"],
            "baseline_wins": counts["baseline"],
            "ties": counts["tie"],
            "decisive_count": decisive_count,
            "candidate_win_rate": candidate_win_rate,
            "baseline_win_rate": (
                counts["baseline"] / decisive_count if decisive_count else None
            ),
            "tie_rate": tie_rate,
            "candidate_share_all": counts["candidate"] / total,
            "preference_margin_all": (
                counts["candidate"] - counts["baseline"]
            )
            / total,
            "candidate_win_rate_interval": interval,
        },
        "thresholds": thresholds,
        "failures": failures,
        "non_candidate_details": non_candidate[:max_details],
        "details_truncated": len(non_candidate) > max_details,
        "settings": {
            "id_field": id_field,
            "outcome_field": outcome_field,
            "confidence": float(confidence),
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
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--outcome-field", default="winner")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--min-candidate-win-rate", type=float)
    parser.add_argument("--min-candidate-win-rate-lower-bound", type=float)
    parser.add_argument("--max-tie-rate", type=float)
    parser.add_argument("--max-details", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.output is not None and _paths_alias(args.dataset, args.output):
            raise ValueError("output must not alias the source dataset")
        report = evaluate_pairwise_preferences(
            _load_jsonl(args.dataset),
            id_field=args.id_field,
            outcome_field=args.outcome_field,
            confidence=args.confidence,
            min_candidate_win_rate=args.min_candidate_win_rate,
            min_candidate_win_rate_lower_bound=(
                args.min_candidate_win_rate_lower_bound
            ),
            max_tie_rate=args.max_tie_rate,
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
