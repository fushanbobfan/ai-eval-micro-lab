"""Text metrics designed to be easy to audit and extend."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping


def normalize(text: str) -> str:
    """Lowercase text, remove punctuation, and collapse whitespace."""
    words = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    return " ".join(words)


def exact_match(expected: str, predicted: str) -> float:
    return float(normalize(expected) == normalize(predicted))


def token_f1(expected: str, predicted: str) -> float:
    expected_tokens = normalize(expected).split()
    predicted_tokens = normalize(predicted).split()
    if not expected_tokens and not predicted_tokens:
        return 1.0
    if not expected_tokens or not predicted_tokens:
        return 0.0

    overlap = sum((Counter(expected_tokens) & Counter(predicted_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_records(records: Iterable[Mapping[str, str]]) -> dict[str, float | int]:
    scores = [
        (
            exact_match(record["expected"], record["predicted"]),
            token_f1(record["expected"], record["predicted"]),
        )
        for record in records
    ]
    if not scores:
        return {"count": 0, "exact_match": 0.0, "token_f1": 0.0}
    return {
        "count": len(scores),
        "exact_match": sum(score[0] for score in scores) / len(scores),
        "token_f1": sum(score[1] for score in scores) / len(scores),
    }

