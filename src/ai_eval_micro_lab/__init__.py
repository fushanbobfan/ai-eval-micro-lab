"""Small, inspectable AI evaluation utilities."""

from typing import Any

from .metrics import exact_match, evaluate_records, token_f1

__all__ = ["evaluate_slices", "exact_match", "evaluate_records", "token_f1"]


def __getattr__(name: str) -> Any:
    if name == "evaluate_slices":
        from .slices import evaluate_slices

        return evaluate_slices
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
