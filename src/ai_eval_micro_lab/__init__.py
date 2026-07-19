"""Small, inspectable AI evaluation utilities."""

from typing import Any

from .metrics import exact_match, evaluate_records, token_f1

__all__ = [
    "evaluate_calibration",
    "evaluate_consistency",
    "evaluate_gate",
    "evaluate_regression_gate",
    "evaluate_selective_prediction",
    "evaluate_slices",
    "exact_match",
    "evaluate_records",
    "token_f1",
]


def __getattr__(name: str) -> Any:
    if name == "evaluate_calibration":
        from .calibration import evaluate_calibration

        return evaluate_calibration
    if name == "evaluate_consistency":
        from .consistency import evaluate_consistency

        return evaluate_consistency
    if name == "evaluate_gate":
        from .gate import evaluate_gate

        return evaluate_gate
    if name == "evaluate_regression_gate":
        from .regression_gate import evaluate_regression_gate

        return evaluate_regression_gate
    if name == "evaluate_selective_prediction":
        from .selective import evaluate_selective_prediction

        return evaluate_selective_prediction
    if name == "evaluate_slices":
        from .slices import evaluate_slices

        return evaluate_slices
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
