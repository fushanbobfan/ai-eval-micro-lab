"""Reports must come out the same on every supported Python.

Python 3.12 changed how the built-in ``sum`` adds floats: ``sum([0.1] * 10)`` is
``0.9999999999999999`` on 3.10 and 3.11 and ``1.0`` from 3.12 on. ``math.fsum``
gives one answer on all of them, so the package totals floats with it. Adding only
0.0 and 1.0 is exact on every version, so those counts keep ``sum``, and so do
integer totals, which ``fsum`` would round.

Python 3.12 also changed ``math.log2`` on Windows, where 3.10 and 3.11 compute it
from ``log`` and 3.12 calls the C library's ``log2``. The two differ in the last bit
for many inputs, so the package uses ``math.log(value, 2)``, which goes through the
same ``log`` on every version.
"""

import builtins
import contextlib
import importlib
import inspect
import io
import pkgutil
import sys
import unittest
from pathlib import Path
from unittest import mock

import ai_eval_micro_lab
from ai_eval_micro_lab.multilabel import evaluate_multilabel
from ai_eval_micro_lab.probabilities import evaluate_probabilistic_classification
from ai_eval_micro_lab.probability_drift import audit_probability_drift

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

# Every command with a documented dataset, so a run reaches its float totals.
COMMANDS = {
    "agreement": ["evaluator-agreement.jsonl"],
    "calibration": ["calibration-audit.jsonl", "--bins", "5"],
    "classification": ["classification-evaluation.jsonl"],
    "cli": ["quality-gate.jsonl"],
    "comparison": ["model-comparison.jsonl", "--samples", "50"],
    "consistency": ["output-consistency.jsonl"],
    "disparity": ["group-disparity.jsonl"],
    "distribution": ["label-drift-reference.jsonl", "label-drift-candidate.jsonl"],
    "gate": ["quality-gate.jsonl"],
    "multilabel": ["multilabel-evaluation.jsonl"],
    "overlap": ["overlap-reference.jsonl", "overlap-candidate.jsonl"],
    "paired_correctness": ["paired-correctness.jsonl"],
    "pairwise": ["pairwise-preferences.jsonl"],
    "position_bias": ["presentation-order-bias.jsonl"],
    "probabilities": ["probabilistic-classification.jsonl", "--top-k", "2"],
    "probability_drift": ["probability-drift.jsonl"],
    "ranking_drift": ["ranking-drift.jsonl"],
    "regression_gate": ["regression-gate.jsonl", "--samples", "50"],
    "retrieval": ["retrieval-ranking.jsonl"],
    "runtime": ["runtime-efficiency.jsonl"],
    "score_drift": ["score-drift.jsonl"],
    "selective": ["selective-prediction.jsonl"],
    "slices": ["slice-evaluation.jsonl", "--slice-by", "category"],
}


def _package_modules():
    return {
        info.name: importlib.import_module(f"{ai_eval_micro_lab.__name__}.{info.name}")
        for info in pkgutil.iter_modules(ai_eval_micro_lab.__path__)
    }


_builtin_sum = builtins.sum


def _sum_refusing_floats(iterable, /, start=0):
    items = list(iterable)
    for item in (start, *items):
        if isinstance(item, float) and item not in (0.0, 1.0):
            raise AssertionError(f"sum() received {item!r}; total floats with math.fsum()")
    return _builtin_sum(items, start)


class FloatSummationTests(unittest.TestCase):
    def test_every_command_is_covered(self):
        commands = {
            name for name, module in _package_modules().items() if hasattr(module, "main")
        }
        self.assertEqual(commands, set(COMMANDS))

    def test_no_module_mentions_log2(self):
        for name, module in _package_modules().items():
            with self.subTest(module=name):
                self.assertNotIn("log2", inspect.getsource(module), "use math.log(value, 2)")

    def test_no_command_passes_a_fractional_float_to_sum(self):
        modules = _package_modules()
        for command, arguments in COMMANDS.items():
            argv = [
                str(EXAMPLES / argument) if argument.endswith(".jsonl") else argument
                for argument in arguments
            ]
            with self.subTest(command=command):
                with mock.patch.object(sys, "argv", [command, *argv]):
                    with contextlib.redirect_stdout(io.StringIO()):
                        with mock.patch.object(builtins, "sum", _sum_refusing_floats):
                            exit_code = modules[command].main()
                self.assertIn(exit_code, (None, 0, 1))

        # The multilabel example has no fractional per-label precision to total.
        with mock.patch.object(builtins, "sum", _sum_refusing_floats):
            evaluate_multilabel(
                [
                    {"expected": ["ant"], "predicted": ["ant"]},
                    {"expected": [], "predicted": ["ant"]},
                    {"expected": [], "predicted": ["ant"]},
                ]
            )

    def test_gate_verdict_holds_when_a_mean_lands_on_its_threshold(self):
        records = [
            {"expected": ["ant"], "predicted": ["cow", "cat"]},
            {
                "expected": ["fish", "bird", "cat", "ant", "cow"],
                "predicted": ["cat", "ant", "fish", "cow", "bird"],
            },
        ]

        report = evaluate_multilabel(records, min_macro_f1=0.8)

        self.assertEqual(report["metrics"]["macro_f1"], 0.8)
        self.assertTrue(report["passed"])

    def test_probabilities_totalling_exactly_one_are_accepted(self):
        scores = {f"label-{index}": 0.1 for index in range(10)}

        report = evaluate_probabilistic_classification(
            [{"expected": "label-0", "scores": scores}], probability_tolerance=0.0
        )
        self.assertEqual(report["metrics"]["count"], 1)

        report = audit_probability_drift(
            [{"case_id": "case", "baseline_scores": scores, "candidate_scores": scores}],
            probability_tolerance=1e-17,
        )
        self.assertEqual(report["case_count"], 1)


if __name__ == "__main__":
    unittest.main()
