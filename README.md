# AI Eval Micro Lab

Small, dependency-free evaluation tools for comparing model predictions with reference answers.

The basic evaluator provides normalized exact match and token-level F1 for JSONL datasets. Each line should contain `expected` and `predicted` strings.

```powershell
python -m ai_eval_micro_lab.cli examples.jsonl
python -m unittest discover -s tests
```

## Paired model comparison

The comparison command evaluates `baseline` and `candidate` predictions against the same `expected` answer. It reports the mean paired improvement for both metrics with deterministic percentile bootstrap confidence intervals.

```powershell
python -m ai_eval_micro_lab.comparison examples/model-comparison.jsonl --samples 2000 --seed 0
```

Using the same records for both systems preserves pairing: each bootstrap sample resamples evaluation cases, not individual scores from unrelated pools. The interval describes uncertainty in the measured dataset and is not a guarantee about future model behavior.

This repository is intended for reproducible learning experiments. Future additions should include tests, a short explanation of the idea, and a runnable example.
