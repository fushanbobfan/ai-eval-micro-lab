# AI Eval Micro Lab

Small, dependency-free evaluation tools for comparing model predictions with reference answers.

The basic evaluator provides normalized exact match and token-level F1 for JSONL datasets. Each line should contain `expected` and `predicted` strings.

```powershell
python -m ai_eval_micro_lab.cli examples.jsonl
python -m unittest discover -s tests
```

## Multiclass classification evaluation

Free-form answer metrics can hide which discrete classes a classifier confuses. The classification command treats `expected` and `predicted` as exact, case-sensitive labels and reports a row-expected, column-predicted confusion matrix alongside accuracy and per-class precision, recall, and F1:

```powershell
python -m ai_eval_micro_lab.classification examples/classification-evaluation.jsonl `
  --min-accuracy 0.70 --min-macro-f1 0.65
```

Labels are the sorted union of expected and predicted values, so a class produced only by the model remains visible with zero support. The report also includes macro precision, macro recall, macro F1, and support-weighted F1. Macro metrics give every observed label equal weight, while weighted F1 reflects the expected-label distribution in the supplied data.

Exit code `0` means the configured accuracy and macro-F1 minimums passed, `1` reports structured threshold shortfalls, and `2` identifies invalid JSON, labels, or configuration. Choose thresholds on separate development data when possible; results on one dataset do not establish future class balance or performance.

## Paired model comparison

The comparison command evaluates `baseline` and `candidate` predictions against the same `expected` answer. It reports the mean paired improvement for both metrics with deterministic percentile bootstrap confidence intervals.

```powershell
python -m ai_eval_micro_lab.comparison examples/model-comparison.jsonl --samples 2000 --seed 0
```

Using the same records for both systems preserves pairing: each bootstrap sample resamples evaluation cases, not individual scores from unrelated pools. The interval describes uncertainty in the measured dataset and is not a guarantee about future model behavior.

## Slice evaluation

Aggregate scores can hide weak categories. The slice command groups a standard `expected`/`predicted` dataset by a named string field and reports both overall and per-slice metrics in deterministic order.

```powershell
python -m ai_eval_micro_lab.slices examples/slice-evaluation.jsonl --slice-by category
```

Use `--min-count` to omit undersized slices from the detailed list. The report keeps their slice and record counts visible, and the overall metrics always include every validated record. Invalid JSON, missing fields, non-string labels, and non-positive thresholds return exit code `2`.

## CI quality gate

Turn the standard evaluation metrics into a deterministic build check by setting one or both minimum scores:

```powershell
python -m ai_eval_micro_lab.gate examples/quality-gate.jsonl `
  --min-exact-match 0.65 --min-token-f1 0.80
```

The JSON result includes the measured scores, configured thresholds, and a structured entry for each shortfall. Exit code `0` means every threshold passed, `1` means the dataset was valid but a quality threshold failed, and `2` means the input or configuration was invalid. An empty dataset is rejected so a missing evaluation artifact cannot silently pass a pipeline.

## Repeated-output consistency

Accuracy against a reference does not show whether repeated runs return the same answer. Group multiple predictions for each case and measure pairwise normalized exact agreement plus pairwise token F1:

```powershell
python -m ai_eval_micro_lab.consistency examples/output-consistency.jsonl `
  --min-exact-agreement 0.45 `
  --min-token-f1-agreement 0.60
```

The report includes pair-weighted overall metrics and a deterministic breakdown for every case with at least two predictions. Singleton cases remain visible in the summary but cannot contribute an agreement pair. Use `--case-field` and `--prediction-field` when an existing JSONL export uses different names. Exit code `0` means both minimums passed, `1` reports threshold shortfalls, and `2` identifies invalid data or configuration.

Repeated samples should use the same prompt, decoding settings, and model version if the goal is to isolate run-to-run variability. Pairwise agreement describes the supplied cases; correlated samples or a small test set can make the result look more stable than future traffic.

## Paired regression gate

An absolute quality threshold can pass even when a new model is worse than the model it replaces. The regression gate compares `baseline` and `candidate` predictions on the same records, then requires the lower bound of each paired bootstrap interval to clear a configured minimum difference:

```powershell
python -m ai_eval_micro_lab.regression_gate examples/regression-gate.jsonl `
  --min-exact-match-difference 0.02 `
  --min-token-f1-difference 0.01 `
  --samples 2000 --confidence 0.95 --seed 0
```

Exit code `0` means both lower bounds met their minimums, `1` reports every metric shortfall, and `2` identifies invalid data or configuration. A minimum of `0` asks the sampled lower confidence bound to support non-regression; negative minimums can express an explicit tolerance. The deterministic percentile interval describes uncertainty in the supplied evaluation set and should not be read as a guarantee about production behavior.

## Confidence calibration audit

Accuracy does not show whether a model's confidence estimates are trustworthy. The calibration command accepts standard `expected` and `predicted` strings plus a numeric `confidence` from `0` to `1`, interpreted as the model's estimated probability that its normalized exact-match answer is correct:

```powershell
python -m ai_eval_micro_lab.calibration examples/calibration-audit.jsonl `
  --bins 5 --max-ece 0.12 --max-brier 0.15
```

The report includes accuracy, mean confidence, Brier score, expected calibration error (ECE), and every non-empty equal-width confidence bin. Brier score and ECE are both lower-is-better; the CLI exits `1` when either configured maximum is exceeded and `2` for invalid data or configuration. Bin boundaries are left-inclusive, with confidence `1` included in the final bin. Small evaluation sets can produce unstable calibration estimates, so the output should be treated as a dataset diagnostic rather than a production guarantee.

## Selective prediction audit

A model can abstain on low-confidence answers when downstream review is available. The selective prediction command evaluates one fixed confidence cutoff, reports the resulting coverage and exact-match risk among accepted answers, and can enforce both a minimum coverage and maximum risk:

```powershell
python -m ai_eval_micro_lab.selective examples/selective-prediction.jsonl `
  --confidence-threshold 0.70 --min-coverage 0.60 --max-risk 0.25
```

The JSON report includes accepted and abstained counts, the selected operating point, and a risk-coverage curve across every observed confidence. Equal-confidence records enter the curve together, so reordering tied inputs cannot change the result. `risk_coverage_area` is a lower-is-better diagnostic computed from those tie-grouped steps. Exit code `0` means the configured operating point passed, `1` means coverage or risk failed, and `2` identifies invalid input or configuration.

Choose the deployed confidence cutoff before evaluating a held-out dataset. Selecting it on the same examples used for the final report can overstate performance, and the measured tradeoff does not guarantee production behavior.

This repository is intended for reproducible learning experiments. Future additions should include tests, a short explanation of the idea, and a runnable example.
