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

## Label distribution drift audit

Class prevalence can change even when two evaluation exports use the same label vocabulary. Compare a reference dataset with a candidate dataset before treating their aggregate scores as directly comparable:

```powershell
python -m ai_eval_micro_lab.distribution `
  examples/label-drift-reference.jsonl `
  examples/label-drift-candidate.jsonl `
  --max-total-variation 0.35 `
  --max-js-divergence 0.15 `
  --max-label-delta 0.35
```

Each JSONL object needs a non-empty, case-sensitive string `label`; use `--label-field` for another schema. The report gives reference and candidate counts, labels seen on only one side, total variation distance, Jensen-Shannon divergence in bits, the largest absolute prevalence change, and a bounded list of label shifts sorted by magnitude. Total variation, Jensen-Shannon divergence, and absolute prevalence deltas all range from `0` to `1`, where `0` means the observed label proportions match.

Exit code `0` means every configured maximum passed, `1` reports structured drift failures, and `2` identifies invalid JSON, empty datasets, malformed labels, unsafe output aliasing, or invalid configuration. This command measures marginal label prevalence only. It does not identify covariate or concept drift, explain why a distribution changed, show whether annotations are correct, or establish that model quality changed. Small samples can also make prevalence shifts unstable; choose gates from the intended sampling process rather than tuning them on the final comparison.

## Evaluator agreement audit

Accuracy against one reference label can overstate evaluator reliability when a dominant class makes chance agreement likely. Compare a human or adjudicated reference rater with another human, heuristic, or model-based evaluator using exact categorical labels:

```powershell
python -m ai_eval_micro_lab.agreement examples/evaluator-agreement.jsonl `
  --min-agreement 0.60 --min-kappa 0.30 `
  --max-disagreements 2 --max-details 20
```

Each JSONL object needs a unique string `id` plus non-empty, case-sensitive `reference` and `rater` labels. The report includes the row-reference, column-rater confusion matrix; observed and chance-expected agreement; Cohen's kappa; per-label usage; sorted disagreement-pair counts; and a bounded list of disagreeing IDs. Source text is not copied into the report. Use the field-name flags when an existing export uses different keys.

Exit code `0` means every configured minimum or maximum passed, `1` reports structured agreement, kappa, or disagreement-count failures, and `2` identifies invalid JSON, duplicate IDs, malformed labels, unsafe output aliasing, or invalid configuration. Kappa is reported as undefined when both raters use the same single label, and a configured kappa gate then fails explicitly instead of treating perfect raw agreement as meaningful chance-corrected agreement.

Agreement is not correctness: two raters can consistently share the same error, and label prevalence can make kappa unstable on small or highly imbalanced samples. Establish the reference process and thresholds independently of the final evaluation set when possible. IDs can still be sensitive, so use non-identifying case keys in artifacts intended for CI or publication.

## Pairwise preference audit

Summarize blinded or otherwise controlled baseline-versus-candidate judgments without discarding ties:

```powershell
python -m ai_eval_micro_lab.pairwise examples/pairwise-preferences.jsonl `
  --min-candidate-win-rate 0.70 `
  --max-tie-rate 0.20
```

Each JSONL object needs a unique string `id` and a `winner` equal to `baseline`, `candidate`, or `tie`. The report gives outcome counts, tie rate, candidate and baseline win rates among decisive comparisons, the candidate share and preference margin across all comparisons, and a Wilson confidence interval for the decisive candidate win rate. A bounded detail list contains only IDs and non-candidate outcomes. Use field-name flags for an existing export.

Exit code `0` means the configured minimum win rate, minimum Wilson lower bound, and maximum tie rate passed; `1` reports structured gate failures; and `2` identifies invalid JSON, duplicate IDs, malformed outcomes, unsafe output aliasing, or invalid configuration. An all-tie dataset keeps decisive rates explicitly undefined, and any configured decisive-rate gate fails rather than inventing a score.

Pairwise preference is not correctness or practical significance. Results depend on the judge, rubric, presentation order, tie policy, sampling process, and independence of comparisons. The Wilson interval describes binomial sampling uncertainty for the supplied decisive outcomes only; it does not correct judge bias, repeated prompts, multiple comparisons, or selection on the same evaluation set. Keep IDs non-identifying in reports intended for sharing.

## Presentation-order bias audit

A pairwise win rate can hide a judge that favors whichever answer appears first or second. Repeat each comparison with the candidate order reversed, then audit whether the selected candidate remains stable:

```powershell
python -m ai_eval_micro_lab.position_bias `
  examples/presentation-order-bias.jsonl `
  --min-complete-pairs 4 `
  --min-robust-preference-rate 0.50 `
  --max-position-flip-rate 0.25 `
  --max-incomplete-pairs 0
```

Each JSONL object needs a non-empty string `pair_id`, distinct non-empty `first` and `second` candidate identifiers, and a `winner` equal to `first`, `second`, or `tie`. A complete pair contains exactly two presentations with the candidates reversed. The report distinguishes stable candidate preferences, stable ties, first-position flips, second-position flips, one-sided tie instability, and incomplete pairs. Bounded details contain only pair IDs and classifications, not candidate identifiers or answer content. Field-name flags support existing exports, and the input size is bounded.

Exit code `0` means every configured count or rate gate passed, `1` reports structured failures, and `2` identifies invalid JSON, duplicate or non-reversed presentations, malformed fields, an oversized input, unsafe output aliasing, or invalid configuration. Rates use complete pairs as their denominator and stay undefined when no complete pair exists; configured rate gates then fail explicitly.

This audit detects order sensitivity only in the supplied reversed presentations. It does not determine which answer is correct, validate the judge or rubric, isolate other prompt-position effects, or guarantee that the measured rates generalize. Candidate identifiers and pair IDs can still be sensitive, repeated cases may be dependent, and thresholds selected on the same final sample can overstate stability.

## Paired correctness comparison

When baseline and candidate outputs share the same reference answers, compare their correctness transitions directly instead of treating two aggregate accuracies as independent:

```powershell
python -m ai_eval_micro_lab.paired_correctness `
  examples/paired-correctness.jsonl `
  --min-accuracy-difference 0.50 `
  --max-regressions 1 `
  --max-exact-p-value 0.05
```

Each JSONL object needs a unique string `id` and string `expected`, `baseline`, and `candidate` fields. Correctness uses the repository's normalized exact match. The report separates both-correct, both-incorrect, baseline-only-correct regressions, and candidate-only-correct improvements; reports each system's accuracy and their paired difference; and summarizes the discordant cases with a Wilson interval plus the two-sided exact McNemar binomial p-value. Bounded details contain IDs and transition names only, not answer text. Field-name flags support existing export schemas.

Exit code `0` means every configured accuracy-difference, regression-count, and exact-p-value gate passed; `1` reports structured gate failures; and `2` identifies invalid JSON, duplicate IDs, malformed fields, unsafe output aliasing, or invalid configuration. The p-value gate is direction-aware: a candidate with no net advantage among discordant cases cannot pass merely because a statistically detectable change favors the baseline.

The exact test addresses only the null model that candidate-only and baseline-only correctness are equally likely among independent discordant cases. It does not measure practical importance, repair incorrect references, account for repeated or clustered cases, or correct for trying multiple datasets, metrics, thresholds, or model variants. Select gates before the final comparison and report the accuracy difference and regression count alongside the p-value.

## Inference runtime audit

Quality metrics do not show whether an evaluated model run met its operational budget. Audit recorded attempts with deterministic success-rate, latency, and token-use summaries:

```powershell
python -m ai_eval_micro_lab.runtime examples/runtime-efficiency.jsonl `
  --min-success-rate 0.80 `
  --max-p95-latency-ms 650 `
  --max-mean-total-tokens 150 `
  --max-failures 1
```

Each JSONL object needs a unique string `id`, a finite non-negative `latency_ms`, non-negative integer `input_tokens` and `output_tokens`, and a Boolean `success`. The report includes attempt counts, success rate, mean and nearest-rank p50/p95/maximum latency, aggregate and per-attempt token use, bounded slow-attempt summaries, and bounded failure IDs. Use the field-name flags for an existing instrumentation schema. The input file is size-bounded, and output aliasing is rejected.

Exit code `0` means every configured budget passed, `1` reports structured threshold failures, and `2` identifies invalid JSON, duplicate IDs, malformed measurements, an oversized input, unsafe output aliasing, or invalid configuration. Latency includes every recorded attempt, including failures. These metrics depend on the supplied instrumentation, hardware, concurrency, cache state, model version, and workload mix. They do not measure answer quality, estimate provider cost, establish capacity, or predict production reliability; compare like-for-like runs and evaluate quality separately.

## Probabilistic classification evaluation

Hard labels hide whether the model assigned sensible probability mass to alternatives. Evaluate a complete multiclass probability distribution on every record:

```powershell
python -m ai_eval_micro_lab.probabilities `
  examples/probabilistic-classification.jsonl `
  --top-k 2 --gate-top-k 2 `
  --min-accuracy 0.60 --min-top-k-accuracy 0.90 `
  --max-log-loss 0.80 --max-brier-score 0.45 --max-ece 0.30
```

Each JSONL object needs a non-empty string `expected` label and a `scores` object whose finite values are between `0` and `1`. Every record must use the same label set, include its expected label, and sum to `1` within the configured tolerance. Equal probabilities are ranked by label for deterministic top-k results.

The report includes top-1 and requested top-k accuracy, natural-log multiclass log loss, the unnormalized multiclass Brier score, mean top-label confidence, equal-width top-label ECE, per-class support/prediction/probability summaries, and the count of records whose expected class received zero probability. Log loss floors zero expected probabilities at `1e-15` by default so the JSON report remains finite; the floor and probability-sum tolerance are explicit settings in the output.

Exit code `0` means every configured minimum and maximum passed, `1` reports structured threshold failures, and `2` identifies invalid JSON, inconsistent labels, malformed probabilities, unsafe output aliasing, or invalid configuration. Brier score here is the sum across classes and ranges from `0` to `2`; it is not divided by the number of classes. Probability quality on one labeled dataset does not establish calibration after class, data, or model drift, and thresholds should be chosen on separate development data.

## Multi-label classification evaluation

When one record can have several correct labels, evaluate exact JSON arrays instead of flattening the task into a single class:

```powershell
python -m ai_eval_micro_lab.multilabel examples/multilabel-evaluation.jsonl `
  --min-micro-f1 0.70 --min-macro-f1 0.50 --max-hamming-loss 0.20
```

Each JSONL object needs `expected` and `predicted` arrays containing unique, non-empty, case-sensitive string labels. Individual arrays may be empty, but at least one label must appear somewhere in the dataset. Label order does not affect the result.

The report includes per-label true/false positives and negatives, precision, recall, and F1. Aggregate results include micro, macro, and support-weighted metrics; exact-set subset accuracy; sample-averaged Jaccard similarity; average expected and predicted label counts; and Hamming loss normalized by records times the observed label vocabulary. A label produced only by the model remains visible with zero expected support. An empty expected/predicted pair contributes a sample Jaccard score of `1`.

Exit code `0` means the configured F1 minimums and Hamming-loss maximum passed, `1` reports every structured threshold failure, and `2` identifies invalid JSON, duplicate labels, malformed arrays, or invalid configuration. These metrics evaluate already-selected label sets. Choose model score cutoffs and quality gates on separate development data when possible; rare labels, incomplete annotations, and label dependence can make one dataset's macro and Hamming results misleading.

## Ranked retrieval evaluation

Evaluate search or retrieval-augmented generation candidates with binary relevance labels at several cutoffs:

```powershell
python -m ai_eval_micro_lab.retrieval examples/retrieval-ranking.jsonl `
  --cutoff 1 --cutoff 3 --gate-cutoff 3 `
  --min-mrr 0.65 --min-recall 0.90 --min-ndcg 0.75
```

Each JSONL object needs a unique string `query_id`, a non-empty list of unique `relevant` document IDs, and an ordered list of unique `retrieved` IDs. The report includes Hit Rate, mean reciprocal rank, mean recall, and mean normalized discounted cumulative gain at every configured cutoff, plus each query's first relevant rank. Use the field-name flags when an export uses different keys.

Exit code `0` means all minimums passed at `--gate-cutoff`, `1` reports structured metric shortfalls, and `2` identifies invalid JSON, duplicate IDs, empty relevance judgments, or invalid configuration. These metrics assume binary and complete relevance judgments; unjudged documents, position bias, or a small query set can distort offline results and do not establish downstream answer quality.

## Dataset overlap audit

Accidental reuse between training, development, and evaluation exports can make an offline score misleading. Compare two JSONL datasets by stable record ID and text:

```powershell
python -m ai_eval_micro_lab.overlap `
  examples/overlap-reference.jsonl examples/overlap-candidate.jsonl `
  --max-overlap-rate 0.20 --max-details 25
```

The audit reports raw exact matches separately from matches found after Unicode NFKC normalization, case folding, and whitespace collapsing. Counts include every matching pair and every distinct record involved, while `--max-details` bounds the ID-only match list. Source text is not copied into the JSON report.

Exit code `0` means the candidate-record overlap rate is within the configured maximum, `1` reports a threshold failure, and `2` identifies invalid JSON, duplicate IDs, blank text, or invalid configuration. Use `--id-field` and `--text-field` for different schemas. This is a deterministic exact-key audit, not a semantic or near-duplicate detector; paraphrases and differently tokenized content can remain undetected, and an overlap rate alone does not prove whether reuse was improper.

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

## Group performance disparity audit

Per-slice averages are useful for exploration, but a release check often needs one explicit statement about the weakest group and the largest observed gap. The disparity command evaluates normalized exact-match accuracy for every named group and attaches a Wilson interval to each finite-sample rate:

```powershell
python -m ai_eval_micro_lab.disparity examples/group-disparity.jsonl `
  --min-group-count 2 `
  --min-worst-group-accuracy 0.50 `
  --max-accuracy-gap 0.35
```

The report keeps every group in deterministic order, identifies tied best and worst groups, and reports overall accuracy, minimum group size, worst-group accuracy, and the best-minus-worst accuracy gap. Exit code `0` means every configured gate passed, `1` reports structured sample-size or performance failures, and `2` identifies malformed data, invalid thresholds, oversized input, or unsafe output aliasing. Custom field names support existing evaluation exports.

These are descriptive exact-match differences in the supplied finite dataset, not a fairness certification or a causal analysis. Group definitions may be incomplete or sensitive, Wilson intervals cover binomial sampling uncertainty only, repeated cases may be dependent, and a small or unrepresentative evaluation set can hide important harms. Choose group definitions and gates before reviewing final results, and investigate context rather than treating a passing gap as proof of equitable behavior.

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
