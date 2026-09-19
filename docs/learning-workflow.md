# Learned scheduling baseline

## What this stage adds

The project can now collect observation-only training examples, fit a hit predictor,
save a portable JSON model and evaluate a frozen model against all thirteen existing
strategies. This is supervised logistic regression with explicit scheduling rules,
not reinforcement learning or an operationally validated receiver controller.

`learned_scheduler.py` provides the history features, artifact schema and inference.
`learning.py` handles bounded sampling, fitting and held-out comparison.
`learning_cli.py` provides train/evaluate commands and atomic report writes.
Inference needs no scikit-learn; training uses its logistic regression estimator.

## Reproduce

```bash
uv sync --extra learning --extra dev
uv run spectra-learn train --runs 100 --seed 0 --max-examples 50000 \
  --output artifacts/hit-model.json
uv run spectra-learn evaluate --model artifacts/hit-model.json --runs 50 --seed 10000 \
  --output reports/generated/learned-held-out.json
```

The equivalent module command is `python -m spectra_scheduler.learning_cli`.
Model artifacts and generated reports remain ignored by Git. Neither includes
wall-clock timestamps. Reports identify their model by SHA-256 fingerprint and retain
training configuration. The artifact records dependency versions and feature order.

## Training contract

- Default training uses mixed, acquisition, tracking and change scenarios, seeds
  0–99, with adaptive-dwell and four-step-dwell collection policies. Starting bands
  vary with seed. Crowded is reserved as an unseen scenario.
- Twelve features describe smoothed band hit rates, recent feedback, observation
  support, observation/hit age, last hit, global hit rate, current-band hit rate,
  tuning distance, staying on the current band, dwell and neighbouring hit rates.
- Features are captured before the target observation. Emitter identities, future
  transmissions, false-alarm labels, absolute step, seed and scenario identifiers
  are unavailable to the predictor.
- Targets are observed hits/misses, including false alarms. Retuning is not labelled
  as a miss. This predicts receiver feedback, not ground-truth transmission activity.
- A seeded uniform reservoir caps feature/label storage at 50,000 examples by
  default. This bounds training examples, not total simulation memory: existing
  simulation results still retain a complete episode.
- Logistic regression uses C=1 and at most 500 iterations, with no class weighting.
  Fitting limits numerical-library threads to one to avoid oversubscription.

## Scheduling and evaluation

The policy finishes retuning and two listening steps before reconsidering its band.
Predicted hit probability is penalised for tuning distance. Maximum dwell is six
listening steps when another band is available. Bands overdue by eighteen steps
receive priority; this is a soft coverage rule, not a hard latency guarantee.

Evaluation rejects overlapping training/evaluation seed intervals. Every policy
sees the same generated truth and seeded receiver model for each scenario/seed.
Reports include every existing scan metric, paired interception deltas and
approximate normal 95% confidence intervals. Intervals are exploratory, with no
multiple-comparison correction. A constant-prior model uses exactly the same
dwell/coverage rules to isolate the contribution of fitted predictions. Brier scores
compare fitted and constant predictions on the same learned-policy observations.

New seeds mostly vary phase and receiver randomness within preset layouts. They do
not establish broad domain generalisation. Crowded offers one additional structural
holdout, not a deployment benchmark. Evaluation currently runs sequentially.

## Initial measured results

The commands above fitted 32,664 listening examples and evaluated fifty held-out
seeds per scenario. Interception is detected transmissions / all transmissions.

| Scenario | Learned | Constant-model wrapper | Strongest existing policy |
| --- | ---: | ---: | --- |
| Acquisition | 9.71% | 9.33% | Transition-band: 11.17% |
| Change | 13.00% | 12.07% | Track-aware: 34.20% |
| Crowded, unseen | 8.57% | 8.57% | Adaptive-dwell: 6.91% |
| Mixed | 6.59% | 7.10% | Adaptive-dwell: 8.55% |
| Tracking | 8.40% | 11.20% | Track-aware: 13.47% |

These are descriptive means, not claims of statistical superiority. The learned
policy is not promoted to the default. Crowded improves over existing strategies,
but the constant-model ablation achieves the same interception: the gain must not
be credited to learning. Predictive Brier score improves against the constant prior
only on change and crowded; the remaining scenarios worsen. Predicting immediate
hits alone does not reliably improve scheduling.

## Checks and remaining work

Regression tests cover pre-observation feature extraction, retune exclusion,
bounded/deterministic sampling, dwell handling, single-band operation, artifact
validation, deterministic fitting/evaluation, split overlap rejection and output
protection. Existing simulator and dataset tests continue to pass.

TSRD remains an independent offline association pipeline; no TSRD validation/test
files were consumed here. Recorded fixed-scan observations cannot directly provide
counterfactual rewards for arbitrary tuning actions. Before connecting these paths,
define dataset-derived emitter/measurement calibration and explicit unit mappings.

Next software work: a stable experiment service/API, job/progress reporting and
artifact management, followed by the GUI. Research work: broaden training layouts,
separate model-selection validation from final evaluation and assess whether richer
temporal prediction improves the policy. Once used for model selection, the reported
evaluation seeds should no longer be described as an untouched final test set.
Native Rust/C++ acceleration remains profiling-driven, not a prerequisite for
streaming large datasets.
