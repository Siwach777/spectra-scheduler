# Reusable replay environment and inference

No training or GUI is required to use these components. Existing MPC files remain
modular and unchanged. Different strategies can share this environment and runner
without sharing network architectures, optimizers or training loops.

| Component | Responsibility |
| --- | --- |
| `pulse_replay.py` | Streaming receiver physics and evaluator-only truth |
| `replay_features.py` | Causal, fixed-size summaries from delivered PDWs |
| `replay_env.py` | Action mapping, reset/step, observable reward, time discount |
| `replay_evaluation.py` | Policy contract, episode runner, paired reference checks |
| Existing `mpc/` modules | Strategy-specific model, search, learning, checkpoints |

## Frozen timing scheduler on external recordings

The trained timing model can now run through this receiver interface. `Transition`
includes `receiver_observation`, containing only the executed window's delivered
PDWs and public timing. Policies implementing `observe_pulses(observation)` receive
this feedback after the action; emitter labels and future arrivals remain evaluator-only.
The original compact feature vector and policy `reset/act` contract remain available.

```bash
.venv-rl/bin/python -m spectra_scheduler.experiments.timing_replay_study \
  --checkpoint artifacts/timing-refine-v1/seed-0/best.pt \
  --legacy-checkpoint artifacts/timing-refine-v1/seed-0/best.pt \
  --plan reports/timing-pdw-validation-plan.json \
  --run-dir artifacts/timing-replay-new --workers 20 --batch-size 20
```

The published plan freezes 32 validation stare recordings, excluding the ten
previously examined files. Without `--plan`, `--max-files` and `--selection-seed`
create a new seeded validation plan. The receiver uses
eight 2250-MHz passbands, 2-ms retuning, 90% detection probability and the model's
1/10/50-ms listening dwells, over each complete 10-second recording. The adapter
requires whole 1-ms ticks and rejects fractional slew/retune timing. It uses actual
PDW timestamps to reconstruct masks and raw counts. Dataset amplitude is not
calibrated dBm, so the power channel is zero and its sensitivity gate is explicitly
disabled. Previously, missing power was interpreted as extremely weak detections:
the selected checkpoint multiplied their counts by approximately 1.6e-14.
`--adapter legacy` reproduces that original capped-count transfer.

| Policy | Mean capture | Mean emitter discovery |
| --- | ---: | ---: |
| Fixed sweep, 50 ms | 10.45% | 95.86% |
| Rate-probe | 51.66% | 84.39% |
| Legacy frozen transfer | 9.85% | 96.40% |
| Corrected frozen forecaster | **43.49%** | **93.35%** |
| Training-only PDW fine-tune | 42.72% | 92.60% |

The corrected forecaster captures 4.16 times the sweep's mean recording-level
ratio. Its paired sweep advantage is 33.04 percentage points [28.19, 38.12].
Rate-probe still captures more; the forecaster discovers more emitters. These are
external **synthetic** recordings, not real RF or hardware-in-the-loop. The adapter
returns actions; externally calibrated intercept-time predictions are not claimed.
See the [current evidence](../reports/timing-pdw-summary.json). The
[initial ten-file report](../reports/timing-replay-summary.json) remains historical.

Fine-tuning used 64 fit and 16 separate development recordings from the training
split, batch 256, 20 preparation workers, eight CUDA epochs and 3.38 GB peak GPU
memory. Epoch 6 improved development capture from 34.89% to 40.21% while retaining
93.21% discovery. Epoch 8 and shorter coverage probes failed the discovery floor.
On the new validation files, the fine-tune's capture difference from the corrected
frozen forecaster was -0.77 points [-2.36, 0.94]. It did not justify replacement.
The test split was untouched; fitting/development hashes are checked against
validation inputs. PDW checkpoints have a distinct version and are rejected by
synthetic CLI/GUI loaders.

Run a new training-only adaptation, then compare its frozen checkpoint:

```bash
.venv-rl/bin/python -m spectra_scheduler.experiments.timing_pdw_refine \
  --checkpoint artifacts/timing-refine-v1/seed-0/best.pt \
  --run-dir artifacts/timing-pdw-new --training-files 64 --development-files 16 \
  --batch-size 256 --workers 20 --epochs 8

.venv-rl/bin/python -m spectra_scheduler.experiments.timing_replay_study \
  --checkpoint artifacts/timing-pdw-new/best.pt \
  --reference-checkpoint artifacts/timing-refine-v1/seed-0/best.pt \
  --plan reports/timing-pdw-validation-plan.json \
  --run-dir artifacts/timing-pdw-comparison-new --workers 20 --batch-size 20
```

`timing_replay_transfer` evaluates missing-power/count ablations on training files;
`timing_pdw_refine --select-policy-only` selects replay probe settings on its
existing training-split development set. `--prepare-only` builds bounded shards
without neural training. Use `timing_replay_study --verify-only --max-files 2` for
serial/batched CUDA integration checks.

## Environment contract

`ReplayEnv(path, receiver, interface)` accepts `ReplayConfig` and `InterfaceConfig`.
Use it as a context manager. `reset()` returns float32 observations. `step(action)`
returns a `Transition` containing observation, reward, terminated, discount and
elapsed microseconds. Reset closes the previous stream. Invalid actions fail before
advancing time. The caller asserts the input is full-spectrum stare data; prefer
`discover_files(root, "stare", split)` over arbitrary paths.

Action indices are `band * number_of_dwells + dwell_index`. Default: 8 bands and
three dwell choices (1, 10, 50 milliseconds), hence 24 actions. Bandwidth times band
count must cover the configured frequency range exactly. These are experimental
defaults, not hardware requirements or optimized settings.

Observations contain nine values per band: visited flag, age relative to episode
duration, last-window hit, log pulse rate, mean amplitude in dataset dB/100,
mean log pulse width, circular angle sine/cosine, and last-window overflow fraction.
Two global values encode episode progress and current band (-1 before first tuning).
Default observation size: 74. Unvisited bands retain a visited flag of zero: absence
of observation is not a negative detection. Each output owns its array. The reusable
workspace avoids rebuilding intermediate state; feature extraction is vectorized.
These summaries are not a learned deinterleaver or emitter-identification system.

Reward is delivered pulse count / `pulse_scale`, minus `retune_cost` times retuning
duration / `reference_us`. It contains no hidden labels or missed-pulse truth. It
does not currently reward emitter fairness and may favor dense bands; reward changes
must be evaluated explicitly rather than assumed to improve the scheduling objective.
Discount is `gamma ** (elapsed_us / reference_us)`, zero at true episode termination.
A variable-duration learner must consume this discount (or deliberately use another
continuous-time objective), not silently substitute one fixed gamma per action.

`specification()` returns serializable versioned feature, action, receiver and reward
settings. Save this alongside strategy checkpoints and call
`validate_specification(saved, current)` in the strategy adapter. The comparison
allows a different episode seed but otherwise requires the same contract, including
reward and receiver settings. Receiver-shift studies need an explicitly declared
alternate specification. Old MPC checkpoints are **not** compatible with the new
74-feature/24-action interface simply because both use a recurrent network.

## Policy inference

Policies may also return `Decision(action, Forecast(...))` to submit predictions
before receiver feedback. Every episode report now includes the versioned
[metrics and prediction contract](evaluation-contract.md), including explicit
missing-prediction coverage for policies returning only an integer action.

Implement two methods:

```python
class MyPolicy:
    def reset(self, specification, seed):
        # Validate checkpoint/interface compatibility and reset recurrent state.
        ...

    def act(self, observation):
        # Return an integer action; no environment or ground-truth access.
        ...
```

Pass the object to `evaluate_policy(path, policy, receiver, interface)`. The policy
owns its device, inference mode and hidden-state handling. The runner measures mean
and maximum action-call latency, elapsed episode runtime and receiver metrics, and
does not retain trajectories. GPU policies must synchronize inside `act` if reporting
GPU execution latency; otherwise timings only measure asynchronous submission.

## Reproducible execution checks

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m spectra_scheduler.replay_evaluation \
  --split val --max-files 3 --workers 2 \
  --output reports/generated/replay-interface-validation.json
```

This pairs sweep and random band selection at the same dwell setting, recording,
receiver and seed. It is an integration benchmark, not a claim of learned performance.
Parallelism is across independent files with spawned CPU processes. Each worker has
bounded replay buffers. Start with 2–4 workers; more workers can saturate disk I/O.
Reports include configurations, paths, per-policy metrics and a recording-level
bootstrap interval for the mean delivery-fraction difference. One recording yields
no interval; empty recordings are excluded from delivery-fraction comparisons.
Multiple stochastic policy seeds and representative recordings are still necessary
for a substantive model benchmark. Output replacement is atomic. No dates are emitted.

The integration smoke run used three validation files (85,240, 414,010 and 190,869
pulses), two CPU workers and both reference policies. Complete episodes including
feature extraction took approximately 0.066–0.160 seconds. This verifies the pipeline,
not model quality; three files are insufficient for a final performance claim.
The generated report is `reports/generated/replay-interface-validation.json`.

## Next integration steps

The integration infrastructure is implemented and validated. Downstream steps include:
- Adapting training-only recording features/rates after the completed frozen timing replay.
- Verifying checkpoint compatibility across feature and action mappings.
- Training and held-out evaluation on diverse pulse traces.
