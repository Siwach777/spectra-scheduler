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
  --run-dir artifacts/timing-replay-new --max-files 10 --workers 20 --batch-size 20
```

This freezes ten validation stare recordings before comparison. The receiver uses
eight 2250-MHz passbands, 2-ms retuning, 90% detection probability and the model's
1/10/50-ms listening dwells, over each complete 10-second recording. The adapter
requires whole 1-ms ticks and rejects fractional slew/retune timing. It uses actual
PDW timestamps to reconstruct listening masks and counts, capped at four per tick.
Dataset amplitude is not calibrated dBm, so the pretrained power channel is zeroed.
It does not claim calibrated forecasts on these recordings or train on validation data.

Mean capture/discovery across ten independent files was 10.19%/96.15% for the
frozen model, 10.91%/94.79% for the 50-ms sweep and 43.97%/82.68% for the causal
rate-probe control. The model's capture difference from sweep was -0.72 percentage
points, interval [-1.60, 0.57]; this establishes no capture gain. The full loop is
implemented, but simulation gains do not automatically transfer to external PDWs.
These are external **synthetic** recordings, not real RF or hardware-in-the-loop.
See the [public summary](../reports/timing-replay-summary.json) for hashes and intervals.
Serial and batched CUDA counts agreed on two separate recording windows.

Recording-specific feature calibration and training-only adaptation remain necessary
before claiming a robust transferred model. Use `--verify-only --max-files 2` for
the serial/batched integration check.

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
