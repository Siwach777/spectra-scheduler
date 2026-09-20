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
must be evaluated explicitly rather than assumed to improve the mission objective.
Discount is `gamma ** (elapsed_us / reference_us)`, zero at true episode termination.
A variable-duration learner must consume this discount (or deliberately use another
continuous-time objective), not silently substitute one fixed gamma per action.

`specification()` returns serializable versioned feature, action, receiver and reward
settings. Save this alongside strategy checkpoints and call
`validate_specification(saved, current)` in the strategy adapter. The comparison
allows a different episode seed but otherwise requires the same contract, including
reward and receiver settings. Receiver-shift studies need an explicitly approved
alternate specification. Old MPC checkpoints are **not** compatible with the new
74-feature/24-action interface simply because both use a recurrent network.

## Policy inference

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

## Before declaring a trained system ready

The integration infrastructure is implemented; it does not imply operational readiness.
Remaining strategy work includes consuming this interface in the selected learner,
checkpoint compatibility checks in that adapter, training and held-out evaluation.
Learned association/track identity, mission acceptance thresholds, calibrated RF
behavior and production deployment are not supplied by the summary encoder. These
need explicit requirements or evidence; GUI work cannot substitute for them.
