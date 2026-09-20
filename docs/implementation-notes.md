# Implementation notes

This is a brief record of what exists and why it was added. Detailed measurements
remain in [experiments.md](experiments.md).

| Area | What is implemented | Purpose |
|---|---|---|
| Foundation | Python package, tests, working plan, and terminal comparison | Keep the first prototype reproducible and focused on simulation |
| Environment | Periodic, hopping, scanning, burst, jittered, windowed, and mode-switching emitters | Represent both stable and changing RF activity |
| Receiver errors | Configurable detection probability and false alarms | Avoid testing schedulers with perfect feedback |
| Fair evaluation | Randomized phases and repeated seeded comparisons | Prevent fixed timing from favoring one scan pattern |
| Truth separation | Public observations contain no emitter identity or false-alarm label | Prevent future policies from using hidden simulator information |
| Metrics | Interception, discovery, coverage, delay, sensitivity, and reacquisition | Keep scheduler, receiver, and change-response effects separate |
| Association metrics | Truth-only V-measure, pairwise scores, purity, mixed tracks, and fragmentation | Diagnose clustering quality and both failure directions without exposing labels to the scheduler |
| Baselines | Fixed, random, shuffled, revisit, UCB, sliding UCB, period-aware, fixed-dwell, and adaptive-dwell scans | Compare simple ideas before adding a larger learning model |
| Dynamic behavior | Emitters can enter, leave, and change operating modes | Test whether learned behavior becomes stale |
| RF strength | Received power, sensitivity threshold, and measurement noise | Separate weak-signal loss from choosing the wrong band |
| Signal measurements | Detected pulses expose independently seeded noisy power and pulse width without emitter IDs | Provide realistic association inputs while preserving the truth boundary |
| Signal tracks | Similar measurements form active tracks; tighter matching can reconnect a recently expired track | Preserve motion history while limiting accidental merges in crowded traffic |
| Track-aware scheduler | Adaptive dwell acquires signals; three associated measurements confirm a track before it guides tuning | Reduce short false track-follows while retaining explicit miss and coverage recovery |
| Retuning | Band changes can use a fixed delay or a delay based on band distance; repeated selections mean dwell | Account for hardware cost without complicating the scheduler interface |
| Adaptive dwell | A sweep waits through retuning and extends productive bands after a hit | Trade broad coverage against longer observation of useful bands |
| Bayesian scheduler | Decaying per-band hit beliefs with exploration and switching cost | Add an inspectable probability-based policy before larger learning models |
| Transition scheduler | A decaying table scores bands that followed recent detected bands | Test whether anonymous hit-to-hit motion is useful before adding signal tracking |
| Change detection | Per-band older and recent hit windows trigger a local belief reset | React to sustained behavior shifts without using simulator change labels |
| Focused scenarios | Separate acquisition, adjacent tracking, mode-change, and crowded association cases | Measure each behavior without hiding it inside the mixed scenario |
| Performance | Shared scenario truth and parallel independent runs | Support larger experiments before considering native code |
| Experiment reports | Versioned JSON/CSV files combine scheduler and association summaries without timestamps | Make results reproducible and easy to compare outside the terminal |
| Command interface | Package command, module entry point, and compatibility script share one argument parser | Keep one tested path for running and exporting experiments |
| Scenario files | Strict JSON definitions support all emitter models, seed offsets, nested modes, and self-contained reports | Run new experiments without editing package code |
| Dataset download | Resumable scan/stare download script with configurable workers, high-performance mode, and a duplicate-run lock | Fetch current splits without archived copies or committing dataset files |

New work should add or update one short row here describing both the change and its
reason.

Dataset ingestion now validates completed TSRD HDF5 files, canonicalises feature
order, streams full-file statistics and draws reproducible bounded samples. Labels
remain separate from measurements. See [dataset-workflow.md](dataset-workflow.md).

Offline association evaluation now offers raw-PDW and scaled-signature HDBSCAN
baselines, deterministic per-file samples, file-local clustering metrics and both
noise-scoring conventions. The dataset command supports process-level parallelism,
optional profiling and atomic JSON reports. Real scan training files and larger
stare files were exercised without touching the validation or test splits.

The first trained scheduler now uses bounded observation-only sampling and logistic
hit prediction, with portable JSON models and explicit dwell/coverage rules.
Held-out evaluation reuses all existing baselines and adds a constant-model ablation
to distinguish prediction gains from scheduling-rule gains. Existing defaults are
unchanged because the fitted model does not consistently outperform them. Commands,
measured results and limitations are in [learning-workflow.md](learning-workflow.md).

Direct-action reinforcement learning adds a neural Double-DQN reference, randomized
train/validation/test worlds and paired benchmarks with untrained-model controls.
Three reference training seeds show gains on randomized layouts but poor coverage
and uneven legacy performance. The main training path now uses SB3-Contrib recurrent
PPO, a Gymnasium adapter over the shared receiver step engine, CUDA optimization,
parallel rollout workers and resumable checkpoints. Larger-model results remain
unproven until benchmarked. See [rl-training.md](rl-training.md) and
[rl-benchmark.md](rl-benchmark.md). The organic-development restriction no longer applies.

Search-driven Neural-MPC adds batched tree inference, parallel actors, observation-only
replay, search policy targets, n-step returns, EMA latent consistency, GPU-capable
minibatches and resumable optimizer/replay checkpoints. Fixed validation selects the
best model; independent diagnostic trajectories track reward-prediction error. Test
evaluation does not train or select checkpoints. See [mpc-training.md](mpc-training.md)
for commands and verification. Competitive performance is not yet established.

Neural-MPC is now split into focused modules under `spectra_scheduler.mpc`: model,
observation encoding, search, scheduler adapter, data/replay, losses, checkpoints,
evaluation, trainer and CLI. The former single-file entry points re-export existing
APIs, preserving commands and checkpoint formats. Seeded collection/search/loss
outputs were checked for exact parity before and after the split; source provenance
now covers the full implementation package.

MPC training now displays live foreground collection/update progress, current losses,
stage ETA, validation metrics and checkpoint-save messages. Parallel worker waits
refresh the display without changing replay ordering. Full JSON history remains in
`progress.json`; there is no need to follow a background log to see training activity.

Fixed descriptor exhaustion in parallel collection: weight snapshots and actor results
now cross process queues as arrays, not shared Torch tensor storages. Replay keeps
local tensors without retaining IPC file handles. A low-limit spawned-worker regression
verifies bounded descriptor usage and exact serial/parallel data parity. Checkpoint
format and training behavior are unchanged.

The MPC learning repair adds training-only temporal exploration, normalized search
scores, auxiliary hit/listening prediction and explicit collapse diagnostics. It also
preallocates rollout/staging buffers, caches targets and packs GRU weights. Old model
inference remains supported; training semantics are versioned separately. The bounded
control comparison improves observation collection and prediction but not both
validation distributions; see [mpc-repair.md](mpc-repair.md). Dataset integration and
remaining backend scope are documented in [project-status.md](project-status.md).
