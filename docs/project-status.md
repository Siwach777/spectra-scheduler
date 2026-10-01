# Project status and dataset role

## Scope

Spectra Scheduler is a passive, receive-only spectrum scheduling prototype.
[project_scope.md](project_scope.md) defines its requirements. It evaluates heuristic,
Bayesian, reinforcement learning, and model predictive control (MPC) scheduling
policies under instantaneous bandwidth constraints in simulation and pulse replay.

## What is implemented

- Model-independent metrics and selected-action forecast evaluation, shared by
  physical-time replay and a discrete-simulation adapter. Includes prediction
  accuracy/calibration, ratio error, intercept-time error with censoring/coverage,
  time-normalized rewards and explicit unavailable metrics. See the
  [evaluation contract](evaluation-contract.md). Causal timing predictors are implemented.

- Discrete-time passive receiver simulator: periodic, hopping, scanning, burst,
  windowed and mode-switching emitters; retuning delays; sensitivity, missed
  detections, false alarms and measurement noise.
- Shared batch/step engine, seeded procedural train/validation/test worlds,
  scenario files, baseline schedulers and observation-based tracking/change detection.
- Metrics for interception, discovery, reacquisition, retuning and coverage;
  reproducible reports and paired baseline comparison tools.
- Streamed HDF5 pulse ingestion, schema/unit validation, bounded samples,
  per-file parallel offline clustering and label-based evaluation.
- Interactive physical-time stare replay: frequency/bandwidth and dwell actions,
  retuning/slew, sensitivity and seeded missed detections, bounded observations,
  evaluator-only labels and aggregate receiver metrics. See [pulse replay](pulse-replay.md).
- Reusable replay reset/step interface, causal PDW summaries, band/dwell action
  mapping, observable rewards and time-dependent discounts. A shared inference
  runner measures receiver results and policy latency; paired file-level evaluation
  supports CPU workers. See [replay interface](replay-interface.md).
- Experimental learned hit models, DQN, recurrent PPO and recurrent model-based
  search training, with saved artifacts. Physical MPC includes optional Gumbel
  search, elapsed-macro-time input and causal coverage probes. Fresh matched
  Gumbel/PUCT comparisons found substantially better discovery with Gumbel but
  worse capture; neither established learned gains over its own initialization.
  See [MPC assessment](mpc-repair.md). These are experimental policies, not a
  satisfactory final scheduler.
- A trained timing forecaster with receding-horizon action planning, native
  1/10/50-tick listening dwells and public retuning costs. The main CLI accepts
  `--timing-model` and repeatable `--mpc-model` options, with capture/discovery,
  phase-planner controls and paired reporting. The current selected checkpoint is
  `artifacts/timing-refine-v1/seed-0/best.pt` (epoch 24); see
  [timing findings](timing-model-findings.md) and [runbook](runbook.md).
- Experimental Whittle, golden sweep and observation-driven scan handover controls.
  Markov belief updates account for receiver errors and elapsed retuning ticks.
  A nine-world screening run retained the current timing model; these controls
  remain in `experiments.scan_strategy_study`, outside the main CLI defaults.
- A browser interface with synchronized head-to-head comparisons, causal receiver
  traces, playback, model availability and JSON exports. The trained timing policy
  runs on CUDA through the same planner; the [GUI guide](../web/README.md) documents
  setup and the HTTP API. Exported simulation truth is for evaluation and display.

## How the dataset is used now

TSRD now feeds physical-time stare replay as well as offline ingestion and pulse
association experiments, but not MPC training. The saved scan-training audit validated
2,500 files containing 233,172,417
pulses. An HDBSCAN signature-feature experiment scored 100,000 sampled pulses from
10 training files; it is neither a learned scheduler nor held-out operational proof.
The saved stare inspection covered three files, not the full stare collection.

The [official dataset card](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset)
describes synthetic pulse descriptor words with arrival time, centre frequency,
pulse width, angle of arrival and amplitude, in scan and full-spectrum stare modes.
Emitter labels are file-local; they cannot be treated as universal emitter classes.

## Project extensions, not all minimum requirements

1. **Strategy-specific learning adapter:** consume the shared replay features,
   actions and physical-time discounts in the chosen learner. The shared environment
   and inference contract are implemented; old discrete-step MPC checkpoints are not
   automatically compatible with the new feature/action space.
2. **Perception training:** use file-local labels for association/contrastive targets,
   or self-supervised timing/next-pulse prediction, then feed estimated track state
   into the scheduler. Labels must not enter runtime observations.
3. **Simulator calibration:** derive training-only pulse-rate, hopping, dwell and
   signature distributions and validate on separate files/scenarios.

Scan recordings are already censored by their recorded schedule; missing bands are
not negative examples and cannot be reconstructed as complete truth. Stare replay
uses explicit MHz and microsecond units. Dataset amplitude in dB is not assumed to
be calibrated receiver dBm. Replay is PDW-level synthetic-data evaluation, not real
RF validation.

## Simulation architecture

Both the abstract synthetic simulator and the physical-time PDW replay engine are
implemented and tested. Replay supports continuous-frequency passbands and explicit
dwell/retune semantics, with a real-data full-recording smoke run.

## Ongoing software development

- Larger fresh-world assessment of the refined timing checkpoint, including discovery,
  reacquisition, independent training seeds and stronger receiver shifts. Existing
  evidence covers development simulations; small screening runs are not final validation.
- Replay-to-learning integration for end-to-end policy training and evaluation.
- End-to-end streaming, latency/resource profiling and long-run robustness at realistic
  pulse rates.
- Agreed software acceptance criteria, separate validation/test runs, reproducible
  comparisons and clear instructions for running the prototype.

## Default ML implementation practices

Use bounded/chunked ingestion, preallocated reusable buffers where shapes are known,
batched inference/losses, suitable parallel workers, cached invariant computations
and minimal device synchronization. Profile before adding native code or concurrency
layers. Check numerical/data parity after optimization and distinguish throughput
improvements from prediction or scheduling improvements. Require held-out learning
evidence before committing substantial training resources.
