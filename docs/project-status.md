# Backend status and dataset role

## What is implemented

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
  search training, with saved artifacts. Available algorithms are not evidence of
  satisfactory final scheduling performance; the latest MPC repair is mixed.

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

## Useful integration paths, not yet implemented

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

## Is the simulation complete?

Both the abstract synthetic simulator and the physical-time PDW replay engine are
implemented and tested. Replay supports continuous-frequency passbands and explicit
dwell/retune semantics, with a real-data full-recording smoke run. It is not a complete
high-fidelity RF or hardware model: calibrated pulse processing, interference,
waveform propagation and hardware integration are not validated. Its explicit model
limitations and recording-horizon assumptions are listed in the replay manual.

## Remaining work beyond a GUI

- A learned scheduler that reliably improves held-out performance, including coverage
  and reacquisition, across multiple seeds and stronger receiver shifts.
- Replay-to-learning integration and perception-to-scheduling state flow.
- Calibration of the receiver model against a specific hardware target.
- End-to-end streaming, latency/resource profiling and long-run robustness at realistic
  pulse rates; native acceleration only where profiling demonstrates a need.
- Frozen acceptance criteria, separate validation/test runs, reproducible comparisons
  and a documented inference/deployment interface.

No evidence supports describing the backend as finished except for its GUI.

## Default ML implementation practices

Use bounded/chunked ingestion, preallocated reusable buffers where shapes are known,
batched inference/losses, suitable parallel workers, cached invariant computations
and minimal device synchronization. Profile before adding native code or concurrency
layers. Check numerical/data parity after optimization and distinguish throughput
improvements from prediction or scheduling improvements. Require held-out learning
evidence before committing substantial training resources.
