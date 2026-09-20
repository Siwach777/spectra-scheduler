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
- Experimental learned hit models, DQN, recurrent PPO and recurrent model-based
  search training, with saved artifacts. Available algorithms are not evidence of
  satisfactory final scheduling performance; the latest MPC repair is mixed.

## How the dataset is used now

TSRD currently feeds offline ingestion and pulse association experiments, not MPC
training. The saved scan-training audit validated 2,500 files containing 233,172,417
pulses. An HDBSCAN signature-feature experiment scored 100,000 sampled pulses from
10 training files; it is neither a learned scheduler nor held-out operational proof.
The saved stare inspection covered three files, not the full stare collection.

The [official dataset card](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset)
describes synthetic pulse descriptor words with arrival time, centre frequency,
pulse width, angle of arrival and amplitude, in scan and full-spectrum stare modes.
Emitter labels are file-local; they cannot be treated as universal emitter classes.

## Useful integration paths, not yet implemented

1. **Stare replay environment:** use the full-spectrum pulse stream as hidden
   reference data. Expose only pulses inside chosen time/frequency windows after
   accounting for receiver retuning and observation errors. This supports controlled
   counterfactual schedule evaluation on the recorded full-spectrum scenario.
2. **Perception training:** use file-local labels for association/contrastive targets,
   or self-supervised timing/next-pulse prediction, then feed estimated track state
   into the scheduler. Labels must not enter runtime observations.
3. **Simulator calibration:** derive training-only pulse-rate, hopping, dwell and
   signature distributions and validate on separate files/scenarios.

Scan recordings are already censored by their recorded schedule; missing bands are
not negative examples and cannot be reconstructed as complete truth. Stare replay
also requires explicit MHz-to-band mapping and microseconds-to-simulation-time
handling. Dataset amplitude in dB is not assumed to be calibrated receiver dBm.
Replay would be a PDW-level synthetic-data evaluation, not real RF validation.

## Is the simulation complete?

The current abstract simulator is functional for algorithm experiments. It is not a
complete high-fidelity RF or hardware model. Dataset replay, continuous-frequency
bandwidth/window semantics, realistic physical-time dwell actions, calibrated pulse
processing and scale/performance validation still need work. Interference, waveform
propagation and hardware integration are not validated by this simulator.

## Remaining work beyond a GUI

- A learned scheduler that reliably improves held-out performance, including coverage
  and reacquisition, across multiple seeds and stronger receiver shifts.
- Dataset-driven environment integration and perception-to-scheduling state flow.
- Explicit band/dwell action semantics, unit conversions and calibrated receiver model.
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
