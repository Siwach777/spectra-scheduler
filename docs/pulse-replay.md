# Physical-time pulse replay

The simulator now has two backends: the existing synthetic discrete-step worlds,
and a streaming PDW receiver for full-spectrum TSRD **stare** recordings. The latter
supports interactive frequency/dwell actions without loading a recording into RAM.
It is a PDW-level simulator, not a calibrated waveform or hardware model.

## Run

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m spectra_scheduler.replay_cli \
  --split train --file-index 0 --bands 8 --bandwidth-mhz 2250 \
  --dwell-us 10000 --retune-us 100 --stop-us 10000000 \
  --output reports/generated/stare-replay.json
```

This runs a fixed sweep, not an ML policy. `--help` lists receiver, memory and
detection settings. Only completed files in the selected stare split are discovered.
The same command with `--split val` or `test` evaluates a separate split; keep test
data out of training and tuning. A file index refers to the sorted discovered paths;
the output records the actual path. No downloads or dataset modifications occur.

## Interactive interface

```python
from pathlib import Path
from spectra_scheduler.pulse_replay import DwellAction, PulseReplay, ReplayConfig

with PulseReplay(Path("data/tsrd/stare/train_stare/config_0.h5"),
                 ReplayConfig(), source_mode="stare") as env:
    while not env.done:
        observation = env.step(DwellAction(center_frequency_mhz=1125, dwell_us=10000))
        # Feed observation.pulses to a perception model or scheduler here.
    metrics = env.report()  # evaluation only; never give truth metrics to the policy
```

Creating another environment starts a new episode. `step` returns only receiver
timing, center frequency, observed PDWs and overflow count. PDW columns are ToA in
microseconds, frequency in MHz, width in microseconds, angle in degrees, amplitude
in dataset dB units. Amplitude is **not** silently reinterpreted as calibrated dBm.
Labels remain evaluator-only and file-local. Existing abstract schedulers and MPC
are unchanged; they are not automatically trained or evaluated on this new interface.

## Receiver contract

- An action requests positive listening dwell after retuning. First tuning is free;
  keeping the same center has no retune delay. A change costs `retune_us`, or the
  greater of that and frequency distance / `slew_mhz_per_us` when slew is configured.
- Time intervals and passbands are half-open. A pulse must arrive while listening,
  lie inside the passband, and finish by the dwell end. Partial pulses are rejected,
  including those crossing adjacent same-frequency dwell windows. This conservative
  boundary rule is explicit; future coherent carryover would be a different model.
- Actions clip to the episode horizon. A horizon reached while retuning returns an
  empty observation. There is no negative-time or out-of-range action fallback.
- Optional sensitivity uses the recorded dB scale. Detection probability uses a
  seeded per-row random draw, independent of chunk size and schedule for paired runs.
- At most `max_observation_pulses` PDWs are delivered per action, earliest first.
  Additional interceptions are counted as overflow, not silently lost. This is an
  observation buffer cap, not a calibrated hardware processing-rate model.
- Truth counts include arrivals within the elapsed episode and configured total
  frequency range, even during retuning. Interception and delivered fractions have
  this same denominator. Discovery uses delivered pulses only; discovery delay is
  conditional on discovered emitters. Unlabelled files have no emitter metrics.
- HDF5 processing is chunked and masks are vectorized. A preallocated observation
  workspace is reused; returned bounded arrays own their data. Peak working storage
  scales with chunk size and observation capacity, not trace length (plus per-emitter
  metric state). The caller must not accumulate all returned observations unboundedly.
- Only consumed chunks are validated. Context-manager exit closes the HDF5 stream
  even on early stop and checks whether the input file changed. Choose an episode
  horizon within the recording's known acquisition interval; last pulse time alone
  cannot establish recording coverage.

## Verification and remaining scope

Tests cover an independent in-memory reference, chunk-size parity, deterministic
detection, label isolation, duplicate arrivals, sensitivity, overflow, timing and
frequency boundaries, slew, empty/unlabelled traces, invalid input and stream closure.
A local full 10-second training trace processed 812,717 pulses in approximately
0.13 seconds with the default sweep: 100,909 interceptions and 47/48 emitters
discovered, with no observation overflow. This is a single-file CPU smoke measurement,
not a general throughput guarantee or learned-model result.

Not modelled here: waveform propagation, pulse collisions, false detections,
measurement noise beyond recorded features, calibrated RF sensitivity and hardware
capture. Reacquisition after a known emitter mode change remains a synthetic-world
metric because these replay inputs do not provide that event annotation. Perception
and learned-scheduler integration, held-out acceptance benchmarks and deployment
remain project work; they do not require adding a GUI first.
