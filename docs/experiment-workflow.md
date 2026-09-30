# Shared ML experiment infrastructure

The `experiments` package manages predictor runs without selecting their learning
algorithm. MPC and recurrent PPO use separate training loops and checkpoint formats;
their evaluations share the same paired-world principle.

## Boundaries

| Shared component | Responsibility |
| --- | --- |
| `storage.py` | Atomic JSON/Torch writes, content hashes, checkpoint verification, single-writer run lock |
| `runner.py` | Run configuration, progress, validation cadence, best/latest selection, checkpoint publication and resume |
| Existing replay/benchmark modules | Causal observations, bounded collection, splits, metrics and paired inference |
| `predictor.py` adapter | Reference model, optimizer, epoch sampling, algorithm-specific state and validation invocation |

New learners implement `configuration`, `train_epoch`, `save_state`, `load_state`,
`export_policy` and `validate`. The runner accepts their validation report and a
metric key path, comparison direction and minimum improvement. Adapters own their
serialization format and RNG state; the runner stores and verifies opaque artifacts.
The generic runner/storage code does not import Torch unless Torch serialization
is explicitly requested. A JSON-only test learner verifies that independence.

The reference adapter trains from hits/misses and evaluator-only forecast targets
using existing causal batches. A frozen training cache is shuffled reproducibly
each epoch. Train and validation plans must have the correct split labels and
disjoint file contents. Test plans are forbidden for training/checkpoint selection.
Reference sweep/random policies and the candidate are evaluated on identical
validation traces/seeds. No test data is consumed by this stage's implementation
checks, which use generated fixtures rather than downloaded held-out recordings.

## Prepare an explicit pilot

Create frozen manifests once from the repository root:

```python
from spectra_scheduler.policy_benchmark import make_plan, write_json

write_json("reports/generated/predictor-train-plan.json",
           make_plan("data/tsrd", "train", max_files=20, seeds=(0, 1)))
write_json("reports/generated/predictor-val-plan.json",
           make_plan("data/tsrd", "val", max_files=10, seeds=(0, 1)))
```

Then use the optional Torch environment:

```bash
.venv-rl/bin/python -m spectra_scheduler.experiments \
  --config examples/predictor-experiment.json \
  --run-dir artifacts/predictor-pilot --epochs 3 --device cuda --threads 1 \
  --validation-workers 2
```

The example caps each epoch at 64 batches. This is a configurable pilot budget,
not a tuned configuration or a promise of convergence. Set
`max_batches_per_epoch` to `null` for a complete pass through the selected training
recordings. Training requires CUDA and fails explicitly if unavailable. Neural
validation batches independent episodes on the GPU with shared weights and
separate observation histories. CPU execution through the Python API is retained
for small deterministic unit checks.

All paths inside the JSON config are relative to the config file. Receiver and
interface configuration can be supplied there; defaults match the existing replay
contract. Unknown fields fail. The default checkpoint-selection metric is mean
validation interception ratio, maximized. That selection does not guarantee good
discovery, coverage or timing: assess the full validation report before promoting
a model, and declare a different selection rule before starting a new experiment
when those tradeoffs require it.

## Artifacts and resume

- `run.json`: immutable semantic configuration, data plans, implementation hashes
  and dependency versions.
- `progress.json`: throttled live phase, counts and elapsed duration; no calendar
  timestamp. Foreground output reports the same phase changes.
- `state.json`: the authoritative latest and best checkpoint references and hashes.
- `checkpoints/epoch-.../policy.bin`: exported inference artifact (Torch for this adapter).
- `checkpoints/epoch-.../training.bin`: model, optimizer, counters and RNG state.
- `checkpoints/epoch-.../validation.json` and `training.json`: full validation results
  and training summary for that committed epoch.

Epoch zero validates and saves initialization before updates; training must earn
an improvement. Ties retain the earlier best checkpoint. Every completed epoch is
validated and saved. All files are written before one atomic `state.json` update
publishes the latest/best pair. Failed validation, interruption or partial writes
leave the previous committed epoch intact. Resume checks hashes for both latest
and best artifacts. Orphan checkpoint directories remain unreferenced and are
retained for inspection; automatic pruning is not implemented. Disk use grows
with retained checkpoints, while collection memory remains bounded.

```bash
.venv-rl/bin/python -m spectra_scheduler.experiments \
  --config examples/predictor-experiment.json \
  --run-dir artifacts/predictor-pilot --epochs 6 --resume --device cuda --threads 1 \
  --validation-workers 2
```

`--epochs` is the desired total, not an additional budget. Data, model settings,
selection rule, code hashes, library versions, device and training thread count
must match. Validation worker count may change. Interrupted work within an epoch
is discarded and that epoch is replayed from the previous committed state.
Ctrl+C and CLI SIGTERM handling record interruption where possible; an abrupt
process kill still releases the POSIX advisory lock. If initialization never
committed, start a new run directory after resolving its failure.

CPU tests verify resumed model weights and Adam state exactly match uninterrupted
execution under the same configuration. GPU state is saved/restored, but bitwise
CUDA reproducibility across hardware or kernels is not claimed. The lock uses
POSIX `fcntl` and currently targets the Linux development environment.

To resolve the best model for the existing benchmark API:

```python
import json
from pathlib import Path
from spectra_scheduler.experiments.storage import checkpoint_path, verify_artifacts
from spectra_scheduler.forecast_model import predictor_spec

run = Path("artifacts/predictor-pilot")
best = json.loads((run / "state.json").read_text())["best"]
verify_artifacts(run, best)
policy = predictor_spec(checkpoint_path(run, best, "policy.bin"))
```

## Full predictor comparison

The study builds one content-hashed training cache from 128 train recordings and
two exploratory receiver seeds. It trains MLP, GRU and causal TCN encoders under
three independent model seeds, with eight full passes over 131,328 unique causal
examples per model. The 16 selection recordings choose each run's checkpoint by
the harmonic mean of capture and discovery. Another 64 validation recordings are
reserved for paired reporting; test recordings are not used in this study.

```bash
.venv-rl/bin/python -m spectra_scheduler.experiments.study \
  --directory artifacts/predictor-study
.venv-rl/bin/python -m spectra_scheduler.experiments.assess \
  --study artifacts/predictor-study
```

`study --evaluate-only` recomputes reporting from verified frozen checkpoints.
The cache is collected with bounded CPU processes and memory-mapped shards; a
small enough cache is held on CUDA across epochs, otherwise two pinned host
buffers pipeline transfers. Neural evaluation batches independent episodes on
CUDA. The assessment adds three probability-averaged ensembles and a short-probe,
long-exploitation observed-rate control without using reporting data to tune them.

The completed development comparison in `artifacts/predictor-study` found mean
capture of 0.422 for MLP, 0.472 for GRU and 0.455 for TCN across training seeds.
MLP seed captures ranged from 0.328 to 0.524, while GRU ranged from 0.451 to
0.489. The GRU ensemble reached 0.510 capture and 0.930 discovery; the stronger
rate-probe control reached 0.597 capture and 0.801 discovery. The GRU ensemble's
paired capture difference from rate-probe was -0.087 with a 95% recording bootstrap
interval of [-0.114, -0.055]. One reporting recording had no transmissions, so
capture summaries use 63 independent recordings. These are development results,
not a claim that the predictor is the strongest capture scheduler.

## Dense action-value study

The first predictor study's supervised loss improved while checkpoint-selection
scheduling quality fell. The MLP consumed only the last feature frame, and its
three runs spent far more actions on short dwells than the stronger rate-probe
control. The dense study instead indexes each *training* stare recording and
labels all 24 legal next actions at every causal behavior decision. Its index
checks the chosen action against the replay engine, including retuning time,
captured pulses, truth counts and first-hit bins. Uniform and rate-probe
behaviors provide different observation histories; validation and reporting
recordings are never indexed for training.

One shared-band model predicts nonnegative, dwell-monotone pulse counts from
the full 16-frame history. It trains with robust count regression for all 24
actions and a decision-focused ranking loss for long-dwell band choice, following the
learning-to-rank view of [decision-focused learning](https://proceedings.mlr.press/v162/mandi22a.html).
At inference, the learned policy uses the rate-probe control's short coverage
probes and long exploitation dwells, replacing only its exploitation band
ranking. This paired policy comparison isolates the learned contribution.

```bash
.venv-rl/bin/python -m spectra_scheduler.experiments.dense_study \
  --run-dir artifacts/dense-value-v2 \
  --cache-dir artifacts/dense-value/cache --workers 20 --epochs 100
```

Collection uses bounded processes and a content-hashed, memory-mapped cache.
Training loads the compact action-label tensors on CUDA and selects each of
three model seeds on the 16-recording selection set before the separate
64-recording report. The cache and checkpoint hashes reject a changed source
or manifest on resume. The generated `comparison.json` is a development
comparison, not a test-set result.

The initial 100-epoch attempt exposed a loss-scale defect: ranking gradients
inflated predicted pulse counts by two to three orders of magnitude while
count error rose. The corrected loss ranks long-dwell bands with log predicted
capture rates, so multiplying all counts cannot improve their ranking. A
five-epoch CUDA check showed both losses falling; the corrected run reused the
verified 174,327-example training cache and completed 100 epochs for each of
three model seeds in `artifacts/dense-value-v2`. Selected epochs were 54, 34
and 56. Count error fell from about 0.65 to 0.18, and ranking regret from
about 0.49 to 0.12; a sampled count calibration check found 76 predicted versus
70 actual pulses on average, rather than tens of thousands.

On the 64 separate reporting recordings (63 with transmissions), mean
per-recording capture was 0.616, 0.616 and 0.620 for the three dense models,
versus 0.597 for rate-probe. Their paired capture differences were 0.0188,
0.0196 and 0.0234, with recording-bootstrap 95% intervals [0.0030, 0.0414],
[0.0023, 0.0414] and [0.0038, 0.0505]. These gains are concentrated in sparse
recordings: the strongest seed's median paired gain was 0.0041, and pooled
pulse-weighted capture was 0.5762 versus 0.5756 for rate-probe. Pooled emitter
discovery was 0.815 versus 0.833 for rate-probe. Neural inference averaged
about 1.0–1.2 ms per action versus 0.069 ms for the control. The model is a
valid but narrow development improvement, not a general-purpose winner.
