# Requirement scenarios, paired benchmarks and predictor adapters

This stage supplies reproducible evaluation and learner interfaces. No production
training or learned-performance claim accompanies it. The metric definitions are
in [evaluation-contract.md](evaluation-contract.md).

## Dedicated scenarios

| CLI scenario | Mechanism | Requirement tested |
| --- | --- | --- |
| `frequency-agile` | An emitter hops through a seeded permutation of eight bands. | Frequency-time scheduling with changing carrier frequency. |
| `spatial-scan` | Two fixed-frequency emitters become visible during periodic beam crossings with unequal revisit periods. | Observation opportunities varying as a spatial beam scans. |
| `periodic-scan` | A short visibility window repeats every sixteen steps; phase is randomized. | Periodic interception and possible phase locking with a fixed sweep. |

`SpatialScanningEmitter` uses a rectangular visibility gate: `(t - phase) mod
scan_period < visible_steps`, intersected with an independent pulse clock. It
enumerates only visible gates instead of generating and discarding invisible pulses.
Truth means signal present at the receiver location. The model does not simulate
antenna patterns or propagation. Existing `ScanningEmitter` remains an adjacent
frequency sweep and is not described as a spatial scanner.

Custom scenarios accept `type: "spatial-scanning"`, with `band`, `scan_period`,
`visible_steps`, optional `pulse_period`, scan `phase` (integer or `"random"`),
`pulse_phase`, power and pulse width. Seeded phases avoid selecting a favorable
alignment for a particular scheduler. Tests verify both blind and successful phases
of a sweep under a perfect receiver; they do not assert that a learned method wins.

```bash
.venv/bin/python -m spectra_scheduler --scenario spatial-scan --runs 10 --workers 2
.venv/bin/python -m spectra_scheduler.policy_benchmark \
  --backend synthetic --split train --seeds 0 1 2 --workers 2 \
  --output reports/generated/requirement-scenarios.json
```

The synthetic benchmark reuses generated truth across policies. It reports the
shared evaluation contract for sweep, shuffled sweep and period-aware scheduling,
with per-scenario summaries. World seeds derive from a versioned split/scenario/seed
namespace. The declared physical duration defaults to 1 ms per abstract step; it
is an experimental scale, not calibration. The Python API accepts another duration.

## Frozen replay benchmarks

```bash
.venv/bin/python -m spectra_scheduler.policy_benchmark \
  --split train --files 3 --seeds 0 1 --workers 2 \
  --plan reports/generated/train-reference-plan.json \
  --output reports/generated/train-reference-comparison.json
```

The first invocation saves an exact seeded sample of recordings, content SHA-256
hashes, split and seeds. Subsequent runs reuse it; CLI split/seeds must match.
Plans reject duplicate files, split violations, path escapes, changed contents and
insufficient requested files. Content is rechecked after comparison. Receiver and
interface settings are embedded in results; use identical settings for paired runs.
Keep test plans for final evaluation after model selection is frozen.

Every `(recording, seed)` executes every policy with the same receiver randomness.
Policy factories create fresh instances. Errors fail the comparison rather than
silently dropping unsuccessful policies or files. Factories must be importable and
spawn-pickleable for multiprocessing; use module-level classes or `functools.partial`.

```python
from spectra_scheduler.policy_benchmark import PolicySpec, benchmark_policies

policies = [
    PolicySpec("baseline", BaselinePolicy, "baseline configuration v1"),
    PolicySpec("candidate", CandidatePolicy, "checkpoint content hash and configuration"),
]
report = benchmark_policies(root, plan, policies, baseline="baseline", workers=2)
```

Arbitrary policies implement the existing `reset(specification, seed)` / `act(obs)`
contract. Provenance is required. `trained_on` records training-file content hashes;
validation/test overlap is rejected when provenance is supplied. The framework
cannot establish the training history of an arbitrary user-written factory; users
must supply honest provenance. `predictor_spec()` registers checkpoint hashes and
training provenance automatically and rejects checkpoint replacement during a run.

Reports retain episode outcomes and policy latency, plus means and paired deltas
for all common numeric metrics. Seeds are averaged **within each recording** before
recording-level bootstrap intervals; repeated seeds are not independent recordings.
Synthetic groups are distinct scenario/seed worlds. Per-scenario summaries expose
regressions that pooled means can hide. Missing metrics remain null and group counts
are reported. Intervals are descriptive, not multiple-comparison-adjusted evidence.

Workers run independent episodes with at most `2 * workers` pending tasks, deterministic
result order, and single-threaded native math in spawned processes. Bootstrap
resampling is vectorized in bounded chunks. Result storage scales with the number
of episode summaries, not pulse trajectories. Workers multiply per-environment
memory; begin with 2–4 and profile before increasing them. `workers=1` respects the
caller's thread configuration. The spawning helper temporarily sets native thread
environment variables; invoke it from the controlling thread, not concurrent threads.

## Predictor and training interfaces

`replay_training.training_batches` accepts **only a train manifest**, a behavior
factory, receiver/interface configuration and `BatchConfig`. It streams pre-action
histories into preallocated NumPy buffers. Each batch carries action, timing class,
capture ratio and validity mask, reward, elapsed seconds, physical-time discount,
termination and timing-bin schema. Histories reset at episode boundaries. Labels
are evaluator-derived after the action and never enter behavior-policy inputs.
Empty-truth ratios are masked; no-intercept windows have their own timing class.

Buffers are borrowed: consume or copy a batch before requesting the next one.
Use `contextlib.closing` if stopping early. This adapter is synchronous and bounded;
it does not promise overlapping data collection and GPU work. It can feed different
learners without changing feature extraction or truth semantics.

`UniformActionPolicy` explores all bands **and dwell choices**. Fixed-dwell sweep
trajectories alone do not cover the complete action space. Action selection affects
training coverage; the adapter does not treat unobserved actions as labeled misses.

`forecast_model.ForecastNetwork` is an optional reference architecture: a small GRU
over a fixed, zero-padded observation history, producing all action outputs in one
batch. Timing has event bins plus a no-event category. Hit probability derives from
the timing distribution; the ratio head predicts a bounded capture ratio. Its
supervised loss is timing cross-entropy plus masked ratio squared error. Actual
intercept time is quantized relative to each action's elapsed window. This model
is an implementation to evaluate, not a selected best method.

`prediction_loss` and `optimization_step` are explicit training hooks, with target
validation, finite-gradient checks and clipping. No training loop is started on
import. Physical-time discounts are retained for future RL algorithms; supervised
loss does not use them. Optimizer resume, curriculum/model selection and full
training orchestration remain the subsequent training stage.

`PredictorPolicy` performs one all-action inference, reconstructs its clock from
public action/receiver semantics, and scores predicted ratio plus coverage age minus
retuning fraction. Both score weights default to 0.05 and are configurable. This
is a declared reference selection rule; it does not guarantee coverage or optimality.
The action clock uses exact configured durations, not float32 feature-time decoding.
When event probability is below 0.5, it emits no finite time estimate; coverage and
restricted timing error expose this abstention in the evaluation contract.

`save_predictor` writes atomic weights-only checkpoints with model dimensions,
feature/action/receiver specification and training hashes. Loading validates schema,
dimensions and finite weights. Inference verifies the entire replay specification
except seed. CPU execution and configurable device/threads are supported; fixed
history and input buffers are reused. GPU outputs are copied once per decision, so
the existing latency measurement includes completion rather than only submission.

The optional learner environment needs the updated `requirements-rl.txt`, including
HDF5 support. No Torch dependency is added to the core package. A checkpoint can
be registered through `predictor_spec` or evaluated using the Torch environment:

```bash
.venv-rl/bin/python -m spectra_scheduler.policy_benchmark \
  --split val --files 10 --seeds 0 1 2 --workers 2 \
  --model artifacts/predictor.pt \
  --output reports/generated/predictor-validation.json
```

The checkpoint must already exist and match the default receiver/interface (or
use the Python API with matching settings). No trained checkpoint is supplied by
this stage. Unit tests use tiny generated examples to verify loss gradients,
serialization, reset parity and the collection-to-inference path; those checks
are not trained-model results.
