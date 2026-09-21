# Shared ML experiment infrastructure

The `experiments` package manages the lifecycle of an experiment without selecting
its learning algorithm. The reference supervised replay predictor has a complete
adapter. Existing MPC uses the shared atomic artifact writers while retaining its
own training/checkpoint format. Other PPO/MPC lifecycle adapters are future work;
their losses and collection mechanics are not forced through the predictor trainer.

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
using existing causal batches. File order and exploration seeds change reproducibly
per epoch. Train and validation plans must have the correct split labels and
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
  --run-dir artifacts/predictor-pilot --epochs 3 --device cpu --threads 1 \
  --validation-workers 2
```

The example caps each epoch at 64 batches. This is a configurable pilot budget,
not a tuned configuration or a promise of convergence. Set
`max_batches_per_epoch` to `null` for a complete pass through the selected training
recordings. There is no automatic large training launch. CUDA is available through
`--device cuda` and fails explicitly if unavailable. Validation inference runs on
CPU so multiple workers do not load copies onto the GPU.

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
  --run-dir artifacts/predictor-pilot --epochs 6 --resume --device cpu --threads 1 \
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

## Efficiency and verification

Collection uses bounded reusable history/batch buffers. The model predicts all
actions in one forward pass; loss totals accumulate on device and transfer at
epoch boundaries. Progress writes are throttled. Validation uses the existing
bounded process pool with native thread limits. Artifact writes use temporary
files, flush/fsync, then replacement. No full dataset or trajectory archive is
loaded into RAM. Collection and optimization remain synchronous; adding overlap
requires profiling and explicit ownership of borrowed batch buffers.

Tests cover generic best/latest selection, changed configuration and corrupted
artifacts, concurrent writers, failed/interrupted epochs, atomic-write failure,
train/validation isolation, checkpoint restoration, exact CPU resume parity and
the CLI's complete generated-fixture training/resume workflow. These are execution
and correctness checks; actual learning experiments and model comparisons remain
the next stage.
