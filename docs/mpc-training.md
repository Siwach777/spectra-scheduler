# Search-driven model training

The collect–train–validate loop is in `spectra_scheduler.mpc_training`. It starts
from a neural model, not a handcrafted scheduling policy. Demonstration initialization
is optional; otherwise all trajectories come from search with learned weights.

## Code layout

Implementation lives in `src/spectra_scheduler/mpc/`:

| Module | Responsibility |
|---|---|
| `config.py` | Model dimensions, reward defaults and training settings |
| `observation.py` | Causal receiver-history encoding and legacy reward helper |
| `model.py` | GRU representation, dynamics, policy and value networks |
| `search.py` | Single-root and batched MCTS |
| `scheduler.py` | Simulator scheduler adapter |
| `data.py` | Demonstrations, parallel search actors and replay storage |
| `learning.py` | Pretraining, n-step targets and multi-step losses |
| `checkpoints.py` | Atomic saves, loads and implementation hashes |
| `evaluation.py` | Validation, fixed probes, test evaluation and benchmarks |
| `trainer.py` | Collect–learn–validate loop and resume orchestration |
| `cli.py` | Argument parsing for existing commands |

`neural_mpc.py` and `mpc_training.py` are compatibility entry points. Existing module
commands, public imports, model state-dict keys and checkpoint versions are unchanged.
New code can import directly, for example `from spectra_scheduler.mpc.model import
NeuralMPCModel`. This is a structural refactor, not a change to the learning algorithm.
Provenance now hashes every implementation module rather than only the entry points.

## Learning loop

1. CPU actors receive latest weights and independent training worlds. Each actor
   batches GRU updates and tree-leaf inference across its worlds.
2. PUCT produces visit-distribution policy targets and root value estimates.
   Dirichlet root noise and sampled actions provide training-only exploration.
   Search backs up discounted edge rewards, truncates at the episode horizon,
   and bounds predicted rewards and values to the configured return range.
3. Bounded replay stores complete observation-only trajectories. Validation and
   test episodes are rejected. Emitter truth is never a loss target.
4. The learner batches full histories and sampled multi-step unrolls on CPU/CUDA.
   Losses cover search-policy cross entropy, n-step value regression, immediate
   reward prediction and latent consistency against an EMA target encoder.
   Terminal targets do not bootstrap; losses are masked beyond episode end.
5. Fixed validation worlds measure reward, interception, discovery and coverage
   for search and policy-only. Separate fixed held-out trajectories track reward
   MSE without changing the diagnostic data after training.
6. Mean search reward across normal and receiver-shift validation selects `best.pt`.
   `latest.pt` always holds the latest model. A worse update never replaces best;
   the best can therefore remain the untrained initialization.

The architecture follows the representation/dynamics/prediction structure of
[MuZero](https://arxiv.org/abs/1911.08265). Latent consistency is related to the
direction explored by [EfficientZero](https://arxiv.org/abs/2111.00210), but this
implementation is not a full reproduction of either paper.

## Train

Run from the repository root:

```bash
# 4,800 episodes and 20,000 learner updates; this is the longer run for you to launch.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-rl/bin/python -m spectra_scheduler.mpc_training \
  --run-dir artifacts/mpc-main --device cuda --workers 6 --threads 2 \
  --episodes 24 --batch-size 32 --iterations 200 --updates 100 --simulations 32
```

Use `--device cpu` if CUDA is unavailable. Six workers with four worlds each is a
starting configuration, not a measured optimum. Actors use one Torch thread each;
`--threads` controls learner threads. More workers reduce each actor's inference
batch size. Collection and learning are synchronous, not overlapped; full GPU
utilization is not promised. Optional warm start: `--initial artifacts/neural-mpc.pt`.

The GRU remains 128 units. This stage changes training, not network capacity. The
29-feature causal encoder contains previous band/outcome, measurements and per-band
hit/age summaries, not the full `Context` representation. The downloaded pulse
dataset is not consumed by this loop.

## Inspect, stop and resume

Each iteration prints a JSON record. `progress.json` contains losses, fixed-probe
reward MSE, validation and elapsed durations, without authored wall-clock timestamps.

```bash
less artifacts/mpc-main/progress.json

# Saved settings are restored; only specify overrides.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-rl/bin/python -m spectra_scheduler.mpc_training \
  --run-dir artifacts/mpc-main --resume --iterations 400
```

Ctrl+C stops the run. Resume restarts the last incomplete iteration. `training.pt`
contains online/target weights, optimizer, replay, diagnostic trajectories, RNG,
history and next episode index. Only total iterations, device, threads and worker
count may change. Changing workers changes future exploration RNG grouping.
Exact continuation is tested on CPU with unchanged settings, not across devices.

Training and inference checkpoints use atomic replacement and restricted
weights-only loading. A directory lock prevents concurrent trainers. Existing runs
are not silently overwritten. `environment.json` records versions and source hashes.

## Evaluate without training

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-rl/bin/python -m spectra_scheduler.mpc_training \
  --run-dir artifacts/mpc-main --evaluate --evaluation-split test \
  --validation-episodes 30 --workers 6
```

This loads `best.pt` with its saved search settings and writes `evaluation-test.json`.
Here `--validation-episodes` sets evaluation sample count. Test uses a separate
procedural namespace and never selects checkpoints. Evaluation compares search and
policy-only on identical worlds; it does not include baselines or confidence intervals.
Reserve test for final evaluation. The older `neural_mpc benchmark` supports baseline
comparisons but differs in finite-horizon search handling; do not conflate results.

## Verification and limits

```bash
.venv-rl/bin/python -m pytest tests/test_mpc_training.py tests/test_neural_mpc.py -q
```

Tests cover reward-sensitive search, terminal targets, replay isolation, gradient
flow, exact CPU resume and test evaluation. Multi-process CPU runs also completed.
CUDA could not be verified here: PyTorch and the external driver check reported no
accessible GPU. No driver or system configuration was changed.

The bounded learning check in `artifacts/mpc-loop-verified` used five iterations,
80 training worlds, 160 updates and four validation worlds per distribution. The
fixed held-out reward MSE rose from 0.1504 to 0.1886. Randomized search interception
fell from 0.1080 to 0.0380, and shifted interception from 0.0888 to 0.0349. Best-model
selection retained iteration zero through that point. One additional iteration
exercised CLI resume and became the best checkpoint: randomized interception rose
to 0.1382 and shifted interception to 0.1882. Fixed-probe MSE remained worse at
0.1863. The same network without search still intercepted more on these validation
worlds (0.2171 and 0.2204). A two-world test execution check produced only 0.0248
and 0.0080 search interception; it is not a statistically useful test benchmark.
Thus optimization and checkpoint selection work, but evidence for generalization
remains mixed. A long training run is not justified by these measurements alone.

Replay samples uniformly; there is no prioritized replay, old-trajectory reanalysis,
asynchronous GPU inference server or stochastic/chance-node dynamics. Deterministic
latent rollouts can be inaccurate under partial observability. Training loss reduction
alone is not scheduling improvement. Short checks do not establish competitive
performance or a SOTA result.
