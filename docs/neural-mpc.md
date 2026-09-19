# Neural-MPC experiment

This is an experimental learned-model planner, not an established replacement
for the current schedulers. The design is inspired by
[MuZero](https://arxiv.org/abs/1911.08265), but currently implements demonstration
pretraining only. Search-target self-improvement is not implemented.

## Implemented

- A 128-unit GRU consumes past selected bands, hit/listening outcomes, signal
  measurements and per-band hit-rate/age summaries. It does not consume truth.
- Learned action-conditioned dynamics predict latent state and observed reward.
- Policy/value heads train on demonstrated actions and discounted episode returns.
- Five-step imagined rollouts train latent consistency, reward, policy and value.
  Eight rollout starting points are processed together for lower Python overhead.
- PUCT search is capped at depth five and replans after each real observation.
  Selection uses edge reward plus discounted child value. Predicted rewards are
  clipped to the shared reward's physical range.
- Learned schedulers have no forced sweep, dwell, coverage or retune-retry rules.
  The policy-only ablation uses exactly the same checkpoint without tree search.
- Demonstrations use five baseline policies on eight-band procedural training
  worlds. Evaluation uses the shared paired benchmark on separate validation worlds,
  including receiver shift, interception, coverage, discovery and decision latency.
- Training and evaluation use `RewardConfig(hit=1, retuning=0.05, coverage=0.2)`.
  The recurrent input encoder is custom; `Context` is reused for reward accounting,
  not as the representation input. Existing simulator code is unchanged.

## Commands

Run from the repository root with the existing PyTorch environment.
`--episodes` counts episodes **per teacher**, so 100 means 500 trajectories,
but only 100 distinct worlds, each observed under five teacher policies.

```bash
# Small pipeline check; not a performance-trained checkpoint
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-rl/bin/python -m spectra_scheduler.neural_mpc train --episodes 4 --epochs 2 --output artifacts/neural-mpc-smoke.pt

# Larger experiment to run yourself; CUDA is optional
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-rl/bin/python -m spectra_scheduler.neural_mpc train --episodes 100 --epochs 50 --device cuda --threads 2 --seed 0 --output artifacts/neural-mpc.pt

# Includes all baseline policies and the same network without search
.venv-rl/bin/python -m spectra_scheduler.neural_mpc benchmark --model artifacts/neural-mpc.pt --runs 30 --simulations 50 --suites randomized receiver-shift mixed acquisition tracking change crowded --output reports/generated/neural-mpc.json

.venv-rl/bin/python -m pytest tests/test_neural_mpc.py -q
```

Weights and metadata are saved as `.pt` and `.json`; loading uses PyTorch's
restricted weights-only loader. Artifacts and generated reports remain ignored.

## Limits and interpretation

Initial CPU smoke results: 20 teacher trajectories, two epochs, three validation
worlds per suite, 16 search simulations per decision. Interception fractions:

| Suite | Search | Same network without search | Adaptive dwell |
|---|---:|---:|---:|
| Randomized | 0.0227 | 0.1111 | 0.1214 |
| Receiver shift | 0.0303 | 0.0814 | 0.0768 |
| Tracking | 0.1444 | 0.3111 | 0.1333 |

Search took approximately 2.2 milliseconds per decision versus 0.067 milliseconds
for policy-only, excluding observation processing. These small-sample results
do not establish a ranking; they specifically do not demonstrate planning gains.
The complete local report is `reports/generated/neural-mpc-smoke.json`.

The smoke run checks execution, not convergence or superior performance. Training
is currently one complete trajectory per update, with vectorized rollout starts;
there is no parallel environment collection, episode minibatching, resume support,
automatic validation checkpoint selection or MCTS self-improvement yet.

Heuristic demonstrations are not optimal labels. Offline data do not cover every
counterfactual action, so MCTS can exploit model errors. Deterministic latent
dynamics also approximate a partially observed, stochastic environment. Compare
search against policy-only and strong baselines across multiple training seeds
before treating it as an improvement. The fixed-depth planner bootstraps values;
it does not model episode termination or carry an exact remaining-time state.

This experiment uses simulator trajectories, not the downloaded pulse dataset.
That dataset alone does not supply action-conditioned scheduling rewards.
GPU utilization is not guaranteed for these small recurrent networks; CPU may be
faster for sequential inference. No SOTA or optimal-action claim is established.
