# MPC learning repair: changes and controlled evidence

## Changes

- Four-step temporal persistence is applied only to exploratory training actions.
  Temperature anneals across collection iterations. Evaluation still chooses each
  step from the network/search without forced dwell, sweeps or coverage rules.
- Batched search normalizes observed Q scores per tree before combining them with
  policy-prior exploration. The old unnormalized search remains selectable.
- A new auxiliary head predicts hit and listening outcomes from imagined latent
  states, trained against receiver observations. It does not consume emitter truth.
- Validation reports dominant-band fraction and bands visited, warning when more
  than 95% of decisions choose one band. These are diagnostics, not reward overrides.
- Rollout arrays and recurrent input buffers are preallocated. Learner host/device
  buffers are reused, n-step targets cached and CUDA copies use pinned host buffers.
  Masked losses avoid per-unroll Python decisions that synchronize the GPU.
  Both online and copied target GRUs explicitly flatten parameters after setup/load.

## Controlled CPU experiment

Both runs used seed zero, 20 iterations, 32 new episodes per iteration, 50 updates
per iteration, batch size 32, four actors, 32 MCTS simulations and 16 validation
worlds per distribution. The control disabled normalized search and observation
loss, held actions for one step and did not anneal temperature. Both retained the
same network initialization for the shared layers and the same optimization work.

Local artifacts: `artifacts/mpc-repair-control` and `artifacts/mpc-repair-trial`.

| Metric at final iteration | Control | Revised |
|---|---:|---:|
| Collection listening fraction | 8.80% | 77.40% |
| Collection hit fraction | 1.51% | 10.76% |
| Fixed-probe reward MSE | 0.1694 | 0.1305 |
| Randomized validation interception | 6.28% | 9.96% |
| Receiver-shift validation interception | 7.86% | 6.15% |

These are final-iteration comparisons, not best-checkpoint comparisons. Revised
checkpoint selection retained iteration zero: its initial normal/shift interception
was 15.06%/13.85%, and no later validated checkpoint exceeded its combined observed
reward. The control selected iteration ten. Thus these changes improve data quality
and prediction error but do not establish successful end-to-end scheduling learning.
There is only one training seed and no held-out test result for this repair. No
large follow-on training run is justified by these results alone.

An isolated CPU learner check (two threads, 16 trajectories, observation head off,
20 measured forward/backward passes after warmup) measured median 27.53 ms before
and 25.08 ms with reusable staging and cached targets, with identical reported loss.
This is about 9% lower time in that microbenchmark, not an end-to-end/GPU speed claim.
The focused CUDA regression also passed: pinned-buffer reuse, forward/backward on
the online/copied target GRUs, and no recurrent contiguous-weight warning after
explicit packing. Full learning comparisons above remain CPU experiments.

## Compatibility and commands

Model and training schema are now version 3. Version-2 inference checkpoints remain
loadable, and their evaluation retains legacy search settings. Old training runs
must not be resumed under the new algorithm. Keep their artifacts; use a new run
directory. `--initial path/to/best.pt` can initialize shared weights from an older
model while creating the new auxiliary head and a fresh optimizer/replay.

```bash
# Reproduce the bounded revised experiment; use a new directory if this exists.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-rl/bin/python -m spectra_scheduler.mpc_training \
  --run-dir artifacts/mpc-repair-trial --iterations 20 --episodes 32 --workers 4 \
  --batch-size 32 --updates 50 --simulations 32 --validation-episodes 16 \
  --validation-every 5 --device cpu

# For a matching control add:
# --exploration-hold 1 --final-temperature 1 --no-normalize-search --observation-loss-weight 0
```

The repair does not yet add learned dwell-duration actions, dataset-driven replay,
receiver-shift training curricula or a calibrated uncertainty model. Those remain
separate experiments rather than claims supported by this change.
