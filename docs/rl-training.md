# Recurrent scheduler training

The main learned scheduler uses SB3-Contrib recurrent PPO with separate 256-unit
actor and critic LSTMs. A local duration-aware rollout buffer discounts rewards
and generalized advantages by elapsed **physical receiver ticks**, while the
maintained PPO optimizer handles clipped policy updates. The older NumPy
Double-DQN in `rl.py` remains a lightweight reference.

## Decision model

The policy receives causal receiver history for eight bands: hit rates, recent
feedback, listening and hit ages, tuning distance, and a validity mask. It never
receives emitter identity, future pulses or simulator truth. It selects one of 24
band-and-dwell actions: eight bands times 1, 4 or 8 physical ticks. Observations
update every tick while a chosen dwell holds the band. Retuning, including its
lost listening time, comes from the shared receiver engine. On smaller legacy
worlds only nonexistent bands are masked; the learned policy has no forced
sweep or coverage override.

Per-tick reward is the observed hit indicator, minus 0.05 for retuning and a
coverage penalty on mean band age. The recurrent run uses coverage weight 0.5;
the NumPy reference used 0.05, so their results are not a pure architecture
comparison. The mixed procedural training distribution includes shifted
receivers in one quarter of worlds. A macro action sums its per-tick rewards
with discount 0.99; the PPO return and advantage recurrences also account for
its actual elapsed ticks.

## CUDA run and checkpoint selection

The development host has a Core Ultra 9 275HX (24 threads), 32 GB DDR5 and an
RTX 5070 Laptop GPU with 8 GB VRAM. The separate `.venv-rl` contains CUDA
PyTorch and SB3-Contrib. Training and neural validation require CUDA and fail
explicitly if it is unavailable.

```bash
.venv-rl/bin/python -m spectra_scheduler.recurrent_cli train \
  --run-dir artifacts/recurrent-dwell --steps 2000000 \
  --workers 24 --torch-threads 2 --device cuda --seed 0
```

`--steps` counts **decision actions**, not physical ticks. Each run uses 24 CPU
environment actors, 128 decisions per actor per rollout, 1,024-sample learner
minibatches and four PPO epochs. `start` can replace `train` for a detached run;
its log and progress file show whether the child started successfully. `--resume`
continues from the saved model and optimizer at the requested total decision
budget. It restores per-worker episode indices, but live recurrent state and
random streams are not bitwise identical to uninterrupted training.

Before updates, the trainer saves `untrained.zip`. It selects `best.zip` using
the mean per-world harmonic capture/discovery score on 16 normal and 16
receiver-shift validation worlds (seeds 10000–10015 in each distribution).
`latest.zip` stores the most recent completed update. Every ZIP has a sidecar
with its SHA-256 digest, source settings, decision count and physical-tick count.
The loader rejects a digest mismatch. `progress.json` records status and counts;
`progress.csv` records optimizer diagnostics without a calendar timestamp.

Three independent training seeds completed 2,002,944 decisions each. They
experienced 8,664,133, 8,725,577 and 9,056,667 physical ticks respectively.
All selected checkpoints beyond initialization; seed 2's best checkpoint was
saved after 1,419,264 decisions. Each completed run received the same paired
development assessment.

## Paired development assessment

```bash
.venv-rl/bin/python -m spectra_scheduler.experiments.recurrent_assess \
  --run-dir artifacts/recurrent-dwell --runs 30 --seed 20000
```

This CUDA assessment compares greedy and seeded sampled execution of the frozen
PPO model with its untrained initialization, thirteen existing schedulers,
four- and eight-tick sweeps, and a longer adaptive dwell control. Each policy
receives the same scenario and receiver realization. Reports include paired
capture differences and bootstrap intervals. Only the randomized and
receiver-shift suites use new procedural layouts; eight other suites use fixed
layouts with changed phase/noise seeds. The separate test namespace is not used
for checkpoint selection or these development comparisons.

Aggregate the three assessment reports with:

```bash
.venv-rl/bin/python -m spectra_scheduler.experiments.recurrent_aggregate \
  --run-dir artifacts/recurrent-dwell \
  --run-dir artifacts/recurrent-dwell-seed1 \
  --run-dir artifacts/recurrent-dwell-seed2 \
  --output artifacts/recurrent-three-seed-comparison.json
```

The aggregate resamples training seeds and paired world indices independently. Across the
three seeds, greedy PPO capture was 0.236 versus 0.142 for adaptive-long on
unseen randomized layouts (paired difference interval [0.032, 0.156]), and
0.165 versus 0.086 under receiver shift ([0.032, 0.128]). It also led on
frequency-agile and spatial-scan suites. It trailed adaptive-long on acquisition
(0.148 versus 0.195), change (0.154 versus 0.464), and periodic scan (0.043
versus 0.114). Mixed, tracking and crowded differences were unresolved by the
reported intervals. Fixed-layout suites vary phase and noise, not layout;
three seeds and 30 development worlds per suite are still limited evidence.

## Sources

- [PPO](https://arxiv.org/abs/1707.06347) and [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438): clipped policy updates and discounted advantage estimation.
- [SB3-Contrib recurrent PPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html): maintained recurrent policy and rollout implementation.
- [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264): motivation for reporting training-seed variability and paired uncertainty.
