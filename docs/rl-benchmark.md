# RL benchmark: protocol and initial evidence

The benchmark measures a learned decision policy, not just hit-prediction accuracy.
It retains handwritten strategies as controls; none overrides an RL decision.

## Evaluation contract

- Every policy receives identical generated transmissions and seeded receiver noise
  within a world. Simulator truth remains unavailable to policy inputs/rewards.
- Training, validation and test procedural generators have independent seed
  namespaces. Layouts vary emitter count, bands, periods, hopping paths, burst
  patterns, arrival windows, mode-change times and receiver characteristics.
- `randomized` tests new layouts from the training generator family.
- `receiver-shift` additionally changes band count (where applicable), horizon,
  detection probability, false-alarm rate, noise and retuning delay.
- The five legacy presets are separate diagnostic suites. Their new seeds vary
  phases/noise, not all structural assumptions. Test legacy seeds receive an offset
  to avoid reusing validation worlds.
- Frozen inference is deterministic, with per-episode state reset. No online weight
  updates or hidden-truth reward are used in evaluation.
- Reports retain per-world metrics and scenario hashes, model hashes, configuration,
  source hashes, means and paired interception differences. Approximate 95% normal
  intervals are over scenario seeds, not all sources of training uncertainty, and
  have no multiple-comparison correction. One run gives no interval.
- Multiple DQN artifacts require distinct training RNG seeds. Their reported
  variability is separate from world-to-world variation. A single recurrent run is
  not evidence of training-seed robustness.
- Primary measurements are interception, discovery, first-detection/reacquisition
  delay, retuning and maximum band gap. Training reward is also reported, but cannot
  substitute for these measurements or establish deployment success.

`SimulationEpisode` supplies the same step engine to the Gymnasium environment and
ordinary simulator comparisons. Tests check their observation/result equivalence.

## Completed lightweight reference experiment

Three independently initialized Double-DQN models, seeds 0/1/2, each trained for
1,500 procedural episodes (180,000 receiver steps). Final checkpoints were evaluated
without model selection. Thirty validation worlds per suite start at seed 10000.
Training used six bands, a 64-unit shared action scorer, gamma 0.97, a 10,000-transition
replay, batch size 64, target updates every 250 steps and epsilon-greedy exploration.

Interception means below average the three fitted models across the same worlds;
the best existing policy is selected descriptively per suite, not claimed to be a
universal winner.

| Suite | RL mean | Best existing baseline | Untrained network mean |
| --- | ---: | --- | ---: |
| Randomized layouts | 21.75% | Transition-band: 15.16% | 8.86% |
| Receiver shift | 9.48% | Adaptive-dwell: 6.33% | 4.31% |
| Crowded | 10.89% | Track-aware: 6.68% | 6.69% |
| Mixed | 7.20% | Adaptive-dwell: 8.86% | 7.64% |
| Acquisition | 6.72% | Transition-band: 10.90% | 13.85% |
| Tracking | 9.33% | Track-aware: 14.11% | 6.19% |
| Change | 17.30% | Track-aware: 34.44% | 27.41% |

These are promising gains on randomized layouts, not universal superiority. Some
untrained policies do surprisingly well on fixed layouts, which reinforces the
importance of structural randomization and retaining controls.

Coverage is a major weakness: RL mean maximum band gaps were 98.02 of 120 steps on
randomized worlds and 177.56 of 180 steps on crowded worlds. Crowded emitter
discovery averaged 58.89%. Improved interception can coexist with neglected bands;
there is no demonstrated acceptable interception–coverage operating point yet.
The change-suite RL results vary from 9.11% to 31.44% across training seeds.

Artifacts remain local and ignored: `artifacts/rl-seed0.json` through
`artifacts/rl-seed2.json`, with `reports/generated/rl-validation.json`. The supervised
hit predictor was also included in that detailed report. No TSRD files were needed
or consumed by RL training. The final procedural test benchmark has not been used
for model development; unit checks of split separation are not a final evaluation.

## Main recurrent experiment

The recurrent PPO path replaces the small reference with a PyTorch LSTM actor/critic
and parallel eight-band simulation. Its default coverage penalty is stronger, so
it changes both architecture and objective. Checkpoint selection is confined to a
small validation set. Its performance must be assessed in a new common-world report,
not compared directly against this six-band reference table.

Larger capacity, a faster GPU and more training are resources, not evidence of a
better model. In particular, gains in shaped reward alone, checkpoint selection on
the final test, or gains bought with severe coverage loss do not establish success.

Commands and continuation/progress instructions: [rl-training.md](rl-training.md).
