# MPC diagnosis and assessment

The earlier eight-action MPC chooses a band every receiver tick. Its search and
direct policy both selected one band throughout the 30-world legacy test run:
mean bands visited was 1.0 and the dominant-band action fraction was 1.0 on
randomized and receiver-shift suites. The selected checkpoint intercepted
12.45% and 9.83% of transmissions on those suites, respectively. These
figures describe a historical checkpoint, not the current training target.

The repair experiment with normalized search, a listening/hit observation
head, and four-tick exploratory persistence improved collection listening
from 8.80% to 77.40% after 20 CPU iterations. Its final randomized validation
capture rose from 6.28% to 9.96% against a matched control, while receiver-shift
capture fell from 7.86% to 6.15%. Validation still selected the untrained
iteration-zero checkpoint. Thus the partial repair did not establish better
end-to-end scheduling. The old artifacts remain under
`artifacts/mpc-repair-control` and `artifacts/mpc-repair-trial`.

The revised MPC model uses 24 band-and-dwell actions (eight bands, 1/4/8
physical ticks). Its search discounts by elapsed ticks, masks unavailable
bands, and evaluates simulations in bounded CUDA batches. Collection mixes
receiver conditions; replay-root reanalysis refreshes policy and value targets
from saved causal histories. Old eight-action checkpoints remain inference-only
compatibility artifacts. The scheduler updates its observation state on every
physical tick while retaining the chosen band for the learned dwell.

Assess a completed run against untrained search/direct-policy controls, all
standard schedulers, four- and eight-tick sweeps, and the longer adaptive dwell
control on paired development worlds:

```bash
.venv-rl/bin/python -m spectra_scheduler.experiments.mpc_assess \
  --run-dir artifacts/mpc-dwell-reanalyse --runs 30 --seed 30000
```

The assessment loads `best.pt` and the saved `untrained.pt` initialization on
CUDA, uses the training run's MCTS simulation count, and writes
`mpc-assessment.json` beside the checkpoints. Each policy receives the same
world and receiver realization. The report includes per-world capture and
discovery, paired capture differences and bootstrap intervals, checkpoint
hashes, and implementation hashes. Fixed legacy layouts receive new
phase/noise seeds; only the randomized and receiver-shift suites contain new
procedural layouts. This is one training seed and a development comparison,
so it cannot establish training-seed robustness or a test-set result.

Both repaired variants completed 48 iterations with the same 96-episode,
80-update and 48-simulation budget. The paired assessment found randomized
capture of 0.1248 for the no-reanalysis control, 0.1835 for replay-root
reanalysis, and 0.134 for adaptive-long; receiver-shift capture was 0.0814,
0.1515 and 0.091 respectively. The reanalyzed search still visited only
roughly two to three bands, captured none on spatial and periodic scans, and
reached 0.022 on change worlds versus 0.464 for adaptive-long. Its gains on
two development suites therefore do not justify deployment as a general
scheduler. See [EfficientZero](https://arxiv.org/abs/2111.00210) for the
reanalysis motivation; these measurements are specific to this implementation.

## Fresh physical-MPC search comparison

The physical synthetic receiver uses eight bands with 1/10/50-tick dwells. A
new Gumbel search option follows the central ideas of
[policy improvement by planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO):
Gumbel top-k root candidates, sequential halving, completed-Q policy targets
and completed-Q non-root selection. The theoretical improvement result assumes
correctly evaluated action values. Our learned dynamics does not satisfy that
assumption by construction, so capture must be measured.

Fresh PUCT and Gumbel runs used the same random seed, 64 iterations, 32 worlds
per iteration, 40 updates, 16 simulations, eight reanalysis episodes per
iteration and no checkpoint initialization. Best checkpoints were selected on
eight validation worlds. A separate 30-world paired development assessment per
suite produced:

| Suite | PUCT capture / discovery | Gumbel capture / discovery | Gumbel untrained-search capture |
| --- | --- | --- | --- |
| Randomized | 0.1575 / 0.4030 | 0.1041 / 0.8462 | 0.1099 |
| Receiver shift | 0.1396 / 0.3916 | 0.0852 / 0.9587 | 0.0786 |

PUCT search capture was approximately equal to its own untrained search and
direct trained policy. Gumbel search greatly improved discovery but gave up
capture; it did not reliably improve on its own untrained search. Neither
method is established as a strong learned scheduler. Saved paired reports and
bootstrap intervals are in `artifacts/mpc-physical-fresh-control` and
`artifacts/mpc-physical-gumbel-fresh`. The test split remains unused.

The physical model also has optional elapsed-time input for macro actions and
a causal shortest-dwell coverage probe shared by collection and inference. An
initial fresh 64-iteration run with both options selected its untrained
checkpoint; it has no separate paired assessment. These are experimental
switches, not validated gains. Reanalysis was already active in all three
physical runs, despite being disabled in the generic configuration default.
