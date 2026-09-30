# Multi-teacher on-policy scheduling experiment

This experiment creates independently specialized scheduling policies and
integrates them into one student on that student's own causal observation states.
The implemented training recipe and declared evaluation protocol are described
here; gains require the separately frozen reporting comparison.

## Research and adaptation

[MOPD](https://arxiv.org/abs/2606.30406), sections 3.1–3.2 and 4.4, develops
specialist teachers from a common initialization and freezes them while the
student generates trajectories. Each teacher scores the matching student
prefixes; integration minimizes reverse KL. Same-origin initialization supports
distributional alignment. The finite receiver menu permits exact summation over
all 24 legal actions, removing the need for truncated vocabulary objectives.

[GAGAR](https://arxiv.org/abs/2609.32577), section 3, redistributes positive
advantages using rankings of successful code implementations, preserving their
sum. Receiver capture and discovery already provide continuous outcome signals.
This experiment uses those measured outcomes for specialist RL and implements
MOPD integration without a language-model grader or advantage redistribution.

This is a scheduling adaptation of MOPD's specialist/integration structure.
Its student integration is distinct from the existing grouped rollout optimizer:
it receives full teacher action distributions on student states, rather than
episode reward advantages. Specialist development reuses the existing RLOO and
clipped update implementation.

## Interfaces and boundaries

`ExpandedActionResidual` preserves every old-head logit at initialization. It
adds a zero-initialized unbounded linear correction and a trainable global prior
scale initialized at one, constrained to `[0.05, 2]`. The original bounded
residual remains available. This removes the old head's inability to reverse
prior-score gaps greater than eight while retaining the strong selected policy
as the initial control. All teachers and the student share this initialization.

The frozen causal timing forecaster, receiver bandwidth, retuning, native dwell
menu `(1, 10, 50)`, and coverage constraints supply the same public interface.
Every actor sees the same existing causal features. Domain labels select a
teacher only during training; the deployed student receives neither a scenario
identifier nor emitter truth. Mandatory coverage probes remain legal-action
constraints and have zero distillation gradient.

Teacher distributions are evaluated on the exact features, priors and legal
menus recorded while the student acts. They stay frozen. Every distillation
iteration recollects trajectories from the current stochastic student. The loss
is the exact conditional `KL(student || teacher)` across all legal actions,
averaged over free decisions within each world, then over worlds. This balances
short and long trajectories. Several gradient updates reuse each iteration's
states; the next iteration collects fresh states.

## Declared study

Two training seeds independently specialize three teachers for 64 iterations
each, using 20 procedural worlds and four grouped trajectories per world. Each
student then receives 32 balanced-domain distillation iterations. Neural acting,
training and validation require CUDA. The minibatch is 512, native thread count
is limited, CPU world preparation uses 20 configurable worker processes, and
trajectory retention has an explicit one-GiB feature-storage bound.

Checkpoint selection uses seeds 2000–2011 on each required scenario. Teachers
are selected on their own domain; the student is selected on the equal-scenario
mean of capture plus 0.05 times discovery. The initial checkpoints remain
eligible. Training worlds use a separate hashed procedural namespace. Reporting
seeds 20000–20099 are written into the configuration before any training.

Reporting compares both frozen students against the previous best fixed head,
native PUCT and Gumbel MPC, and round-robin with dwells 10 and 50. Every paired
world must have the same truth denominator and 512-tick physical budget. Reports
retain paired intervals, pooled capture, discovery, acquisition delay and the
full evaluation-contract forecasts. Runtime student inference has no teacher
dispatch. Original checkpoints remain immutable.

```sh
.venv-rl/bin/python -m spectra_scheduler.experiments.mopd_study \
  --initial-actor artifacts/grouped-mpc-grid-v1/seed-1/best.pt \
  --run-dir artifacts/mopd-native-v1 \
  --specialist-iterations 64 --student-iterations 32 \
  --worlds 20 --group 4 --workers 20 --batch-size 512 \
  --seeds 0 1 --report-seed 20000 --report-runs 100

.venv-rl/bin/python -m spectra_scheduler.experiments.mopd_report \
  --run-dir artifacts/mopd-native-v1 --output-dir artifacts/mopd-native-report-v1 \
  --mpc artifacts/mpc-physical-fresh-control/best.pt \
  --mpc artifacts/mpc-physical-gumbel-fresh/best.pt \
  --workers 20 --inference-batch-size 20
```

Expanded artifacts use version two and `load_mopd`; the existing grouped-policy
loader and checkpoints retain their original behavior. Focused CPU unit checks
cover KL direction, teacher gradient isolation, legal-action masking,
equal-world weighting, train-only dispatch, exact initialization parity and
corrections beyond the old head's bound. Neural experiment evidence is obtained
on CUDA.
