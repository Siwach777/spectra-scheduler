# MiMo research and joint timing experiment

The next experiment targets two observed limitations: independent-band timing
hypotheses can disagree about a shared hopping cycle, and the existing forecaster
was trained on histories from five fixed behavior policies rather than its own
decisions. The acceptance comparisons remain native MPC and round-robin, with the
previous best ML policy included to establish whether there is a new gain.

## Paper findings and decisions

| Primary source | Mechanism examined | Scheduling implication and status |
| --- | --- | --- |
| [MiMo-V2.6 technical report](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/resolve/main/MiMo_V2_6_technical_report.pdf), sections 4.1, 4.3 and 5.6 | Large grouped rollouts; offline quality rubrics; online redistribution of positive advantages; MOPD2 uses autonomous student rollouts and continuations from teacher or SFT prefixes. | Adopt dense feedback on student-generated states and retain the strong initialization. Our implementation does not reproduce MiMo's full pipeline, agentic grader or token-level distillation. |
| [MOPD](https://arxiv.org/abs/2606.30406), sections 3.1–3.2 and 4.4 | Frozen specialized teachers supervise the student's own trajectories using reverse KL. The corrected top-k objective includes the missing-mass correction; teachers from a common initialization stabilize integration. | Exact action-distribution KL would be feasible over our small finite action menu. The current experiment instead uses training-only receiver capture labels to supervise future counts, so it is state aggregation inspired by OPD, not a claim of implementing MOPD. |
| [GAGAR](https://arxiv.org/abs/2609.32577), section 3 | Rank passing code trajectories, reduce lower-quality positive advantages and rescale to preserve their sum; failed trajectories keep their outcome signal. | Deferred: our reward is already continuous capture and discovery, rather than identical binary pass scores. A learned textual grader would add noise where the simulator already provides measurable outcomes. The useful lesson is to expose coverage and delay alongside capture instead of hiding them in one success score. |
| [D³-MOPD](https://arxiv.org/abs/2608.24987), section 3 | Adapt the domain mixture using normalized remaining teacher KL and smoothed descent velocity, with a floor for each domain. | Deferred pending evidence of unequal convergence. Our count loss is not teacher KL; borrowing its formula without a meaningful teacher gap would misrepresent the method. Record per-scenario selection curves first. |
| [MOPD-Router](https://arxiv.org/abs/2609.30837), sections 2–3 | Compare teacher routing strategies and use specialization relative to a common base to assess expertise; low entropy alone does not establish expertise. | Do not route runtime decisions using hidden scenario identities or assume the most confident prediction is correct. Joint and independent timing experts are blended using causal measurements. The blend is our own architecture, not ExpertAlign. |
| [DAgger](https://proceedings.mlr.press/v15/ross11a.html) | Aggregate supervision at states reached by the learner because its own decisions change the future observation distribution. | This is the closest precedent for the implemented student-state collection. Our labels are future counts, not expert actions, so the paper's imitation-learning guarantees do not directly transfer. |

These are method-level findings from primary sources. Their reported LLM gains do
not establish gains for RF scheduling. The prior RL research and unsuccessful
variants remain in [trajectory-policy-research.md](trajectory-policy-research.md)
and [dense-blend-findings.md](dense-blend-findings.md).

## Implemented experiment

`JointTimingNetwork` keeps the trained independent-band forecaster frozen and adds
a global causal temporal encoder, a shared learned period gate, and a learned
blend. Period hypotheses cover the existing generic range from 2 through 144 ticks;
no hidden period, hopping permutation, emitter identity or scenario label is an
input. Band parameters are shared and outputs follow a permutation of input bands.

The independent expert estimates a phase rate separately for each band. The
competing-band expert additionally treats phase-aligned hits on other bands as
evidence against simultaneous activity. A trainable mixture avoids making this
single-emitter hypothesis universal. The learned blend can retain the frozen
forecaster where joint evidence is unhelpful. Aperiodic rate estimates remain
available.

Stage one fits all-band future receiver-capture counts on the verified existing
218,239-state training cache. Stage two collects 1,200 additional training worlds
per model seed while the selected student acts on CUDA. Inputs are recorded before
the action. Training-only counterfactual labels use the same deterministic receiver
engine as actual observations, and chosen-action labels are checked against actual
captures. The historical states stay in training to reduce forgetting.

Two model seeds, batch 256 and 20 configurable collection workers are declared.
Only CPU world preparation uses worker processes; neural acting, training and
validation require CUDA. Cached frozen-base predictions avoid repeating its
forward pass during every gradient update. Memory-mapped arrays, two pinned batch
buffers and bounded world batches limit aggregate retention.

The policy uses the native MPC dwell menu `(1, 10, 50)`, public retune delays,
revisit limit 256, probe 10 and exploration 0.02. Checkpoint selection uses twelve
development worlds per scenario and equal-scenario mean capture plus 0.15 times
discovery. Reporting seeds 18000–18099 are declared before training and are
disjoint from selection and procedural training namespaces. The larger discovery
coefficient responds to the previous model's spatial discovery deficit.

The reporting script compares each new forecaster alone and with the same frozen
previous-best action head. It also evaluates that previous policy, the original
forecaster, native PUCT and Gumbel MPC, round-robin and nonlearned phase controls.
Every paired result must have the same truth denominator and 512-tick physical
budget. Capture intervals, discovery, censored acquisition delay and the full
evaluation-contract forecasts are retained. Final test data remain unused.

## Reproduction

```bash
.venv-rl/bin/python -m spectra_scheduler.experiments.joint_study \
  --checkpoint artifacts/timing-report-v1/trained.pt \
  --cache-dir artifacts/timing-belief-cache \
  --run-dir artifacts/joint-timing-v1 \
  --epochs 6 --student-epochs 4 --student-worlds 400 \
  --workers 20 --batch-size 256 --seeds 0 1 \
  --report-seed 18000 --report-runs 100

.venv-rl/bin/python -m spectra_scheduler.experiments.joint_report \
  --run-dir artifacts/joint-timing-v1 \
  --output-dir artifacts/joint-timing-report-v1 \
  --previous-best artifacts/grouped-mpc-grid-v1/seed-1/best.pt \
  --mpc artifacts/mpc-physical-fresh-control/best.pt \
  --mpc artifacts/mpc-physical-gumbel-fresh/best.pt \
  --workers 20 --inference-batch-size 20
```

Add `--verify-only` to the reporting command to compare the batched evaluator with
the shared serial CUDA evaluator on selection worlds. Checkpoints contain both
the frozen base and the new weights and load through `load_joint`.
