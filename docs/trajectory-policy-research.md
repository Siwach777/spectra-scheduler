# Research behind the timing scheduler and trajectory policy

The experiment target is true capture on paired physical receiver worlds against
the repository's trained PUCT MPC, Gumbel MPC and round-robin with useful listening
dwells. Requirements and the passive information boundary come exclusively from
[project_scope.md](project_scope.md). Results are documented in
[timing-model-findings.md](timing-model-findings.md).

## Diagnosis from the implementation

The legacy recurrent context clips recent observation ages at 24 ticks; the MPC
encoder clips visit ages at 30 ticks. MPC's macro encoder preserves aggregate
hits and listening fractions but compresses the timing of interior detections.
These representations discard information needed to distinguish periodic phases
and long gaps. A larger network cannot reconstruct time detail absent from its
inputs. This is a representation limitation; it does not prove that a recurrent
latent state cannot learn useful timing.

The observable RL reward counts hits, which can include false alarms, and mixes
capture, retuning and coverage. Improving this reward or a count-prediction loss
need not improve true capture. The dense learned gate demonstrated that problem:
its training objective improved while selection capture declined. Physical dwell
durations also vary, so optimizing reward per decision can favor a different
policy than optimizing capture over a fixed elapsed horizon.

A genuine execution bug was found in sampled recurrent assessment: physical dwell
was decremented in both `choose_band` and `observe`. The fix leaves physical dwell
decrement exclusively in listening observations; legacy checkpoints retain their
original decision contract. Earlier sampled physical-policy results from the
buggy adapter should not be used as evidence.

The timing model addresses representation and supervision first: it retains
288 receiver ticks with an explicit listening mask, counts and measured power;
a shared causal temporal network weights generic periodic hypotheses and an
aperiodic rate model. Training labels cover every band for the next 80 ticks,
including bands the behavior policy did not choose. Simulated truth is permitted
for those labels and training returns. It never enters runtime policy features.

## Paper-to-implementation decisions

| Primary source | Relevant result or mechanism | Scheduling adaptation and status |
| --- | --- | --- |
| [RLOO: Back to Basics](https://arxiv.org/abs/2402.14740), section 2.3 | Other sampled responses to the same prompt supply a leave-one-out return baseline for REINFORCE. | Implemented: four independent policy trajectories on each procedural world; each advantage subtracts the other three returns. Full-episode capture/discovery returns replace token or dwell-local feedback. |
| [Understanding R1-Zero-Like Training / Dr. GRPO](https://arxiv.org/abs/2503.20783), section 3 | Response-length division and within-group standard-deviation normalization can change the intended optimization weighting. | Implemented: sum each trajectory's decision contributions with a constant `episodes * maximum physical horizon` divisor. No trajectory-length division or group standard-deviation scaling. Variable dwell creates different decision counts even at equal physical horizons. |
| [DAPO](https://arxiv.org/abs/2503.14476), section 3 | Separate upper and lower clipping thresholds and explicit entropy monitoring help maintain exploration in its LLM experiments. | Implemented: ratio clipping `[0.8, 1.28]`, entropy logging and a small entropy term. Zero-variance world groups are counted, but the sampler does not discard or replace them; doing so would change our task distribution. |
| [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) | Group-relative policy updates reduce reliance on a separate learned value baseline. | Used as motivation for a critic-free experiment, not as a claim of a faithful GRPO implementation. The actual baseline is RLOO and the update is clipped. |
| [GSPO](https://arxiv.org/abs/2507.18071), algorithm section | Sequence-level likelihood ratios give a different trust-region treatment from token-level updates. | Reviewed, deferred. A scheduling trajectory has variable listening and retuning time; directly treating dwells as tokens would require specifying and testing a different objective and ratio normalization. |
| [DreamerV3](https://arxiv.org/abs/2301.04104), world-model and behavior-learning sections | A learned recurrent state model supports actor learning from imagined trajectories. | Reviewed, deferred. The current model predicts capture counts and timing; it is not a recurrent stochastic state-space model and does not train an actor through imagined rollouts. |
| [TD-MPC2](https://arxiv.org/abs/2310.16828), algorithm section | Task-oriented latent dynamics, value estimates and regularization support receding-horizon control. | Reviewed, deferred. It motivates auditing latent-value accuracy and terminal value before increasing search budgets; the repository's discrete MPC is not TD-MPC2. |
| [AWAC](https://arxiv.org/abs/2006.09359) | Offline experience can initialize a policy before online reinforcement learning. | Abstract reviewed. Its useful initialization principle is adopted through a trained timing prior; advantage-weighted critic regression was not implemented. |
| [Agent Lightning](https://arxiv.org/abs/2508.03680), training/execution interface | Decoupled execution and credit assignment enable training over complex agent trajectories. | Reviewed as supporting motivation for a separate collection/update interface. Its framework and hierarchical credit assignment were not imported. |

These transfers are hypotheses about shared optimization problems, not evidence
that LLM benchmark gains transfer to RF scheduling. No language model is part of
the receiver scheduler. The full implementation is an RLOO advantage estimator
with asymmetrically clipped updates, KL to the starting policy and bounded action
residuals. Clipping and repeated updates make it an approximate policy update;
the unbiased leave-one-out baseline alone does not make the whole algorithm
unbiased.

## Implemented trajectory experiment

The pretrained forecaster remains frozen. A shared two-layer action network with
256 hidden units receives its 80-tick forecast plus 20 causal action/context
statistics. A zero-initialized output layer preserves the initial forecaster's
greedy action choice. The residual is bounded, and a KL penalty discourages
discarding that useful prior. Coverage probes remain an explicit observed-age
rule, so improvement must also be compared with a strong structural control.

Each iteration prepares 20 independent worlds using 20 processes and collects
four independently sampled trajectories per world. The return is
`true_capture_ratio + 0.05 * emitter_discovery_ratio` at the complete 512-tick
horizon. Both terms use truth only in training/evaluation. Action features use
measured receiver history, public retune durations and elapsed time. Feature
quantization is identical during action collection and replay to preserve old
log-probability parity.

CUDA evaluates the frozen forecaster and action network in batches. Optimizer
minibatches contain 512 decisions, with gradient accumulation over complete
trajectory groups and four update passes. The float16 feature store has an
explicit one-GiB bound; host/GPU history buffers are reused. CPU worker BLAS
threads are limited. Two policy seeds each receive 80 iterations, or 6,400
trajectories per seed. They share one forecaster training seed.

Two action-menu experiments are retained: 1/4/8/16/32 ticks for the timing policy,
and MPC's exact 1/10/50 ticks to remove action-menu differences from the main
comparison. Checkpoint selection uses only seeds 2000–2011. Reporting seeds are
declared before training and are distinct from selection and training namespaces.
The original final test split is unused.

## What remains unresolved

Learning a timing belief is the largest established contributor. Initial
trajectory updates do not give a consistent capture improvement over that
forecaster across all required scenarios. Spatial discovery can decline as the
policy exploits productive bands. A capture-only win is therefore insufficient
to claim the whole project is solved.

Interception-ratio forecasts remain poorly calibrated on some periodic worlds.
Public detection-probability calibration assumes unseen signals are above
sensitivity; it cannot infer arbitrary subthreshold arrivals. A model that
separately estimates arrival intensity and receiver response needs a declared
distributional assumption and a paired threshold sweep. Longer periods, changing
emitters, different band counts and replay recordings also need independent
validation. These are measured or interface-specific gaps, not reasons to add
unrelated model capacity without evidence.
