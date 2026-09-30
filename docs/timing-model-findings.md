# Trained timing scheduler versus MPC and round-robin

## Additional planner refinement

The timing forecaster now supports receding-horizon action planning and controlled
warm-start training on planner-generated histories. One seed completed 24 epochs
of 256 updates, with CUDA batches of 256 and 20 preparation workers. Its selected
checkpoint is `artifacts/timing-refine-v1/seed-0/best.pt` (epoch 24).

On 32 selection worlds per scenario, capture was 35.33% agile, 52.78% spatial and
68.75% periodic; discovery was 100%, 85.94% and 100%, respectively. These are
development selection results, not fresh holdout comparisons. The second seed
was stopped after epoch 14. The main simulator CLI loads this checkpoint with
`--timing-model`; see [runbook.md](runbook.md) for the command and paired controls.

A trained causal timing model with a trajectory-trained action policy beats the
repository's saved PUCT MPC, Gumbel MPC and 50-tick round-robin checkpoints on
fresh simulated requirement worlds. The main comparison uses MPC's exact
1/10/50-tick action menu. Both action-policy training seeds give positive paired
capture intervals against all three competitors in every required scenario.
This is a development holdout result; it does not establish performance on final
test recordings or hardware.

Requirements come exclusively from [project_scope.md](project_scope.md).
The algorithm choices and primary-paper references are in
[trajectory-policy-research.md](trajectory-policy-research.md).

## Main comparison with the same action menu

Each competitor receives the same world, stochastic receiver realization, eight
bands and 512 physical ticks. One tick is explicitly 1 ms. All learned policies
choose among the same 24 band/dwell actions: eight bands times 1/10/50 listening
ticks. Retuning consumes additional elapsed time and the receiver clips the last
action at the horizon. Round-robin-50 cycles bands with a 50-tick listening dwell;
round-robin-10 is included in the full report. A one-tick sweep wastes almost all
its time retuning, so it is not the headline baseline.

The forecaster was frozen before actor training. Actor checkpoints were selected
only on seeds 2000–2011, using mean capture plus 0.05 times mean discovery.
Reporting seeds 16000–16099 were declared in the training configuration before
training began. These are 100 fresh worlds per required scenario, with separate
hashed training and reporting seed namespaces. The final test split is unused.
The MPC checkpoints retain their saved search method, 16 simulations, depth five,
discount and normalization settings. This compares against the existing trained
MPC implementations, not against every possible MPC design or training budget.

Mean per-world true interception ratio, expressed as a percentage:

| Policy | Frequency-agile | Spatial scan | Periodic scan |
| --- | ---: | ---: | ---: |
| Round-robin-50 | 10.95% | 10.85% | 11.58% |
| PUCT MPC | 11.11% | 10.36% | 11.11% |
| Gumbel MPC | 10.81% | 10.50% | 9.36% |
| Trained timing forecaster before trajectory RL | 14.42% | 32.68% | 52.86% |
| Trajectory policy, training seed 0 | 14.31% | 35.35% | 54.42% |
| Trajectory policy, training seed 1 | 14.86% | 35.09% | 54.71% |

Seed 1 has the higher checkpoint-selection score, 0.42903 versus 0.41707;
its selected iteration is 56, while seed 0 selected iteration 80. Both are
reported, rather than choosing a different seed for each reporting scenario.
The ratio is true captured pulses divided by all in-spectrum truth pulses, not
observed hits or a comparison with an untrained initialization.

Seed 1's paired mean capture gains, in percentage points, with 95% bootstrap
intervals from 2,000 resamples of independent worlds:

| Reference | Frequency-agile | Spatial scan | Periodic scan |
| --- | --- | --- | --- |
| PUCT MPC | +3.75 [3.23, 4.28] | +24.73 [19.63, 29.63] | +43.61 [37.10, 49.87] |
| Gumbel MPC | +4.05 [3.50, 4.60] | +24.59 [21.68, 27.42] | +45.35 [41.31, 49.00] |
| Round-robin-50 | +3.91 [3.36, 4.46] | +24.24 [21.87, 26.54] | +43.13 [39.47, 46.27] |
| Timing forecaster before RL | +0.44 [0.09, 0.77] | +2.42 [0.67, 4.36] | +1.85 [-1.32, 5.08] |

Seed 0 also has a positive spatial gain over the forecaster: +2.67 points
[0.89, 4.63]. Its agile and periodic intervals include zero. The matched-menu
experiment therefore establishes an additional spatial scheduling gain for both
policy seeds, not a universal RL gain on every behavior.

Pulse-pooled capture uses sums of captures and truth across worlds. Seed 1 gives
14.86%, 35.92% and 56.83%, versus PUCT's 11.11%, 10.57% and 12.87%.
Its pooled gains over PUCT have positive intervals in all three cases:
[3.23, 4.28], [19.74, 30.71] and [35.94, 51.64] percentage points.
Mean ratios omit worlds with no truth pulses; pooled denominators and support
counts are retained in the JSON report.
Frequency-agile and spatial means have 100 supported worlds each; periodic means
have 97, because three worlds contain no truth pulses.

The full report is
[`artifacts/grouped-mpc-grid-report-v1/comparison.json`](../artifacts/grouped-mpc-grid-report-v1/comparison.json).
It includes all baseline policies, both seeds, per-world counts, predictions,
paired intervals, checkpoint hashes and the frozen action menu. Reporting checks
that every paired policy has the same truth denominator and physical time budget.

## What was trained

The timing forecaster uses a shared causal temporal convolutional network over
288 ticks of measured listening masks, detection counts and received power. It
learns to weight generic periods 2–144 and an aperiodic expert, and predicts the
next 80 ticks for every band. No emitter period, phase, identity, scenario label
or future event is a runtime input. Band parameters are shared, with permutation
equivariance checked explicitly.

Training used 2,400 worlds and 218,239 causal states for 16 epochs. Labels are
counterfactual true receiver captures on every band, permitted by the simulated
training scope. Labels for actually selected bands are checked against the shared
receiver engine. Float16 memory-mapped input, two bounded pinned buffers and CUDA
transfers overlap ingestion with training. Epochs after the first resumed with
batch 256 and 20 preparation workers; the first epoch used batch 64. Peak CUDA
allocation was 3.97 GB and later epochs processed about 1,651 states per second.

The action network uses two 256-unit hidden layers and a bounded residual to the
trained forecaster's scores. RLOO supplies full-trajectory advantages; clipped
updates, KL regularization and a constant physical-horizon divisor borrow
specific ideas from LLM RL papers. This is an adaptation, not a reproduction of
GRPO, DAPO or GSPO. The frozen forecaster provides intercept predictions while
the action head changes band/dwell choices.

Each of two actor seeds received 80 iterations, 20 worlds per iteration and four
sampled trajectories per world: 12,800 trajectories and 6,553,600 physical ticks
combined. Optimizer batch size was 512; preparation used 20 worker processes with
limited nested threads. The matched-menu run took 226.8 seconds and peaked at
653 MB of CUDA allocation. Parent peak RSS was about 2.01 GB; this is not an
aggregate worker-memory measurement. Two actor seeds share one forecaster seed,
so they are not two independent end-to-end model-training replications.

The matched report took 42.5 seconds with neural inference batches of 20 and
20 CPU control workers. Peak CUDA allocation was 168 MB. Forecaster inference
averaged 277 microseconds per decision when amortized across a batch of 20;
this is not serial decision latency. Native MPC batching was checked against
native serial search on selection worlds, including Gumbel's independent RNG
state. Both selected actor policies were also checked against the complete
shared serial CUDA macro evaluator on three selection worlds each.

## Required metrics and tradeoffs

The selected matched-menu seed-1 policy emits forecasts before each action, with
100% forecast coverage. Definitions follow
[evaluation-contract.md](evaluation-contract.md); percentages below are means
across supported worlds, rather than pooled estimates.

| Metric | Frequency-agile | Spatial scan | Periodic scan |
| --- | ---: | ---: | ---: |
| Conditional probability of detection | 89.94% | 90.15% | 88.68% |
| False-alarm probability on negative listening ticks | 1.93% | 1.98% | 2.05% |
| True captures per simulated second | 49.53 | 62.64 | 54.41 |
| Observable reward per simulated second | 62.07 | 77.80 | 70.31 |
| Correct action-window capture classifications | 77.59% | 92.62% | 94.69% |
| Brier score | 0.1720 | 0.0646 | 0.0489 |
| Conditional intercept-time MAE | 2.327 ms | 2.527 ms | 2.930 ms |
| Timing-event coverage | 70.08% | 82.65% | 82.45% |
| Restricted intercept-time MAE | 2.065 ms | 0.987 ms | 0.760 ms |
| Interception-ratio forecast MAE | 0.1348 | 0.1849 | 0.2652 |
| Emitter discovery | 100% | 72.5% | 94% |

Observable reporting reward is `observed_hit - 0.05 * retuning_tick`, identical
for competitors. Training actor returns instead use true episode capture and
discovery. Reward per action is also saved, but actions have unequal duration;
time-normalized reward and true capture are the appropriate main comparisons.

Sensitivity is a configured simulated-dBm receiver threshold, not calibrated
hardware sensitivity. Required-world sensitivity loss is zero because this
suite contains above-threshold signals. That does not establish sensitivity
robustness. Public detection-probability calibration converts the predicted
capturable share into an arrival ratio under that assumption; arbitrary unseen
subthreshold arrivals remain unresolved.

Capture improves while spatial discovery falls from round-robin's 100% to 72.5%.
PUCT and Gumbel discovery on that scenario are 17% and 68%. The forecaster-only
policy reaches 74.5%. Coverage probes bound revisit age but do not guarantee
discovering an emitter whose short visibility window falls between visits.
Interception-ratio error is also substantial on periodic worlds. Neither this
capture win nor high classification accuracy completes those remaining goals.
Mean first-detection delay, censoring undiscovered emitters at the episode end,
is 24.6/231.4/103.8 ms for the selected policy versus round-robin's
12.6/188.6/158.8 ms. PUCT gives 19.3/414.4/382.8 ms and Gumbel gives
13.5/273.2/280.2 ms. Acquisition delay therefore also has a real tradeoff with
repeated capture; the learned policy is not best on every objective.

## Transfer to MPC's broader physical-world distribution

The same frozen matched-menu checkpoints were assessed on MPC's procedural
randomized, receiver-shift and focused-periodic generators, without additional
tuning. Seeds 42000–42019 supply 20 fresh worlds per suite. These worlds have
different layouts, horizons and receiver conditions from requirement training.
The serial tick adapter preserves the entire chosen listening dwell through
retuning; its execution was checked against the shared macro-action engine.

| Policy | Randomized capture | Receiver-shift capture | Periodic capture |
| --- | ---: | ---: | ---: |
| Round-robin-50 | 10.95% | 8.55% | 11.04% |
| PUCT MPC | 8.18% | 6.94% | 16.58% |
| Gumbel MPC | 11.19% | 9.52% | 12.18% |
| Multiscale recurrent PPO | 14.98% | 11.77% | 3.89% |
| Timing forecaster before RL | 22.84% | 18.78% | 21.96% |
| Matched-menu actor seed 0 | 23.99% | 18.46% | 26.15% |
| Matched-menu actor seed 1 | 25.04% | 18.70% | 26.11% |

For seed 1, capture gains over PUCT are +16.86 [11.44, 22.18],
+11.76 [7.41, 16.04] and +9.53 [-5.09, 22.68] percentage points.
The broader periodic interval includes zero; its higher mean is not an
established PUCT win. Gains over Gumbel and round-robin-50 have positive intervals
in all three suites. Actor seed 1 improves randomized capture over the forecaster
by +2.20 points [0.30, 4.38], with no clear additional gain under receiver shift
or in this smaller periodic sample.

The report is
[`physical-comparison.json`](../artifacts/grouped-mpc-grid-report-v1/physical-comparison.json).
It records paired world digests, receiver sensitivity loss, discovery,
reacquisition and per-emitter-family outcomes. Its reward uses the same observable
MPC benchmark reward for all competitors. The run took 297.0 seconds and peaked
at 39.0 MB of CUDA allocation. Seed-1 `choose_band` calls averaged
0.459/0.436/0.903 ms per physical tick, including calls that simply hold a dwell.
These serial figures differ from batched decision costs; they do not establish
a hardware latency deadline.

## Additional experiments

The original 1/4/8/16/32-tick menu was assessed independently on reporting seeds
14000–14099. Actor seeds 0/1 captured 17.15%/18.12% agile, 40.24%/39.43% spatial
and 58.52%/55.66% periodic, compared with PUCT's 11.35%/10.25%/12.08% and
round-robin-50's 10.90%/10.89%/11.56%. All paired capture intervals against both
MPC checkpoints and round-robin-50 were positive. The different action menu makes
this a separate experiment, not the main controlled comparison.

An earlier 100-world forecaster report on seeds 12000–12099 established learned
signal against a matched untrained network and an observed-rate control. Against
a strong nonlearned beta-binomial phase policy, the learned forecaster had a clear
agile advantage but no clear spatial or periodic advantage. The full structural
controls remain visible; the trained model should not be described as universally
superior to all timing heuristics.

The multiscale recurrent PPO run completed 524,800 decisions and 6,951,338
physical ticks using 20 environment workers, batch 1024, longer physical-time
discount/trace factors and causal long-age features. It did not match the timing
model's capture in the broader comparison. Dense learned blending and the
unsuccessful learned gate are documented separately in
[dense-blend-findings.md](dense-blend-findings.md).

## Reproduce and load the candidate

CUDA is required for both training and neural evaluation. Use fresh output
directories; reporting refuses to overwrite frozen results. The full forecaster
training command is:

```sh
.venv-rl/bin/python -m spectra_scheduler.experiments.timing_study \
  --run-dir artifacts/timing-reproduction \
  --cache-dir artifacts/timing-reproduction-cache \
  --worlds 800 --epochs 16 --workers 20 --batch-size 256 --seeds 0
```

The retained forecaster was selected on checkpoint/policy selection worlds;
its exact bytes are already stored with both actor runs. To reproduce the
matched-action actor and its main comparison from those bytes:

```sh
.venv-rl/bin/python -m spectra_scheduler.experiments.grouped_study \
  --checkpoint artifacts/timing-report-v1/trained.pt \
  --run-dir artifacts/grouped-mpc-grid-reproduction \
  --iterations 80 --worlds 20 --group 4 --workers 20 --batch-size 512 \
  --seeds 0 1 --dwells 1 10 50 --probe 10 --report-seed 16000 --report-runs 100

.venv-rl/bin/python -m spectra_scheduler.experiments.grouped_report \
  --run-dir artifacts/grouped-mpc-grid-reproduction \
  --output-dir artifacts/grouped-mpc-grid-report-reproduction \
  --mpc artifacts/mpc-physical-fresh-control/best.pt \
  --mpc artifacts/mpc-physical-gumbel-fresh/best.pt \
  --workers 20 --inference-batch-size 20
```

For serial/batched parity, add `--verify-only` to the reporting command. It checks
selection worlds and does not overwrite a report. The experiment modules provide
Python integration; a hardware or replay frontend integration is not implemented.

```python
from pathlib import Path
from spectra_scheduler.grouped_policy import GroupedTimingPolicy, load_grouped
from spectra_scheduler.experiments.calibrated_timing import configure_public_detection
from spectra_scheduler.scenarios import build_requirement_scenario
from spectra_scheduler.synthetic_evaluation import evaluate_scheduler

path = Path("artifacts/grouped-mpc-grid-v1/seed-1/best.pt")
forecaster, actor, settings, metadata = load_grouped(path)
policy = GroupedTimingPolicy(forecaster, actor, settings)
simulation = build_requirement_scenario("periodic-scan", 42)
configure_public_detection(simulation, policy)  # Only the public receiver response.
report = evaluate_scheduler(
    simulation, policy, step_seconds=0.001,
    reward=lambda observation: float(observation.hit) - 0.05 * (not observation.listening),
    reward_description="observed_hit - 0.05 * retuning",
)
```

Keep `forecaster.pt` at the parent run directory when moving checkpoints; the
loader verifies its content hash. The policy assumes millisecond ticks, forecasts
80 ticks ahead and rejects public retune/dwell combinations outside that horizon.
Source hashes in historical artifacts identify the code used for those runs.
Subsequent changes added API checks, a short-horizon loss fix and parity reporting;
they do not alter valid 80-tick model inference. Strict cache provenance checks
require a fresh cache after source edits.

Focused verification covered 23 unique checks across timing-history causality,
counterfactual receiver labels, permutation equivariance, count-loss gradients,
short forecast horizons, retune limits, macro/tick execution parity, calibration,
leave-one-out advantages, legal-action gradients and recurrent compatibility.
Ruff and `git diff --check` passed. Neural parity checks and all training/evaluation
experiments used CUDA; unit checks used CPU without training a model.
