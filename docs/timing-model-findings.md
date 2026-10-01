# Trained timing scheduler versus MPC and round-robin

## Current checkpoint

The timing forecaster now supports receding-horizon action planning and controlled
warm-start training on planner-generated histories. One seed completed 24 epochs
of 256 updates, with CUDA batches of 256 and 20 preparation workers. Its selected
checkpoint is `artifacts/timing-refine-v1/seed-0/best.pt` (epoch 24).
Its SHA-256 is `f821a7672b5b378154caa89d981893b77a62ddc0119ea74053483aa611ca744f`.
The planner uses 1/10/50-tick listening dwells, an 80-tick forecast horizon,
revisit setting 512 and a 10-tick coverage probe. Retuning consumes physical time.

On 32 selection worlds per scenario, capture was 35.33% agile, 52.78% spatial and
68.75% periodic; discovery was 100%, 85.94% and 100%, respectively. These are
development selection results, not fresh holdout comparisons. The second seed
was stopped after epoch 14. The main simulator CLI loads this checkpoint with
`--timing-model`; see [runbook.md](runbook.md) for the command and paired controls.

## Fresh comparison of the selected checkpoint

Settings were frozen on 32 development worlds per scenario before evaluating
100 fresh worlds per scenario, seeds 48000–48099. Every policy received the same
eight bands, 512 physical ticks, truth and stochastic receiver realization.
Adaptive controls select their complete 1/10/50-tick listening dwell globally on
selection worlds; retuning is additional elapsed time. This compares the configured
implementations, without claiming optimal tuning of every algorithm.

Mean per-world true capture:

| Policy | Frequency-agile | Spatial scan | Periodic scan |
| --- | ---: | ---: | ---: |
| Round-robin, 50-tick listening dwell | 11.10% | 11.17% | 10.87% |
| UCB, 50-tick dwell | 11.17% | 17.59% | 18.91% |
| Sliding UCB, 10-tick dwell | 9.66% | 10.48% | 11.18% |
| Bayesian occupancy, 10-tick dwell | 10.40% | 11.34% | 10.65% |
| Thompson sampling, 10-tick dwell | 9.80% | 23.62% | 30.07% |
| Discounted Thompson sampling, 10-tick dwell | 9.94% | 10.90% | 12.49% |
| Beta-rate Whittle, 50-tick dwell, age bonus | 11.08% | 9.31% | 8.99% |
| Markov Whittle, 1-tick dwell | 9.62% | 25.24% | 32.86% |
| Golden sweep, 50-tick dwell | 10.89% | 10.84% | 10.37% |
| Non-neural phase planner | 27.57% | 45.85% | 64.15% |
| Saved PUCT MPC | 11.14% | 11.17% | 8.26% |
| Saved Gumbel MPC | 10.54% | 10.12% | 9.84% |
| Selected timing forecaster and planner | **34.37%** | **51.56%** | **67.66%** |

Paired capture gains over round-robin-50 are +23.26 [22.12, 24.34],
+40.39 [37.11, 43.49] and +56.79 [53.17, 60.09] percentage points,
with 95% world-bootstrap intervals. Periodic capture has 99 supported worlds;
one world contains no truth pulses. The periodic gain over the phase planner
is +3.51 points with interval [-0.07, 7.05], so it is not an established gain.

Discovery is 100%/86%/97%, versus round-robin's 100%/97.5%/98%.
Spatial discovery loss is -11.5 points [-16, -7]; first-detection delays are
24.21/184.95/99.91 ms versus 14.4/182.39/194.11 ms. Strong capture therefore
coexists with a real spatial discovery weakness. These remain synthetic
development holdout results, not final-test or hardware results.

The public [summary and provenance](../reports/timing-selected-summary.json)
contains checkpoint/source hashes, settings and paired intervals. Full local
results are `artifacts/scan-comprehensive-report-v1/comparison.json`; its frozen
selection is `artifacts/scan-comprehensive-selection-v1/selection.json`.

## Coverage and earlier screening

### Coverage selection

A 32-world-per-scenario coverage study retained the incumbent. Uniform 128-tick
revisits raised spatial discovery from 85.94% to 92.19%, while spatial capture
fell from 52.78% to 33.92% and periodic capture from 68.75% to 43.42%.
Targeted recovery probes used public hit history, a physical-time budget including
retuning, and a forecast opportunity-cost gate. None increased discovery while
retaining at least 90% of incumbent capture and avoiding discovery loss in every
scenario. The experimental recovery policy is not enabled in the CLI or GUI.
The completed selection is `artifacts/coverage-opportunity-v2/selection.json`.
Short probes and band coverage do not guarantee interception of brief windows.

A later diagnosis found nine missed spatial emitters in 96 development worlds;
eight never transmitted during the receiver's listening ticks on their bands.
Acquisition extensions therefore consolidate already-selected short probes,
using measured-power support, at most 10% added elapsed time and a 5% forecast
opportunity-cost gate. Extensions begin only after a reliable signal is observed;
initial search and mandatory coverage are preserved.

On 300 fresh worlds (seeds 54000–54099), spatial discovery changed from 81.0% to
83.5%, with paired difference 2.5 points [-2.5, 7.5]; capture changed from 48.65%
to 49.00%. Agile capture changed from 35.34% to 34.83%, and periodic from 66.38%
to 66.44%; discovery in those scenarios was unchanged. The spatial discovery gain
is not established, so this extension is also experimental and the CLI/GUI retain
the incumbent. Fixed sweeps, adaptive controls, the phase planner and both MPC
models used identical fresh worlds and physical-time budgets. See the
[acquisition evidence](../reports/timing-acquisition-summary.json).

Run a new selection without retraining:

```bash
.venv-rl/bin/python -m spectra_scheduler.experiments.coverage_study \
  --checkpoint artifacts/timing-refine-v1/seed-0/best.pt \
  --run-dir artifacts/coverage-selection-new --runs 32 --batch-size 20
```

Use `--mode acquisition` for contiguous extensions. Diagnose completed episodes
with `timing_discovery`; report a frozen choice with `coverage_report`, passing
`--coverage-selection`, the existing `--control-selection`, and the saved `--mpc`
checkpoints. Discovery diagnostics join truth only after scheduling finishes.

### Matched scan controls

The audited screening run reused the current checkpoint without changing its
weights. Two selection seeds per scenario chose control settings; reporting used
three separate seeds, 28000–28002, for each required scenario. All policies shared
eight bands, 512 physical ticks, receiver realizations and native listening dwells.
The saved MPC models retained their own search settings.

Mean per-world true capture percentages on these nine reporting worlds:

| Policy | Frequency-agile | Spatial scan | Periodic scan |
| --- | ---: | ---: | ---: |
| Current trained timing planner | 33.92% | 58.47% | 69.00% |
| Non-neural phase planner | 33.53% | 43.34% | 72.12% |
| Markov Whittle, 1-tick dwell, no age bonus | 10.14% | 22.51% | 39.74% |
| Golden acquisition followed by timing planner | 28.27% | 50.76% | 25.75% |
| Round-robin-50 | 12.09% | 8.83% | 13.57% |
| PUCT MPC | 12.09% | 0.00% | 31.33% |
| Gumbel MPC | 11.31% | 4.85% | 13.54% |

The trained planner's discovery was 100%, 83.33% and 100%; round-robin-50
discovered every emitter in these worlds. Markov Whittle's spatial discovery was
50%. The phase planner captured more periodic opportunities than the trained
planner. The selected checkpoint remains unchanged; three seeds per scenario
are a smoke check, not evidence of universal superiority or a final benchmark.

The full counts, discovery, acquisition times, paired intervals and frozen settings
are in `artifacts/scan-strategy-audit-report/comparison.json`; its selection is in
`artifacts/scan-strategy-audit-selection/selection.json`. These are local ignored
artifacts. Source hashes identify the implementation used when each run executed.

`whittle_policy.py` implements beta-rate and Markov-belief controls, signed
persistence fitting, receiver-error conditioning and a 256-step numerical subsidy
index approximation. The stationary rate is estimated separately; the dynamics
fit is not joint maximum likelihood. Markov beliefs evolve through unobserved
retuning ticks, and band scores account for public tuning duration.
`scan_handover.py` feeds the full causal history to the existing timing model
while acquisition runs. These controls are experimental and are not main CLI defaults.

The numerical index was compared with a reference implementation in five regimes;
focused checks also cover the closed-form branches in
[Liu and Zhao](https://arxiv.org/abs/0810.4658), negative-correlation recovery and
noisy belief updates. Observation-error treatment is related to
[imperfect-observation restless-bandit analysis](https://arxiv.org/abs/2108.03812).
Finite lookahead, estimated dynamics and native macro dwells limit applicability
of the papers' optimality results. Fifteen focused CPU checks passed; neural
screening used CUDA. Reproduction commands are in [runbook.md](runbook.md).

## Earlier trajectory-policy comparison

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

### Comparison with the same action menu

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

## Full serial decision latency

Batch-one measurements on the RTX 5070 Laptop GPU include receiver-feedback
ingestion since the previous decision, history encoding, CUDA prediction,
retune-aware planning and forecast creation. Six fresh worlds per scenario use
seeds 52000–52005; warmup worlds are excluded. Costs are measured per new
band/dwell decision, rather than diluted across dwell-holding ticks.

| Runtime | Agile p99 | Spatial p99 | Periodic p99 |
| --- | ---: | ---: | ---: |
| Python planner, ordinary CUDA inference | 3.246 ms | 1.951 ms | 3.043 ms |
| Compiled planner, ordinary CUDA inference | 1.903 ms | 1.509 ms | 1.632 ms |
| Compiled planner, captured CUDA inference | **0.532 ms** | **0.472 ms** | **0.464 ms** |

The accelerated path had no violations of the declared 1-ms software budget over
5,986 decisions; medians were 0.421–0.425 ms. Counts and report metrics match the
original 300-world benchmark, with six additional serial checks. Unit checks
also cover native planning values, horizon clipping and ties.

These are warmed scheduler measurements, excluding checkpoint loading, CUDA
capture, simulator truth and receiver I/O. The first warmup decision reached
1.567 ms. This supports software feasibility on this host, not a certified
receiver deadline or a measured neural CPU deployment. Both acceleration options
are explicit in the CLI and GUI. See the
[measurement summary](../reports/timing-latency-summary.json) and
[reproduction commands](runbook.md#accelerated-runtime-and-serial-latency).

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
Python integration; the selected forecaster also has a
[PDW replay adapter](replay-interface.md#frozen-timing-scheduler-on-external-recordings).
There is no hardware receiver integration.

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
