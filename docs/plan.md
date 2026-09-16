# Working plan

## Goal

The receiver has less instantaneous bandwidth than the spectrum it must monitor.
At each step it chooses a band, listens for a short period, and receives incomplete
information. The project will compare simple scan strategies with a strategy that
learns where and when transmissions are likely to occur.

## First prototype

The first useful version will have:

1. A spectrum divided into numbered bands.
2. Periodic and frequency-hopping emitters.
3. One receiver listening to one band per time step.
4. A fixed round-robin scan and a random scan.
5. A small adaptive strategy based on recent detections.
6. Reproducible scenarios using fixed random seeds.
7. Detection rate, interception rate, and first-detection delay.
8. A runnable terminal demonstration and automated tests.

The simulator will keep the complete generated events separate from the observations
given to a scan strategy. This prevents a strategy from accidentally using future or
hidden information.

The first prototype is now working. It has perfect and noisy receiver modes, dynamic
emitter patterns, several scan strategies, hand-checked metrics, and repeated seeded
comparisons. The simple adaptive strategies currently remain baselines rather than a
final solution.

Current measurements and failed approaches are recorded in
[experiments.md](experiments.md). This keeps changes to the scenario or metrics from
silently replacing earlier results.

Randomizing emitter phases exposed alignment bias in the first scenario. The current
period-aware policy improves average interception but delays discovery, so the next
iteration will add a coverage limit instead of only maximizing repeated hits.

Discovery and coverage measurements now show that a fixed sweep can become phase-locked
with periodic emitters. A shuffled sweep is the next baseline before changing the
period-aware policy again.

The shuffled sweep now provides a stronger discovery baseline. Before adding another
scheduler, receiver truth annotations should be separated from the observation object
passed to policies so future strategies cannot accidentally distinguish false alarms.

That separation is now implemented: policies receive only time, selected band, and
detection count. Matched emitter identities and false-alarm labels are retained in a
separate record used by the evaluation code.

Repeated experiments are now large enough for runtime to matter. Profiling showed
that each strategy regenerated the same scenario truth, so comparisons now generate
it once per seed and reuse it across strategies. Independent seeds can also run in
separate worker processes. Native code is still unnecessary at this scale.

The comparison environment now includes a late-arriving emitter, an emitter that
leaves, and a radar that changes band and repetition interval. The next scheduler
iteration should detect that its older timing evidence has become stale.

Evaluation now reports the fraction of mode changes reacquired and the delay from a
change to the next true detection. These truth labels remain outside the observation
given to schedulers.

A sliding-window UCB baseline now forgets observations after 20 receiver steps. It
improves reacquisition over the all-history UCB policy without knowing when a simulated
mode change occurs. Explicit change detection remains a later experiment.

Transmissions now carry received power in dBm. The receiver applies a sensitivity
threshold with seeded measurement noise, and evaluation separates sensitivity losses
from missed detections after the signal was detectable.

Band changes can now consume receiver retuning steps. Repeated selections represent
dwell without changing the scheduler interface. Evaluation reports time spent retuning,
and adaptive policies ignore those steps instead of learning them as signal misses.

Retuning delay can now increase with band distance. An adaptive dwell sweep waits until
the receiver actually listens, uses a minimum observation dwell, and extends that dwell
after detections up to a fixed safety cap.

The environment also includes an emitter that sweeps adjacent bands and reverses at
the edges of its configured range. This gives future predictors a structured motion
pattern that is different from an arbitrary frequency-hopping sequence.

A first probability-based scheduler now maintains a decaying Beta belief for each
band. It balances posterior hit probability, uncertainty, tuning distance, and a
maximum coverage gap. The baseline is deliberately per-band and inspectable; its
results show that band probabilities alone do not predict a scanning emitter's next
move.

A small extension now learns a decaying table between successive detected bands. It
uses only receiver observations, so it cannot tell whether two hits came from the same
emitter. The comparison shows only a marginal improvement, which makes anonymous
transitions a baseline rather than the final prediction method.

A per-band change detector now compares an older binary hit window with a shorter
recent window. A sustained shift resets only the affected Bayesian belief. It never
receives the simulator's known mode-change labels, and it ignores retuning steps.

Scenario construction now lives outside the comparison runner. Alongside the mixed
case, focused acquisition, adjacent tracking, and mode-change scenarios can be selected
from the command line. This makes it possible to check the intended effect of a policy
before judging its result in a crowded spectrum.

Detected observations can now include anonymous signal measurements. The receiver
reports measured power and pulse width while emitter identities remain only in the
evaluation record. False alarms also receive plausible measurements so a scheduler
cannot identify them from a missing field. Existing count-based schedulers continue to
work unchanged.

Similar power and pulse-width measurements can now form short-lived tracks. A
track-aware scheduler starts with adaptive dwell, follows the predicted band of a
confirmed track, retries through retuning, and forces overdue coverage. The first
motion estimate was linear. It now reflects within observed scan limits only after a
direction reversal has been seen, and repeated prediction misses temporarily return
control to acquisition.

Association evaluation now replays receiver observations through a fresh tracker and
uses truth labels only after assignment. It reports purity, mixed tracks, confirmed
tracks, and the number of track fragments per detected emitter. These values are not
available to a scheduler during simulation.

A longer crowded scenario now places eight emitters in three deliberately similar
signature groups. Pairwise association precision, recall, and F1 complement purity so
that splitting every measurement into a separate track cannot appear successful.
Simultaneous measurements are assigned jointly by lowest normalized cost instead of
depending on their input order.

Repeated association evaluation now supports the same seed range and worker-process
options as strategy comparison. A versioned report combines both summaries and writes
deterministic JSON or tabular CSV without runtime timestamps. The package command,
module entry point, and compatibility script all use the same tested interface.

Pulse-width measurements now include independent seeded relative error. The existing
power-noise sample keys are preserved, so enabling the new uncertainty does not change
the received-power sequence or unrelated scheduler baselines.

Expired tracks now remain in a short archive. A later measurement can reconnect one
only with tighter power and pulse-width thresholds than normal active association.
This preserves its motion history and identifier while keeping archived tracks out of
scheduler decisions until a measurement revives them.

Track-guided scheduling now requires three associated measurements instead of two.
This small evidence threshold performed better than an attempted variance-based gate
on the current short scenarios and adds no new estimator state.

## Later stages

- Add user-defined scenario configuration after the built-in report workflow settles.
- Add V-measure to align association evaluation with the external benchmark.
- Add a TOA-derived or finer-frequency feature before more crowded-track tuning.
- Import a manageable subset of the Turing Synthetic Radar Dataset for calibration.
- Profile the simulator before deciding whether any part should move to Rust.
- Add an API and graphical demonstration only after the experiment format is stable.

## Project layout

```text
src/spectra_scheduler/
  models.py       shared events and observations
  emitters.py     transmission generators
  simulation.py   environment and receiver loop
  receiver.py     missed detections and false alarms
  schedulers.py   scan strategies
  change_detection.py  observation-rate change detector
  tracking.py     anonymous measurement association and motion estimate
  scenarios.py    mixed and focused simulation cases
  metrics.py      experiment measurements
  comparison.py   repeatable strategy comparisons
scripts/
  run_comparison.py
tests/
```

The layout can be split further when a module becomes difficult to understand. It
should not be divided into extra layers in advance.

## Checks before adding machine learning

- Fixed seeds reproduce the same events and results.
- Strategies are tested on identical events.
- Only the simulator and metrics code can access complete truth.
- Baseline results are saved before tuning the adaptive strategy.
- Metric definitions have small hand-checkable tests.

## Technology

- Python 3.12 and NumPy for the initial simulator.
- `uv` for the environment and dependency lock file.
- pytest for tests and Ruff for formatting/linting.
- HDF5 support when the external dataset is introduced.
- Rust with PyO3/maturin only after profiling identifies a useful native boundary.
