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
emitter patterns, nine scan strategies, hand-checked metrics, and repeated seeded
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

## Later stages

- Add a scanning emitter that sweeps adjacent bands.
- Add probabilistic beliefs and explicit change detection.
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
