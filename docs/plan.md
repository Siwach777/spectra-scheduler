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

The first prototype is now working. It has perfect and noisy receiver modes, four
emitter patterns, four scan strategies, hand-checked metrics, and repeated seeded
comparisons. The simple adaptive strategies currently remain baselines rather than
final solutions.

## Later stages

- Add signal strength, sensitivity, retuning time, and variable dwell.
- Add scanning and changing emitters.
- Replace the recency strategy with probabilistic beliefs and change detection.
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
