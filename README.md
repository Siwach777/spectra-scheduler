# Spectra Scheduler

A prototype for SIH26055, which asks for a smarter way to scan a wide frequency
range with a receiver that can listen to only a small part of it at once.

The first version is a Python simulation. It will contain a few kinds of emitters,
a narrow-band receiver, simple scan strategies, and measurements of how often and
how quickly transmissions are detected.

## Current direction

- Build a small reproducible simulator.
- Model emitters that enter, leave, scan adjacent bands, or change behaviour.
- Model received power, sensitivity loss, and repeatable receiver noise.
- Account for distance-based retuning cost and adapt dwell after detections.
- Establish fixed-sweep and random baselines.
- Add an adaptive scheduler that learns from hits and misses.
- Forget stale observations so a scheduler can respond to changed emitters.
- Detect sustained per-band hit-rate changes without exposing simulator truth.
- Compare every strategy on the same generated scenarios.
- Consider a Rust engine only if the Python version is demonstrably too slow.
- Build a graphical interface after the experiments are reliable.

See [docs/plan.md](docs/plan.md) for the working plan and
[docs/implementation-notes.md](docs/implementation-notes.md) for a brief explanation of
what each part is for.

## Run the prototype

The current comparison uses seeded emitter timing, missed detections, and false alarms.

```bash
PYTHONPATH=src python3 scripts/run_comparison.py
```

Use several seeds for a more useful comparison:

```bash
PYTHONPATH=src python3 scripts/run_comparison.py --runs 30
```

Larger comparisons can use multiple CPU cores because each seeded run is independent:

```bash
PYTHONPATH=src python3 scripts/run_comparison.py --runs 2000 --workers 4
```

Use a focused scenario to inspect one scheduler behavior at a time:

```bash
PYTHONPATH=src python3 scripts/run_comparison.py --scenario change --runs 100
```

Available scenarios are `mixed`, `acquisition`, `tracking`, and `change`.

Run the tests with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
