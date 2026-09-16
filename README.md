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
- Expose anonymous measured power and pulse width for later signal association.
- Associate measurements into expiring tracks and schedule confirmed predictions.
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

Available scenarios are `mixed`, `acquisition`, `tracking`, `change`, and `crowded`.
The crowded case is longer and deliberately contains emitters with similar measured
signatures, so it is mainly useful for checking track association.

Include track-association quality in the terminal output:

```bash
PYTHONPATH=src python3 -m spectra_scheduler --scenario crowded --runs 100 --association
```

Save a complete, repeatable experiment report for later analysis:

```bash
PYTHONPATH=src python3 -m spectra_scheduler \
  --scenario crowded --runs 100 --workers 4 \
  --output reports/crowded.json
```

The output format is inferred from `.json` or `.csv`, or can be selected with
`--format`. Reports include the scenario, seed range, scheduler summaries, association
summary, and a schema version. They do not contain timestamps. The older
`scripts/run_comparison.py` command remains available as a wrapper.

See [docs/report-format.md](docs/report-format.md) for the stable fields.

Run the tests with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
