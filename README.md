# Spectra Scheduler

A prototype for SIH26055, which asks for a smarter way to scan a wide frequency
range with a receiver that can listen to only a small part of it at once.

The first version is a Python simulation. It will contain a few kinds of emitters,
a narrow-band receiver, simple scan strategies, and measurements of how often and
how quickly transmissions are detected.

## Current direction

- Build a small reproducible simulator.
- Establish fixed-sweep and random baselines.
- Add an adaptive scheduler that learns from hits and misses.
- Compare every strategy on the same generated scenarios.
- Consider a Rust engine only if the Python version is demonstrably too slow.
- Build a graphical interface after the experiments are reliable.

See [docs/plan.md](docs/plan.md) for the working plan.
