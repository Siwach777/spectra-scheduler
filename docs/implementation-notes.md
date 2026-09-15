# Implementation notes

This is a brief record of what exists and why it was added. Detailed measurements
remain in [experiments.md](experiments.md).

| Area | What is implemented | Purpose |
|---|---|---|
| Foundation | Python package, tests, working plan, and terminal comparison | Keep the first prototype reproducible and focused on simulation |
| Environment | Periodic, hopping, burst, jittered, windowed, and mode-switching emitters | Represent both stable and changing RF activity |
| Receiver errors | Configurable detection probability and false alarms | Avoid testing schedulers with perfect feedback |
| Fair evaluation | Randomized phases and repeated seeded comparisons | Prevent fixed timing from favoring one scan pattern |
| Truth separation | Public observations contain no emitter identity or false-alarm label | Prevent future policies from using hidden simulator information |
| Metrics | Interception, discovery, coverage, delay, sensitivity, and reacquisition | Keep scheduler, receiver, and change-response effects separate |
| Baselines | Fixed, random, shuffled, revisit, UCB, sliding UCB, period-aware, and dwell scans | Compare simple ideas before adding a larger learning model |
| Dynamic behavior | Emitters can enter, leave, and change operating modes | Test whether learned behavior becomes stale |
| RF strength | Received power, sensitivity threshold, and measurement noise | Separate weak-signal loss from choosing the wrong band |
| Retuning | Band changes can consume receiver steps; repeated selections mean dwell | Account for hardware cost without complicating the scheduler interface |
| Performance | Shared scenario truth and parallel independent runs | Support larger experiments before considering native code |

New work should add or update one short row here describing both the change and its
reason.
