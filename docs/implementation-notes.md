# Implementation notes

This is a brief record of what exists and why it was added. Detailed measurements
remain in [experiments.md](experiments.md).

| Area | What is implemented | Purpose |
|---|---|---|
| Foundation | Python package, tests, working plan, and terminal comparison | Keep the first prototype reproducible and focused on simulation |
| Environment | Periodic, hopping, scanning, burst, jittered, windowed, and mode-switching emitters | Represent both stable and changing RF activity |
| Receiver errors | Configurable detection probability and false alarms | Avoid testing schedulers with perfect feedback |
| Fair evaluation | Randomized phases and repeated seeded comparisons | Prevent fixed timing from favoring one scan pattern |
| Truth separation | Public observations contain no emitter identity or false-alarm label | Prevent future policies from using hidden simulator information |
| Metrics | Interception, discovery, coverage, delay, sensitivity, and reacquisition | Keep scheduler, receiver, and change-response effects separate |
| Association metrics | Truth-only purity, mixed-track, and fragmentation measurements | Diagnose tracking without exposing labels to the scheduler |
| Baselines | Fixed, random, shuffled, revisit, UCB, sliding UCB, period-aware, fixed-dwell, and adaptive-dwell scans | Compare simple ideas before adding a larger learning model |
| Dynamic behavior | Emitters can enter, leave, and change operating modes | Test whether learned behavior becomes stale |
| RF strength | Received power, sensitivity threshold, and measurement noise | Separate weak-signal loss from choosing the wrong band |
| Signal measurements | Detected pulses expose independently seeded noisy power and pulse width without emitter IDs | Provide realistic association inputs while preserving the truth boundary |
| Signal tracks | Similar measurements form expiring tracks, estimate motion, and reflect only after an observed reversal | Predict from receiver evidence instead of joining hits through hidden identities |
| Track-aware scheduler | Adaptive dwell acquires signals; confirmed tracks guide tuning until misses or the coverage guard interrupt pursuit | Combine broad search and focused tracking with explicit recovery limits |
| Retuning | Band changes can use a fixed delay or a delay based on band distance; repeated selections mean dwell | Account for hardware cost without complicating the scheduler interface |
| Adaptive dwell | A sweep waits through retuning and extends productive bands after a hit | Trade broad coverage against longer observation of useful bands |
| Bayesian scheduler | Decaying per-band hit beliefs with exploration and switching cost | Add an inspectable probability-based policy before larger learning models |
| Transition scheduler | A decaying table scores bands that followed recent detected bands | Test whether anonymous hit-to-hit motion is useful before adding signal tracking |
| Change detection | Per-band older and recent hit windows trigger a local belief reset | React to sustained behavior shifts without using simulator change labels |
| Focused scenarios | Separate acquisition, adjacent tracking, and mode-change cases | Measure each scheduler behavior without hiding it inside the mixed scenario |
| Performance | Shared scenario truth and parallel independent runs | Support larger experiments before considering native code |

New work should add or update one short row here describing both the change and its
reason.
