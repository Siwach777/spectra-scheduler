# Experiment notes

## Initial noisy comparison

Date: 14 September 2026

The first repeated comparison used 30 seeds, six bands, 60 time steps, 85% receiver
detection probability, and 5% false-alarm probability. The scenario contained fixed,
hopping, burst, and jittered emitters.

| Strategy | Interception ratio | Receiver P(detect) | False alarms | Mean first delay |
|---|---:|---:|---:|---:|
| Round-robin | 29.0% ± 2.2% | 83.2% | 1.9 | 4.9 |
| Random | 14.6% ± 4.4% | 85.4% | 2.3 | 21.7 |
| Revisit on hit | 14.4% ± 4.4% | 84.2% | 2.1 | 23.0 |
| UCB bandit | 18.5% ± 4.9% | 86.6% | 2.1 | 9.5 |
| Period-aware probe | 17.4% ± 4.3% | 83.4% | 2.5 | 29.2 |

Interception ratio is the useful scheduling measurement here. Receiver P(detect) is
conditional on the receiver already being tuned to the correct band, so a high value
does not imply a good scan strategy.

### What was learned

- A strategy that immediately revisits every hit is easily distracted by false alarms.
- UCB learns which bands have produced hits, but not when those bands will be active.
- The first period-aware scheduler probes a detected band for too long. This delays
  coverage of the remaining spectrum and can estimate a period from false alarms.
- Several emitter phases are fixed in this scenario. Their alignment with round-robin
  may account for part of its advantage and must be randomized before drawing a strong
  conclusion.

### Next check

Randomize emitter phases with the scenario seed, retain the same receiver settings,
and rerun the comparison. After that, constrain timing probes with a maximum coverage
gap rather than tuning the scheduler against the current fixed scenario.
