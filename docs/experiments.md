# Experiment notes

## Initial noisy comparison with fixed emitter phases

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

## Randomized-phase comparison

Date: 14 September 2026

The emitter phases were then generated from each scenario seed. The comparison was
expanded to 100 seeds; receiver settings and the 60-step duration remained unchanged.

| Strategy | Interception ratio | Receiver P(detect) | False alarms | Mean first delay |
|---|---:|---:|---:|---:|
| Round-robin | 14.1% ± 7.3% | 84.2% | 2.6 | 30.0 |
| Random | 14.5% ± 4.0% | 84.5% | 2.5 | 22.2 |
| Revisit on hit | 13.8% ± 4.8% | 82.9% | 2.6 | 24.6 |
| UCB bandit | 15.5% ± 4.0% | 84.1% | 2.5 | 24.3 |
| Period-aware probe | 17.3% ± 4.6% | 86.4% | 2.2 | 28.1 |

Random phases remove much of round-robin's earlier advantage. The period-aware policy
now has the highest average interception ratio, but its long probing periods still
delay discovery of other emitters. The next change should limit how long any band can
go unvisited and should evaluate emitter discovery separately from event interception.

## Coverage-limited timing comparison

Date: 14 September 2026

Emitter discovery ratio and maximum unvisited-band gap were added to show the cost of
staying on productive bands. The period-aware policy was limited to a configured
12-step coverage gap.

| Strategy | Interception ratio | Emitter discovery | Mean max gap | Mean first delay |
|---|---:|---:|---:|---:|
| Round-robin | 14.1% ± 7.3% | 60.2% | 5.0 | 30.0 |
| Random | 14.5% ± 4.0% | 86.4% | 24.5 | 22.2 |
| Revisit on hit | 13.8% ± 4.8% | 79.8% | 9.0 | 24.6 |
| UCB bandit | 15.5% ± 4.0% | 80.0% | 13.7 | 24.3 |
| Period-aware probe | 15.5% ± 5.1% | 78.4% | 12.4 | 27.5 |

The tail of an episode can finish before a policy corrects an overdue band, so the
reported mean maximum can be slightly above the period-aware scheduler's live limit.

Round-robin visits every band regularly but can repeatedly miss a periodic signal when
the sweep cadence and emitter phase never align. Random scanning breaks this phase lock
and discovers more emitters, but has poor worst-band coverage. A useful next baseline is
a shuffled sweep: visit every band once per cycle while changing the order each cycle.

## Shuffled sweep baseline

Date: 14 September 2026

The shuffled sweep was evaluated with the same 100 seeds. It visits every band once
per cycle but changes the order between cycles.

| Strategy | Interception ratio | Emitter discovery | Mean max gap | Mean first delay |
|---|---:|---:|---:|---:|
| Round-robin | 14.1% ± 7.3% | 60.2% | 5.0 | 30.0 |
| Random | 14.5% ± 4.0% | 86.4% | 24.5 | 22.2 |
| Shuffled sweep | 14.3% ± 3.7% | 88.2% | 9.8 | 20.9 |

Changing the order is enough to break the repeated phase alignment while retaining a
coverage guarantee. It does not increase total interceptions, but it is currently the
best discovery baseline and is a better reference than fixed round-robin alone.

## Repeated-run performance check

Date: 14 September 2026

Profiling a 2,000-seed comparison showed that scenario truth was being regenerated for
all six strategies. Reusing it once per seed reduced runtime from about 2.63 seconds to
2.07 seconds on the development machine.

The independent seed runs were then distributed across worker processes. For 5,000
seeds, one worker took 5.26 seconds and four workers took 2.15 seconds, a 2.45x speedup.
Processes are used because the simulation is CPU-bound; Python threads would still
contend on the interpreter lock. Results from sequential and parallel execution are
checked for exact equality.

## Dynamic-emitter comparison

Date: 14 September 2026

The 60-step scenario was changed so the search emitter stops at step 36, the burst
emitter begins at step 20, and the tracking emitter changes from band 4 with a
seven-step period to band 0 with a four-step period at step 30. The other settings
were kept unchanged and the comparison used 100 seeds.

| Strategy | Interception | Discovery | Reacquired | Reacquisition delay | Max gap |
|---|---:|---:|---:|---:|---:|
| Round-robin | 14.3% ± 8.6% | 57.6% | 49.0% | 17.9 | 5.0 |
| Random | 14.7% ± 4.1% | 83.2% | 70.0% | 17.1 | 24.5 |
| Shuffled sweep | 13.9% ± 4.4% | 83.2% | 71.0% | 18.0 | 9.8 |
| Revisit on hit | 13.8% ± 5.0% | 80.4% | 64.0% | 18.4 | 8.7 |
| UCB bandit | 15.0% ± 4.5% | 75.8% | 65.0% | 18.9 | 13.7 |
| Period-aware probe | 16.2% ± 5.0% | 76.8% | 62.0% | 19.3 | 12.5 |

The period-aware policy still has the best average interception, but it reacquires
fewer changed emitters and takes longer than the shuffled sweep or random scan. Its
old timing evidence remains active after the radar has moved. A sliding-window policy
is the next experiment so observations eventually become stale.
