# Experiment notes

## Initial noisy comparison with fixed emitter phases

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

Profiling a 2,000-seed comparison showed that scenario truth was being regenerated for
all six strategies. Reusing it once per seed reduced runtime from about 2.63 seconds to
2.07 seconds on the development machine.

The independent seed runs were then distributed across worker processes. For 5,000
seeds, one worker took 5.26 seconds and four workers took 2.15 seconds, a 2.45x speedup.
Processes are used because the simulation is CPU-bound; Python threads would still
contend on the interpreter lock. Results from sequential and parallel execution are
checked for exact equality.

## Dynamic-emitter comparison

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
old timing evidence remains active after the emitter has moved. A sliding-window policy
is the next experiment so observations eventually become stale.

## Sliding-window UCB baseline

UCB was changed to retain only the most recent 20 observations. A 1,000-seed run was
used for this comparison so the reacquisition result was not based on a small set of
emitter phases.

| Strategy | Interception | Discovery | Reacquired | Reacquisition delay | Max gap |
|---|---:|---:|---:|---:|---:|
| Shuffled sweep | 14.1% ± 4.1% | 83.0% | 68.5% | 17.4 | 9.8 |
| UCB bandit | 15.4% ± 4.6% | 76.0% | 58.4% | 19.4 | 13.7 |
| Sliding-window UCB | 15.8% ± 4.5% | 81.9% | 72.4% | 17.4 | 14.5 |
| Period-aware probe | 16.1% ± 5.3% | 76.8% | 64.0% | 18.6 | 12.3 |

Discarding old evidence improves every reported UCB outcome except maximum coverage
gap. It nearly matches the period-aware policy's interception ratio while reacquiring
more changed emitters. The window size is still a fixed assumption; it should later be
compared with explicit change detection rather than tuned only on this scenario.

## Signal strength and sensitivity

Each emitter was assigned a received power between -72 and -92 dBm. The receiver used
a -90 dBm sensitivity threshold, 3 dB seeded measurement variation, 85% detection
probability above the threshold, and a 5% false-alarm probability. Results below use
1,000 seeds.

| Strategy | Interception | Receiver P(detect) | Sensitivity loss | Discovery |
|---|---:|---:|---:|---:|
| Round-robin | 11.4% ± 7.7% | 83.4% | 22.6% | 47.4% |
| Random | 11.7% ± 4.0% | 84.7% | 18.4% | 69.3% |
| Shuffled sweep | 11.4% ± 3.7% | 85.0% | 18.8% | 69.4% |
| Revisit on hit | 10.5% ± 3.9% | 85.3% | 20.6% | 68.7% |
| UCB bandit | 12.1% ± 4.2% | 85.2% | 19.1% | 65.1% |
| Sliding-window UCB | 12.5% ± 4.0% | 85.3% | 19.1% | 70.0% |
| Period-aware probe | 13.4% ± 5.3% | 84.6% | 17.1% | 64.3% |

Receiver P(detect) remains close to its configured 85% once a tuned signal crosses the
sensitivity threshold. The lower interception ratios therefore have two distinct
causes: scanning the wrong band and correctly tuning to a signal that is too weak.
Keeping these counts separate prevents scheduler quality from being confused with
receiver sensitivity.

## Retuning and dwell comparison

The receiver was given a one-step retuning delay. Selecting the same band again means
dwelling, so the scheduler interface did not need a separate action type. A two-step
dwell sweep was added, and UCB policies ignore retuning observations rather than
recording them as signal misses. The period-aware policy explicitly retries a band
after retuning. Results below use 1,000 seeds.

| Strategy | Interception | Discovery | Reacquired | Retuning time | Max gap |
|---|---:|---:|---:|---:|---:|
| Round-robin | 0.5% ± 0.7% | 5.7% | 0.0% | 98.3% | 60.0 |
| Dwell sweep | 5.9% ± 4.5% | 32.7% | 21.3% | 48.3% | 11.0 |
| Random | 2.1% ± 1.9% | 21.7% | 16.3% | 81.9% | 58.6 |
| Shuffled sweep | 0.5% ± 0.9% | 5.1% | 2.9% | 95.9% | 60.0 |
| Revisit on hit | 0.5% ± 0.8% | 6.0% | 0.0% | 97.8% | 60.0 |
| UCB bandit | 5.5% ± 4.0% | 40.8% | 37.0% | 45.7% | 16.8 |
| Sliding-window UCB | 5.7% ± 3.7% | 42.8% | 49.9% | 45.7% | 19.7 |
| Period-aware probe | 5.9% ± 3.3% | 47.1% | 44.0% | 44.8% | 14.9 |

A one-step sweep is not viable when every band change also costs one step: almost the
entire episode is lost to retuning. Fixed dwell restores useful listening time, while
the adaptive schedulers naturally remain on an unobserved band after a retune. The
next policy should choose dwell length from recent evidence instead of using a fixed
two-step setting.

## Distance-based retuning and adaptive dwell

The receiver retains a one-step minimum retune cost and can cross two band indices per
step. An adaptive dwell sweep waits through retuning, collects at least two listening
observations per band, and adds up to two steps after each hit with a six-step cap.
Results below use 1,000 seeds.

| Strategy | Interception | Discovery | Reacquired | Retuning time | Max gap |
|---|---:|---:|---:|---:|---:|
| Dwell sweep | 3.5% ± 2.8% | 28.5% | 0.0% | 55.0% | 58.0 |
| Adaptive dwell | 8.1% ± 3.5% | 54.5% | 56.5% | 30.5% | 24.9 |
| UCB bandit | 5.9% ± 3.0% | 47.2% | 43.0% | 52.0% | 19.6 |
| Sliding-window UCB | 5.7% ± 2.7% | 47.1% | 43.2% | 52.0% | 22.2 |
| Period-aware probe | 6.1% ± 2.9% | 47.2% | 36.5% | 50.1% | 16.3 |

Fixed two-step dwell does not cover the longest retune at the spectrum wrap, producing
a poor maximum coverage gap. Adaptive dwell counts only actual listening steps, so it
waits through that transition and spends more of the episode receiving. Extending a
band after a hit improves interception and reacquisition, but its wider coverage gap
shows the cost of staying longer on active bands.

## Adjacent-band scanning emitter

A sixth emitter was added that moves from band 1 to band 4 and reverses at each edge.
It transmits every two steps at -83 dBm. This pattern tests whether a future scheduler
can learn structured movement rather than treating every band independently. Results
below use 1,000 seeds with the distance-based retuning model.

| Strategy | Interception | Discovery | Reacquired | Retuning time | Max gap |
|---|---:|---:|---:|---:|---:|
| Dwell sweep | 2.3% ± 1.9% | 23.6% | 0.0% | 55.0% | 58.0 |
| Adaptive dwell | 8.3% ± 3.0% | 62.3% | 53.7% | 28.6% | 26.8 |
| UCB bandit | 5.4% ± 2.3% | 51.0% | 39.6% | 50.9% | 20.6 |
| Sliding-window UCB | 5.2% ± 2.3% | 51.1% | 43.9% | 51.1% | 23.4 |
| Period-aware probe | 5.5% ± 2.5% | 50.0% | 33.8% | 49.9% | 16.4 |

Adaptive dwell still has the highest interception and discovery. None of the current
policies explicitly predicts the scanner's next adjacent band, so this scenario is a
useful target for the probability-based scheduler.

## Decaying per-band probability baseline

A probability scheduler was added with a Beta hit belief for each band. Evidence
decays toward its prior so old observations lose influence. Its score combines the
posterior hit probability, an uncertainty bonus, and a penalty for longer retunes.

The first version had no maximum coverage gap. It reached 9.1% interception, 50.2%
discovery, and 31.2% reacquisition, while spending 23.0% of receiver steps retuning.
Its mean maximum band gap was 45.1 steps, showing that it could abandon an apparently
quiet band for most of an episode.

A coverage limit was then added. Results below use 1,000 seeds and the same scanning
emitter scenario.

| Strategy | Interception | Discovery | Reacquired | Retuning time | Max gap |
|---|---:|---:|---:|---:|---:|
| Adaptive dwell | 8.3% ± 3.0% | 62.3% | 53.7% | 28.6% | 26.8 |
| Bayesian band | 6.6% ± 2.5% | 52.2% | 26.3% | 41.7% | 20.5 |

The limit prevents long neglect of a band but causes more distant retunes and gives
up the unbounded policy's interception gain. More importantly, a separate belief for
each band cannot represent the scanner's direction of travel. The next scheduler
experiment should learn observed band transitions instead of further tuning these
constants against one scenario.

## Anonymous transition baseline

The probability scheduler was extended with a decaying table between successive
detected bands. A transition contributes to the score only when the hits are close
enough to be plausibly related. The scheduler still receives no emitter identity.

| Strategy | Interception | Discovery | Reacquired | Retuning time | Max gap |
|---|---:|---:|---:|---:|---:|
| Bayesian band | 6.6% ± 2.2% | 52.2% | 26.3% | 41.7% | 20.5 |
| Transition band | 6.6% ± 2.2% | 52.4% | 27.3% | 41.7% | 20.5 |

A small sensitivity check across transition weights from 0.15 to 1.25 kept mean
interception between 6.5% and 6.6%. The added table therefore gives only a marginal
reacquisition improvement. Hits from several emitters and false alarms are mixed
together, so anonymous transitions are too ambiguous to reliably track the scanning
emitter. The next step is explicit change detection; later tracking will require
measured signal features rather than hidden simulator identities.

## Explicit hit-rate change detection

A change-aware Bayesian policy compares older and recent binary observations for each
band. When their hit rates differ by the configured threshold, it resets that band's
belief and retains the rest of the scheduler state. Retuning observations are excluded.

| Strategy | Interception | Discovery | Reacquired | Retuning time | Max gap |
|---|---:|---:|---:|---:|---:|
| Bayesian band | 6.6% ± 2.2% | 52.2% | 26.3% | 41.7% | 20.5 |
| Change-aware | 6.6% ± 2.2% | 53.8% | 29.5% | 42.1% | 20.6 |

Across 1,000 seeds, the detector fired in 81.7% of runs with 1.20 detected changes per
run on average. It improves discovery and reacquisition modestly without changing
interception. The detector sees aggregate activity on a band, not individual emitters,
so a detected rate shift is not necessarily the known simulated emitter mode change.
Focused evaluation scenarios are needed before combining this detector with more
prediction rules.

## Focused scenario checks

Three smaller scenarios isolate initial acquisition, one adjacent-band scanner, and
one periodic emitter that changes band halfway through the run. Each result below uses
1,000 seeds with the same retuning model as the mixed comparison.

| Scenario | Leading strategy | Interception | Relevant outcome |
|---|---|---:|---:|
| Acquisition | Transition band | 10.6% | 48.4% discovery |
| Adjacent tracking | Adaptive dwell | 11.3% | 100.0% discovery |
| Mode change | Change-aware | 22.4% | 85.8% reacquisition |

In the mode-change case, plain Bayesian scoring reaches 20.2% interception and 71.0%
reacquisition. Resetting stale band beliefs therefore has a clear effect when the
target behavior is isolated. In adjacent tracking, adaptive dwell remains ahead of the
transition policy's 8.4% interception. The anonymous transition table still cannot
reliably associate separated detections with the same moving emitter.

## Measurement-based signal tracks

Detected power and pulse width are now associated into expiring tracks. The initial
implementation used two associated observations to confirm a track and provide a
linear band-motion estimate. The scheduler uses adaptive dwell for acquisition,
follows confirmed predictions, and interrupts track following when the configured coverage
gap is reached. A later experiment below raises the current default to three.

| Scenario | Track-aware interception | Main baseline | Other outcome |
|---|---:|---:|---:|
| Mixed | 8.4% ± 3.2% | Adaptive dwell: 8.3% | 61.7% discovery |
| Acquisition | 8.2% ± 4.3% | Bayesian band: 10.5% | 58.3% discovery |
| Adjacent tracking | 9.7% ± 2.7% | Adaptive dwell: 11.3% | 99.9% discovery |
| Mode change | 42.4% ± 13.5% | Change-aware: 22.4% | 92.9% reacquisition |

The configured 30-step coverage guard gives a 29.8-step mean maximum gap in mixed
traffic; the end of a short run can still finish before an overdue correction. The
tracker is especially effective after confirming a stationary periodic signal, while
linear extrapolation loses the adjacent scanner when it reverses direction. The next
tracking change should represent motion confidence and boundary reversal rather than
increasing dwell/tracking time.

### Miss recovery and observed reversal

The scheduler now abandons a confirmed track after two listening misses. A suppressed
track becomes eligible again when a new matching measurement appears. Band predictions
remain linear until the tracker has actually observed a direction change; after that,
they reflect within the learned band range.

| Scenario | Before | After | Other after-result |
|---|---:|---:|---:|
| Mixed interception | 8.4% | 8.2% | 26.4 maximum gap |
| Adjacent tracking interception | 9.7% | 10.4% | 100.0% discovery |
| Mode-change reacquisition | 92.9% | 93.1% | 13.9-step delay |

Recovery improves the reversing-scanner case and prevents prolonged tracking of a missed
prediction. It gives up a small amount of mixed interception, where measurements from
several emitters compete for attention. Adaptive dwell still leads adjacent tracking
at 11.3%, so association quality should be measured directly before adding more
scheduler rules.

## Signal association quality

The tracker is now evaluated separately from scheduling. Truth emitter labels are
joined to track assignments only after the simulated run, so the scheduler still sees
anonymous measurements. Purity measures the majority-label fraction within tracks;
fragmentation counts how many track IDs represent each detected emitter.

| Scenario | Purity | Tracks per emitter | Confirmed tracks | Mixed tracks |
|---|---:|---:|---:|---:|
| Acquisition | 95.3% | 1.14 | 0.11 | 0.06 |
| Adjacent tracking | 99.6% | 2.27 | 0.88 | 0.02 |
| Mode change | 99.8% | 1.96 | 1.64 | 0.01 |
| Mixed | 95.4% | 1.53 | 1.62 | 0.43 |

These 1,000-seed results use observations produced by the track-aware scheduler. The
matcher rarely combines different emitters in the isolated cases. Its main failure is
fragmentation: sparse detections and expiry split the adjacent scanner into more than
two tracks on average. Mixed traffic also produces nearly half a mixed track per run,
so simply extending track lifetime could reduce fragmentation while increasing false
associations. Pulse-width measurement uncertainty should be modeled before changing
that balance.

### Conservative track reconnection

Expired tracks are now archived for a limited interval. Reconnection requires a pulse
width difference no greater than 0.10 microseconds and a power difference no greater
than 4 dB, both tighter than active-track matching. A small coarse comparison rejected
wider settings because they reduced mixed-scenario purity for only a modest continuity
gain.

| Scenario | Purity | Tracks per emitter | Track-aware interception |
|---|---:|---:|---:|
| Acquisition | 95.1% | 1.05 | 7.9% |
| Adjacent tracking | 99.5% | 1.75 | 11.4% |
| Mode change | 99.8% | 1.59 | 34.6% |
| Mixed | 95.1% | 1.33 | 8.2% |

Compared with the noisy-measurement baseline, scanner fragmentation falls from 2.39
to 1.75 tracks per emitter, and adjacent-tracking interception rises from 10.5% to
11.4%. Mixed purity decreases from 95.9% to 95.1%, while mixed interception remains
8.2%. Reconnection improves continuity, but a fixed rectangular tolerance still does
not express how uncertain each track estimate is.

### Pulse-width measurement uncertainty

Focused and mixed receivers now add 5% standard relative error to pulse width using a
separate seeded noise stream. The power-noise stream was deliberately left unchanged
so the new measurement does not move unrelated baseline results.

| Scenario | Purity | Tracks per emitter | Track-aware interception |
|---|---:|---:|---:|
| Acquisition | 95.4% | 1.14 | 7.9% |
| Adjacent tracking | 99.6% | 2.39 | 10.5% |
| Mode change | 99.8% | 2.13 | 35.7% |
| Mixed | 95.9% | 1.55 | 8.2% |

Purity remains high because the configured emitter pulse widths are separated, but
fragmentation increases in the isolated tracking and mode-change cases. This confirms
that exact pulse width was masking some association loss. The next association change
should reconnect compatible expired tracks conservatively rather than simply widening
the matching tolerance.

### Track confirmation threshold

An uncertainty-aware matching experiment estimated online power and pulse-width
variance after three observations, then normalized residuals by that spread. Even with
noise-based floors and fixed hard caps, it increased fragmentation and reduced
mode-change interception. The extra estimator state was removed.

A simpler comparison changed only the number of associations required before a track
can guide tuning. Three observations performed better than two; requiring four lost
too much stationary tracking time.

| Scenario | Two observations | Three observations |
|---|---:|---:|
| Mixed interception | 8.2% | 8.4% |
| Mixed discovery | 61.4% | 61.9% |
| Adjacent tracking interception | 11.4% | 12.3% |
| Mode-change interception | 34.6% | 34.1% |
| Mode-change reacquisition | 93.0% | 93.4% |

The three-observation default improves focused motion tracking and the mixed result
without materially changing mode-change performance. Variance-based association needs
longer tracks and a harder crowded scenario before it is worth revisiting.

## Crowded association stress test

The crowded scenario runs for 180 receiver steps across eight bands. Eight emitters
form three groups with deliberately similar power and pulse width, and one signal
arrives after the run is underway. Association quality is measured over 1,000 seeds.

| Metric | Result |
|---|---:|
| Track-aware interception | 6.6% |
| Emitter discovery | 90.8% |
| Association purity | 80.1% |
| Homogeneity | 82.7% |
| Completeness | 67.4% |
| V-measure | 74.1% |
| Pairwise association precision | 51.2% |
| Pairwise association recall | 31.3% |
| Pairwise association F1 | 37.4% |
| Tracks per emitter | 2.53 |
| Mixed tracks per run | 4.42 |

Pairwise scoring counts whether pairs of measurements that belong together share a
track and whether unrelated pairs were joined. It exposes both failure directions:
purity alone rewards excessive splitting, while fragmentation alone does not penalize
mixed tracks.

Fixed tolerance sweeps, motion-weighted scores, hard band-speed limits, mature-track
tightening, and adaptive lifetime experiments were rejected. Each either left F1
unchanged or traded false joins for more fragmentation. Power and pulse width do not
contain enough information to cleanly distinguish the deliberately overlapping
groups. The next large association step should add another receiver-visible signal
descriptor rather than continue tuning the same two features.

Multiple detections in one receiver observation are now assigned jointly by normalized
measurement cost. This removes dependence on measurement input order. It fixes a
structural ambiguity but does not materially change the aggregate crowded result.
