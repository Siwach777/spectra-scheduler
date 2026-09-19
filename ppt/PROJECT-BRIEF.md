# Spectra Scheduler — Project Brief

## 1. Project identity and pitch

**Problem statement:** SIH26055 — Smart Scan Strategy for Electronic Support (DRDO: Development of Smart Scan Strategy for Electronic Warfare in the absence of prior reliable intelligence of emitters and their operating characteristics).

**Expected solution:** Machine learning based Electronic Support receiver scheduler software.

**Category:** Software. **Organisation:** DRDO.

**Team name / Team ID:** To be added. Confirm the exact theme on the official portal;
public statement copies disagree on that field.

**One-line pitch:** Spectra Scheduler is a closed-loop spectrum-management platform
being developed to improve signal interception through adaptive receiver scheduling,
pulse association and learning from incomplete observations.

**Presentation summary:** A receiver cannot continuously observe the entire spectrum.
Spectra Scheduler combines a configurable RF simulation environment, measurement-based
tracking and dataset-backed association benchmarks to evaluate where and when a
receiver should listen. The planned learning layer aims to improve the
interception–latency trade-off while preserving discovery coverage and accounting
for receiver constraints.

## 2. What problem are we solving?

The frequency range of interest exceeds the receiver's instantaneous bandwidth.
The receiver must therefore allocate limited observation time across frequency bands.
A signal can be missed even when it is strong enough to detect: the receiver may be
listening elsewhere when the signal occurs.

This creates a **joint frequency–time allocation problem**:

- Which band should the receiver observe next?
- How long should it dwell there?
- When should it revisit previously observed activity?
- How should it continue discovering unfamiliar or newly active sources?
- How should decisions change when earlier observations become unreliable?

A fixed sweep offers predictable coverage but does not adapt its allocation to new
evidence. An excessively exploitative policy has the opposite weakness: repeatedly
visiting productive bands can leave other regions unobserved. Retuning delays, missed
detections and false alarms further complicate the decision.

The project addresses this resource-allocation problem through passive-receiver
simulation and offline data analysis. It focuses strictly on passive reception and cognitive scheduling without active transmission.

## 3. Objectives and intended outcomes

- **Improve interception efficiency:** capture a greater fraction of available
  transmissions under a fixed receiver-resource budget.
- **Reduce acquisition latency:** shorten time to first detection and reacquisition
  after changes in emitter behaviour.
- **Preserve discovery coverage:** prevent persistent neglect of less-observed bands.
- **Improve association quality:** reduce incorrect pulse grouping and track fragmentation.
- **Adapt to non-stationarity:** respond when historical activity no longer predicts
  current observations.
- **Scale data processing:** keep memory use bounded during ingestion and make
  computation costs measurable.
- **Support reproducible decisions:** provide comparable baselines, explicit evaluation
  settings and traceable experiment reports.

Success means improving the trade-off against the **strongest relevant baseline** on
held-out scenarios with equivalent bandwidth, noise and retuning assumptions. Numerical
deployment targets will be established from measured receiver and compute constraints;
the project does not claim an unverified universal interception percentage.

## 4. How the proposed solution works

### A. Model the observation problem

The simulation represents periodic, frequency-hopping, scanning, burst, jittered and
mode-changing activity. Sources can become active, disappear or change behaviour.
The receiver model introduces sensitivity limits, noisy measurements, missed detections,
false alarms and retuning overhead.

Complete simulated truth is retained for evaluation. Policies receive only permitted
receiver observations, preventing access to hidden emitter identities or future events.

### B. Associate measurements

The existing simulator tracker associates anonymous power and pulse-width measurements
into short-lived tracks and can reconnect recently expired tracks under stricter gates.

The external-data pipeline separately benchmarks pulse association using HDBSCAN,
a density-based clustering method that does not require a predefined emitter count.
The baseline compares raw pulse descriptors with scaled signatures that handle feature
scale differences and the circular nature of arrival angle.

These are currently separate evaluation paths. Dataset clusters are not yet connected
to an online receiver scheduler.

### C. Allocate receiver observations

Existing policies include fixed and randomized scans, adaptive dwell, bandit-based
baselines, decaying band beliefs, change-aware behaviour and track-guided revisits.
They establish reference behaviour and expose the cost of exploration, exploitation
and receiver transitions.

The proposed learned extension is a contextual policy that uses observable history
and measurement-derived context. A contextual-bandit approach is the initial candidate;
more complex models must justify their added cost through validation.

### D. Evaluate and improve

Compare policies on identical generated events and compare association methods on
identical sampled pulse populations. Measure both intended gains and adverse trade-offs,
including coverage loss, false associations, rejection rate and computation cost.

## 5. Architecture

The design uses a small modular Python package rather than a distributed service stack.

```text
SIMULATION AND SCHEDULING

Scenario definition -> RF environment -> Receiver observations -> Tracker / policy
                             |                    ^                      |
                             |                    +-- next tuning action-+
                             v
                      Evaluation truth -> Metrics -> JSON / CSV reports

DATASET ASSOCIATION

Completed HDF5 files -> Schema validation -> Bounded batches and sampling
                                                   |
                                                   v
                                      Feature transform -> HDBSCAN
                                                   |          |
                                           Separate labels    v
                                                   +----> Evaluation -> JSON report
```

The dataset branch provides offline association evidence. It is not a substitute for
the interactive environment needed to compare receiver scheduling actions.

Module boundaries allow the reader, association model, scheduler and evaluator to evolve
independently. A native performance module can be introduced at a measured bottleneck
without rewriting the entire system.

## 6. Technology stack and rationale

| Technology | Role | Status and rationale |
|---|---|---|
| Python | Simulation, orchestration, evaluation and command interfaces | Implemented; supports rapid experimentation and a maintainable modular codebase |
| NumPy | Feature arrays and numerical transformations | Implemented; vectorised operations reduce Python-level processing overhead |
| HDF5 / h5py | Pulse-file access and batch reading | Implemented; avoids loading the complete collection into RAM |
| scikit-learn / HDBSCAN | Unsupervised pulse-association baseline | Implemented; supports an unknown emitter count and explicit noise rejection |
| Process-based parallelism | Independent file and simulation evaluation | Implemented; avoids nested numerical-thread oversubscription in clustering workers |
| uv and dependency lock | Reproducible Python environment | Implemented; records resolved dependencies |
| unittest / pytest / Ruff | Automated verification and static checks | Implemented; checks correctness, integration and code consistency |
| PyTorch | Possible learned embeddings or neural policy models | Planned option; adopt only if simpler models are insufficient |
| Rust with PyO3 | Selective native acceleration | Conditional plan; profile first and preserve the Python interface |
| API and graphical dashboard | Experiment control and visual diagnostics | Planned; framework selection follows the stable evaluation workflow |

Dataset size alone does not establish a need for a native engine. The present priority
is bounded ingestion, efficient numerical operations and controlled concurrency. Rust
is the preferred native option if profiling identifies a worthwhile boundary; a parallel
C++ implementation is not currently planned.

## 7. Dataset and data-handling strategy

The project uses the **Turing Synthetic Radar Dataset (TSRD)**, an externally provided
synthetic pulse dataset, not a collection of live operational recordings.

Each pulse provides arrival time, centre frequency, pulse width, angle of arrival and
amplitude. The reader resolves column order from metadata and preserves the dataset's
units. In particular, amplitude in dB is not silently interpreted as simulator power
in dBm, and arrival-time microseconds are not equated to simulation steps.

Data-handling principles:

- Read completed HDF5 files while excluding partial-download and archive paths.
- Validate shapes, feature names, label alignment, finite values and arrival-time order.
- Retain the provided receiver-mode and train/validation/test boundaries.
- Keep emitter labels separate from inference features; label identities are file-local.
- Read full-file statistics in chunks and use explicitly bounded samples for clustering.
- Record sample fingerprints, configuration and library versions for reproducibility.
- Report empty, unlabelled and invalid files rather than silently replacing them.

The current clustering baseline is **transductive**: it fits an unsupervised partition
to each sampled pulse train. It is not yet a reusable supervised emitter classifier.
Recorded scan data also omits activity outside the original listening schedule, so it
cannot by itself establish how a different scheduling policy would have performed.

## 8. What is implemented, and what remains planned?

### Implemented foundation

- Configurable RF simulator and receiver constraints.
- Heuristic and statistical scan-policy baselines.
- Measurement association, expiring tracks and change-response mechanisms.
- Repeatable simulation comparisons and structured result export.
- Chunked TSRD ingestion and full-file validation/statistics.
- Raw-feature and scaled-signature HDBSCAN benchmarks.
- Per-file association metrics, explicit noise scoring and reproducible sampling.
- Process-level evaluation parallelism and optional memory/throughput profiling.

### Planned capabilities

- Broader association evaluation and frozen validation-set comparisons.
- Temporal features and, if justified, learned pulse embeddings.
- An explicit connection between association outputs and receiver-visible policy context.
- A trained scheduling policy evaluated against existing baselines.
- Complete reward/cost, prediction-correctness and intercept-time-error reporting.
- A separately evaluated periodic-scan reference case.
- Profile-guided acceleration, followed by API and dashboard integration.

The simulation models are abstractions, not a calibrated RF hardware model. Spatial
antenna-scan modelling, receiver integration and field validation need additional work.

## 9. Measured engineering evidence

This evidence supports feasibility; it is not a substitute for held-out model evaluation.

**Data ingestion:** all 2,500 scan training files were streamed, covering 233,172,417
pulses. Eight files were valid empty inputs, and no schema/read errors were found.
The observed peak was approximately 59.4 MiB per inspection worker with 65,536-row
batches. A separate check covered approximately 3.6 million pulses in stare training files.

**Association comparison:** two feature configurations were evaluated using the same
100,000 sampled pulses from the first ten scan training files:

| Configuration | File-macro V-measure | File-macro pairwise F1 |
|---|---:|---:|
| Raw pulse descriptors | 0.4519 | 0.1815 |
| Scaled signatures | 0.8435 | 0.8121 |

Both runs used identical clustering settings and the single-noise-cluster scoring
convention. This demonstrates a preprocessing effect on a limited training subset,
not a held-out accuracy claim, full-dataset benchmark or receiver-scheduling gain.
V-measure is a clustering score—not “percentage accuracy.”

Reproduction details and measurement qualifications are in
[the dataset workflow](../docs/dataset-workflow.md). Validation and test splits were
not used for these development runs.

## 10. Validation methodology

### Existing methods

- **Unit testing:** emitter behaviour, receiver constraints, tracking and metric calculations.
- **Integration testing:** scenarios, dataset ingestion, evaluation commands and report output.
- **Schema and boundary testing:** malformed files, empty inputs, missing labels and partial batches.
- **Regression and determinism checks:** fixed-seed replay and sequential/parallel equivalence.
- **Truth-separation checks:** feature sampling and clustering predictions remain independent of labels.
- **Seeded comparisons:** repeat policy experiments under common generated conditions.
- **Resource profiling:** observe read throughput and process peak memory on downloaded files.

### Planned extensions

- Held-out scenario and dataset validation after fixing preprocessing choices.
- Noise, emitter-overlap and distribution-shift stress tests.
- Component ablations to isolate the contribution of tracking, history and learning.
- Sensitivity analysis for dwell, matching and policy parameters.
- Dataset-scale performance benchmarks and decision-latency measurements.
- Final evaluation on untouched test inputs with uncertainty reporting.

Evaluation spans receiver detection and false alarms; interception, discovery and
reacquisition; coverage and retuning overhead; association precision/recall and
V-measure; and memory and latency. Forecast-specific and reward metrics remain part
of the planned learned-policy evaluation.

## 11. Hurdles and mitigation strategy

| Hurdle | Why it matters | Approach |
|---|---|---|
| Partial observability | Unobserved bands are not equivalent to inactive bands | Keep observation and truth paths separate; evaluate decisions in simulation |
| Similar pulse signatures | Different emitters can be merged, or one emitter fragmented | Multi-feature baselines, explicit association metrics and planned temporal features |
| Changing activity | Historical evidence becomes stale | Compare change-aware and decaying-history baselines; stress-test future learned models |
| Feature units and scales | Large numerical ranges can dominate clustering distances | Metadata-driven columns, documented units and explicit feature transforms |
| Angle wraparound | Nearby directions may have numerically distant angle values | Use a circular sine/cosine representation in the signature baseline |
| Data volume | Whole-collection loading is impractical | Batch reading, sample limits, vectorised processing and controlled file-level parallelism |
| Sampling bias | Rare emitters may be omitted and cluster densities altered | Report sample scope and rejection; expand coverage before drawing general conclusions |
| Noise-scoring ambiguity | Rejecting difficult pulses can make results misleading | Include all pulses and publish both noise-cluster and noise-singleton scores |
| Simulator overfitting | Policies can exploit assumptions that do not generalise | Hold out scenarios, vary conditions and compare against strong baselines |
| Recording bias | Fixed scan recordings cannot reveal all alternative observations | Separate offline association evidence from interactive scheduling evaluation |
| Real-time constraints | A useful policy may still be too slow to execute | Measure latency against the configured decision budget; accelerate only proven bottlenecks |

## 12. Development roadmap and acceptance gates

1. **Broaden association benchmarking.** Expand training-file coverage and freeze a
   preprocessing baseline. Gate: reproducible results with explicit sampling and noise behaviour.
2. **Validate generalisation.** Evaluate the frozen configuration on validation files
   and held-out synthetic conditions. Gate: understand failures, not merely improve an average score.
3. **Introduce learned scheduling.** Define the observation/action/evaluation contract
   and compare a lightweight contextual policy with existing strategies. Gate: measurable
   trade-off improvement without hidden-truth access or unacceptable coverage loss.
4. **Complete the performance model.** Add the remaining prediction/reward metrics and
   periodic-scan reference analysis. Gate: comparable reports under consistent constraints.
5. **Optimise the execution path.** Profile larger workloads, improve batching and data
   structures, then consider Rust. Gate: measured improvement with equivalent outputs.
6. **Deliver the interface.** Add experiment controls, spectrum/decision visualisation
   and report exploration through an API/dashboard. Gate: a reproducible end-to-end demonstration.

## 13. Differentiation and value proposition

The proposed contribution is an integrated, measurable approach—not a claim that
clustering, bandits or adaptive dwell are individually new.

- **Constraint-aware decisions:** account for coverage and retuning rather than only immediate hits.
- **Association-informed context:** aim to connect measurement confidence with scheduling decisions.
- **Change-responsive behaviour:** evaluate recovery when earlier patterns stop being useful.
- **Scalable experimentation:** work with large pulse collections without requiring equivalent RAM.
- **Auditable comparisons:** preserve baseline settings, sampling scope and failure visibility.
- **Modular evolution:** replace data, association or policy components without redesigning the platform.

For RF research laboratories and receiver-development teams, the intended benefit is
faster, more reproducible policy evaluation before hardware trials. Reduced development
cost and improved receiver utilisation are objectives; monetary savings and field
performance have not been established.

## 14. Mapping this brief into the supplied presentation

| Template slide | Suggested emphasis |
|---|---|
| Title | Project pitch, problem ID/title and team details |
| Idea title | Observation bottleneck, closed-loop approach and proposed differentiation |
| Technical approach | Architecture, technology roles, data handling and learning plan |
| Feasibility and viability | Existing foundation, validation methods, engineering hurdles and mitigation |
| Impact and benefits | Performance objectives, intended users, scalable evaluation and delivery goals |
| Research and references | Dataset, benchmark methodology, relevant research and problem-statement source |

Use validation methods rather than test-pass counts. Describe planned capabilities as
design objectives, and retain the sample/training qualification whenever presenting
the association results. Team details remain blank until supplied; the theme still
requires official confirmation.

## 15. References and supporting project material

- [Official SIH problem portal](https://sih.gov.in/sih2026PS)
- [Accessible SIH26055 statement copy — community mirror](https://sih2026.vuce.in/ps/SIH26055)
- [Turing Synthetic Radar Dataset](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset)
- [Dataset research paper](https://arxiv.org/abs/2602.03856)
- [Turing Deinterleaving Challenge](https://github.com/alan-turing-institute/turing-deinterleaving-challenge)
- [Transformer-based deep metric learning research](https://arxiv.org/abs/2503.13476)
- [Project implementation notes](../docs/implementation-notes.md)
- [Dataset workflow and experiment scope](../docs/dataset-workflow.md)
- [Development plan](../docs/plan.md)

The Markdown brief reflects the dataset-integration milestone. The existing PowerPoint
and PDF predate that milestone and are not automatically updated by changes to this file.
