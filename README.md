# Spectra Scheduler

For task-by-task commands, setup and troubleshooting, see the
[running manual](docs/runbook.md).

A passive, receive-only prototype for SIH26055: scheduling a narrow-band receiver
across a wider spectrum. [Project scope](docs/project_scope.md) defines the requirements.

The Python implementation includes a dynamic emitter simulator, a narrow-band
receiver, adaptive scan strategies and repeatable evaluation. A separate dataset
pipeline streams TSRD HDF5 files and benchmarks offline pulse association.
A supervised hit-prediction baseline now supports bounded training, portable JSON
models and held-out comparison; it is experimental, not the default strategy.
See [the learning workflow](docs/learning-workflow.md) for commands and measured results.

## Current capabilities

- Seeded frequency-agile, spatial-scan and periodic-scan simulation with receiver errors.
- Causal learned timing forecasts and action planning that accounts for retuning.
- Console comparisons against round-robin, non-neural phase planning and saved MPC models.
- Browser comparisons with synchronized receiver traces, playback and JSON exports.
- Observation-based tracking, adaptive dwell, stale-belief forgetting and change detection.
- Streamed pulse-data ingestion, offline association and physical-time receive-only replay.
- Paired capture/discovery reports, forecast metrics and bounded CUDA neural evaluation.

See [docs/plan.md](docs/plan.md) for the working plan and
[docs/implementation-notes.md](docs/implementation-notes.md) for a brief explanation of
what each part is for. The implementation-facing literature review is in
[docs/related-work.md](docs/related-work.md).

## Run the trained timing scheduler

The current selected checkpoint is `artifacts/timing-refine-v1/seed-0/best.pt`
(epoch 24). Supply an existing local checkpoint; weights and reports in `artifacts/`
are not included in Git. The command requires the CUDA-enabled `.venv-rl` environment.
See the [environment setup](docs/runbook.md#1-set-up-the-environment) before running it.

```bash
.venv-rl/bin/python -m spectra_scheduler \
  --timing-model artifacts/timing-refine-v1/seed-0/best.pt \
  --mpc-model artifacts/mpc-physical-fresh-control/best.pt \
  --mpc-model artifacts/mpc-physical-gumbel-fresh/best.pt \
  --scenario periodic-scan --runs 30 --seed 28000 --workers 20 \
  --output reports/generated/timing-periodic.json
```

Use `frequency-agile` or `spatial-scan` for the other required behaviors. The
console prints capture and discovery together; JSON retains per-world results
and paired intervals. The [timing findings](docs/timing-model-findings.md) distinguish
checkpoint selection, larger historical comparisons and recent smoke checks.
Experimental Whittle and scan-handover controls did not replace this checkpoint.

## Run the browser interface

```bash
.venv-rl/bin/python web/server.py \
  --timing-model artifacts/timing-refine-v1/seed-0/best.pt
```

Open `http://127.0.0.1:8080` and choose the trained timing comparison. Both policies
receive the same simulated signals and receiver settings. See the
[GUI and HTTP API guide](web/README.md) for available policies, controls and endpoints,
and the [video instructions](web/video/README.md) for the selected demonstration.
The checkpoint is local and requires CUDA; statistical policies also work without it.

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

Available scenarios are `mixed`, `acquisition`, `tracking`, `change`, `crowded`,
`frequency-agile`, `spatial-scan` and `periodic-scan`.
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

Run an experiment without editing Python by supplying a JSON scenario:

```bash
PYTHONPATH=src python3 -m spectra_scheduler \
  --scenario-file examples/custom-scenario.json --runs 100 \
  --output reports/custom.json
```

The format is described in [docs/scenario-format.md](docs/scenario-format.md).

Download the current reference pulse dataset (TSRD) after obtaining Hugging Face access and running
`uvx hf auth login`:

```bash
bash scripts/download_dataset.sh
```

This uses 32 workers and high-performance transfers, keeps all current scan/stare
splits, and excludes the older archive. Files stay in the Git-ignored `data/tsrd/`.
Rerun to resume; optionally supply a worker count. Stop any older manually started
download first. For a detached download with output saved locally:

```bash
mkdir -p data/tsrd
nohup bash scripts/download_dataset.sh > data/tsrd/download.log 2>&1 < /dev/null &
tail -f data/tsrd/download.log
```

Interactive runs show the downloader's progress; redirected logs may update less
frequently. The script prevents duplicate script instances, not manually launched
`hf download` commands.

Run the tests with:

```bash
.venv/bin/python -m pytest -q
```

## Work with downloaded pulse data

Install the optional dataset dependencies, then inspect completed files or run a
bounded HDBSCAN association baseline while the remaining files download:

```bash
uv sync --extra dataset --extra dev
.venv/bin/spectra-dataset inspect --max-files 10
.venv/bin/spectra-dataset evaluate --max-files 10 --sample-rows 10000 \
  --workers 2 --output reports/generated/association.json
```

The pipeline streams HDF5 validation/statistics and clusters a reproducible sample
per file. Reports include association scores, noise coverage, sample fingerprints
and explicit file failures. It does not reinterpret recordings as simulator events
or use emitter labels as model inputs. See [dataset-workflow.md](docs/dataset-workflow.md)
for feature transforms, split boundaries, profiling and measured integration results.

For interactive physical-time receiver simulation on full-spectrum stare recordings,
see [pulse replay](docs/pulse-replay.md). It adds frequency/bandwidth, dwell/retune,
sensitivity, deterministic detection and bounded PDW observations without changing
the existing synthetic simulator or ML training interface.

The [reusable learning and inference interface](docs/replay-interface.md) adds
band/dwell actions, causal PDW features, physical-time rewards/discounts and a
strategy-independent evaluation runner. It can be exercised without training.

Dedicated spatial/periodic/frequency-agile scenarios, frozen multi-policy benchmarks,
and bounded predictor/training adapters are described in
[scenarios, benchmarks and predictors](docs/scenarios-benchmarks-predictors.md).
The shared [evaluation contract](docs/evaluation-contract.md) defines metric units,
prediction targets, missing-data behavior and censoring.

The [trained timing scheduler results](docs/timing-model-findings.md) compare
causal timing forecasts and trajectory-trained policies with native MPC and
round-robin on paired development worlds. The
[trajectory policy research](docs/trajectory-policy-research.md) explains the
implemented objectives and their limitations.
[Joint timing and student-state training](docs/mimo-joint-research.md) and
[multi-teacher policy distillation](docs/mopd-scheduling.md) document further
experiments; their implementation does not imply a demonstrated performance gain.

The [shared experiment workflow](docs/experiment-workflow.md) adds resumable runs,
validation-based checkpoint selection and common artifact management, with a
reference predictor adapter and an explicit pilot configuration.
