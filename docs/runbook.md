# Running Spectra Scheduler

This manual explains which command to run for each task. All commands assume a
terminal opened in the repository root (`spectra-scheduler`), not its parent folder.
There is currently no GUI or API server to start.

## Command map

| What you want to do | Command or script |
| --- | --- |
| Download or resume the dataset | `bash scripts/download_dataset.sh 32` |
| Compare existing scheduling strategies | `.venv/bin/python -m spectra_scheduler` |
| Run the older comparison script | `.venv/bin/python scripts/run_comparison.py` |
| Check downloaded HDF5 files | `.venv/bin/python -m spectra_scheduler.dataset_cli inspect` |
| Benchmark offline pulse clustering | `.venv/bin/python -m spectra_scheduler.dataset_cli evaluate` |
| Train a hit-prediction model | `.venv/bin/python -m spectra_scheduler.learning_cli train` |
| Evaluate a saved scheduling model | `.venv/bin/python -m spectra_scheduler.learning_cli evaluate` |
| Run the full test suite | `.venv/bin/python -m pytest -q` |

The learning commands require additional arguments, shown below. The examples use
the virtual environment's Python explicitly: shell activation is not required.

## 1. Set up the environment

Requirements: `uv`, Python 3.12 (managed by uv if needed), and on Linux, Bash and
`flock` from util-linux for the download script. Initial installation needs network
access.

```bash
cd /home/siwach/Projects/sih/spectra-scheduler
uv sync --locked --extra dataset --extra learning --extra dev
.venv/bin/python --version
```

On another machine, replace the `cd` path with your checkout. Keep all three extras
when syncing the working environment; syncing fewer extras can remove optional
packages. Once installed, the commands below run directly without a registry check.

Installed shorthand commands are `spectra-scheduler`, `spectra-dataset` and
`spectra-learn` under `.venv/bin/`. For example, `.venv/bin/spectra-learn train`
is equivalent to `.venv/bin/python -m spectra_scheduler.learning_cli train`.

## 2. Download or resume the dataset

Obtain access to the Hugging Face dataset, then authenticate interactively. Do not
put a token into a script, command example or committed file.

```bash
uvx hf auth login
bash scripts/download_dataset.sh 32
```

This downloads current scan/stare splits into `data/tsrd/`, excludes `archive/**`,
and uses 32 download workers. Rerun the same command after an interruption to reuse
completed files and resume through the downloader's cache. More workers do not
guarantee higher throughput; this is independent of compute workers below.

To detach the download and follow its log:

```bash
mkdir -p data/tsrd
nohup bash scripts/download_dataset.sh 32 > data/tsrd/download.log 2>&1 < /dev/null &
tail -f data/tsrd/download.log
```

`Ctrl+C` exits `tail`, not the detached download. Logs may update less often than an
interactive progress display. Do not launch both foreground and detached downloads:
the script blocks duplicate script instances, but not independently launched
`hf download` commands. In a foreground download, `Ctrl+C` interrupts the download.

## 3. Run simulator comparisons

A quick run, requiring no downloaded data:

```bash
.venv/bin/python -m spectra_scheduler
```

Compare the thirteen existing strategies over multiple seeds and save a report:

```bash
.venv/bin/python -m spectra_scheduler \
  --scenario mixed --runs 100 --seed 0 --workers 4 \
  --output reports/generated/mixed.json
```

The terminal shows interception, discovery, retuning, coverage and delay metrics.
Saved reports also include track-association results. Use a `.csv` output instead
of `.json` for a spreadsheet-friendly report.

Available scenarios:

| Scenario | Purpose |
| --- | --- |
| `mixed` | Several emitter behaviours and receiver imperfections |
| `acquisition` | Discovering emitters |
| `tracking` | Following a moving signal |
| `change` | Responding to an emitter mode change |
| `crowded` | More crowded traffic and ambiguous signal association |

Show association metrics without saving a report:

```bash
.venv/bin/python -m spectra_scheduler --scenario crowded --runs 30 --association
```

Run a custom scenario definition:

```bash
.venv/bin/python -m spectra_scheduler \
  --scenario-file examples/custom-scenario.json --runs 30 --workers 2 \
  --output reports/generated/custom.json
```

Use either `--scenario` or `--scenario-file`, not both. See
[scenario-format.md](scenario-format.md) for the JSON structure and
[report-format.md](report-format.md) for report fields. The older
`scripts/run_comparison.py` is a wrapper for the same simulator interface, not a
separate experiment. Existing policies run by default; trained timing checkpoints
can be included below. The older hit-prediction models use section 6.

To include the trained timing planner, use the CUDA environment and a local
checkpoint. Capture and discovery are printed together; JSON reports also include
forecast scores and paired intervals. The existing comparison remains available
without `--timing-model`.

```bash
.venv-rl/bin/python -m spectra_scheduler \
  --timing-model artifacts/timing-refine-v1/seed-0/best.pt \
  --scenario frequency-agile --runs 30 --seed 24000 --workers 20 \
  --output reports/generated/timing-agile.json
```

Add repeatable `--mpc-model PATH` options for saved physical MPC checkpoints on
`frequency-agile`, `spatial-scan` or `periodic-scan`. Timing comparisons also
include full listening-dwell round-robin controls and a non-neural phase predictor
with the same planner. `--inference-batch-size` defaults to 20; `.csv` output is
supported. CUDA is required. Track association uses the original comparison.

## 4. Inspect completed dataset files

Start with a small selection:

```bash
.venv/bin/python -m spectra_scheduler.dataset_cli inspect \
  --root data/tsrd --mode scan --split train --max-files 10 \
  --output reports/generated/scan-inspect.json
```

This validates schema, feature order and values, then streams full-file statistics.
It does not fit clusters or train a scheduler. Completed files can be inspected
while other files download; partial/cache files are not selected. A missing or empty
selection is not evidence that the download is complete.

Inspect all currently completed scan training files:

```bash
.venv/bin/python -m spectra_scheduler.dataset_cli inspect \
  --mode scan --split train --max-files 0 --workers 2 --profile \
  --output reports/generated/scan-inspect-all.json
```

`--max-files 0` means all discovered files, not zero files. It does not verify the
remote repository's total download completeness. Use `--mode stare` for stare data
or `--root /path/to/tsrd` when the dataset is stored elsewhere.

## 5. Evaluate dataset pulse clustering

Run the signature-feature HDBSCAN baseline on a bounded sample from each file:

```bash
.venv/bin/python -m spectra_scheduler.dataset_cli evaluate \
  --mode scan --split train --max-files 10 --sample-rows 10000 \
  --features signature --seed 0 --workers 2 --profile \
  --output reports/generated/signature-association.json
```

For a raw-feature comparison, rerun with `--features raw` and a different output
filename. Keep file selection, sample size and seed identical for a fair comparison.

| Option | Meaning |
| --- | --- |
| `--max-files` | Limit files; default 10, zero means all discovered files |
| `--batch-rows` | Rows read per HDF5 chunk; default 65,536 |
| `--sample-rows` | Maximum pulses clustered per file; default 20,000 |
| `--workers` | Independent file-processing processes; default 1 |
| `--min-cluster-size` | HDBSCAN minimum cluster size; default 20 |
| `--min-samples` | HDBSCAN density parameter; default 10 |
| `--profile` | Include elapsed runtime, throughput and process peak memory |
| `--split` | `train`, `val` or `test`; default `train` |

Full files are still streamed for validation, even when clustering uses a small
sample. Increasing sample size can substantially increase clustering cost;
increasing workers also increases total memory consumption. Start with the example
before expanding coverage.

Look at V-measure, pairwise F1, noise coverage and explicit per-file errors in the
JSON report. These measure association, not scheduler interception. A report may
be written even when the command exits nonzero because some files failed. Empty or
unlabelled files cannot provide the same scored evaluation as labelled files.

Use training files for development. Move to `--split val` after fixing an initial
configuration; reserve `--split test` for final evaluation. Do not repeatedly tune
against test results. See [dataset-workflow.md](dataset-workflow.md) for details.

## 6. Train and evaluate the learned scheduler

This path uses simulated receiver feedback. It does **not** train the scheduler
from TSRD recordings or use their emitter labels as policy inputs.

Train and save a model:

```bash
.venv/bin/python -m spectra_scheduler.learning_cli train \
  --runs 100 --seed 0 --max-examples 50000 \
  --output artifacts/hit-model.json
```

Training defaults to `mixed acquisition tracking change`, leaving `crowded` out.
`--runs` is runs per scenario; `--max-examples` caps stored training examples, not
the number of simulated steps. The output contains fitted coefficients, feature
schema, policy settings and training provenance.

Evaluate the saved model on separate seeds:

```bash
.venv/bin/python -m spectra_scheduler.learning_cli evaluate \
  --model artifacts/hit-model.json --runs 50 --seed 10000 \
  --output reports/generated/learned-held-out.json
```

The report compares all thirteen existing strategies, the learned scheduler and
a constant-prediction model with identical scheduling rules. It includes paired
interception differences, approximate confidence intervals and Brier scores.
The command rejects evaluation seeds overlapping training seeds.

For a smaller evaluation, add `--scenarios mixed crowded --runs 5`. Learning
commands currently have no `--workers` option and print a completion summary rather
than a live progress bar. Initial results do not establish consistent ML superiority;
the existing simulator defaults are unchanged. See
[learning-workflow.md](learning-workflow.md) for results and limitations.

## 7. Read results and check the code

Open a generated JSON report in your editor, or pretty-print it in the terminal:

```bash
.venv/bin/python -m json.tool reports/generated/learned-held-out.json
```

Run the full suite, or only the learning tests:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/test_learning.py -q
```

Pytest is the full-suite runner; `unittest discover` alone does not collect all
pytest-style tests. For read-only lint checks on the newly added learning modules:

```bash
.venv/bin/ruff check src/spectra_scheduler/learned_scheduler.py \
  src/spectra_scheduler/learning.py src/spectra_scheduler/learning_cli.py \
  tests/test_learning.py
```

## 8. Output locations and common problems

| Location | Contents |
| --- | --- |
| `data/tsrd/` | Downloaded data, downloader cache and optional download log |
| `artifacts/` | Saved model JSON files |
| `reports/generated/` | Simulator, dataset and learned-policy reports |
| `docs/` | Usage documentation and experiment explanations |

Data, artifacts and generated reports are Git-ignored. Reusing an output filename
replaces the previous report/model; use descriptive distinct names to preserve
comparisons. The learning evaluator prevents its report from overwriting its input
model. Reports do not add wall-clock timestamps; optional profiling reports elapsed
durations.

- **Missing module or command:** run the setup command with all extras from the
  repository root. Use `.venv/bin/python`, not a different system interpreter.
- **Dataset access denied:** confirm dataset access and run `uvx hf auth login`.
- **No files selected:** check `--root`, `--mode`, `--split` and whether files have
  finished downloading. The expected layout includes `scan/train_scan/*.h5` or the
  corresponding mode/split directory.
- **High memory use:** reduce dataset `--sample-rows` and `--workers`; reduce
  `--batch-rows` for streaming buffers. Do not start with all files and large samples.
- **Training/evaluation overlap:** choose disjoint seed intervals. Training with
  `--seed 0 --runs 100` uses seeds 0–99 in each selected scenario.
- **Need another option:** inspect the local help below; simulation, dataset and
  learning commands have different flags.

```bash
.venv/bin/python -m spectra_scheduler --help
.venv/bin/python -m spectra_scheduler.dataset_cli --help
.venv/bin/python -m spectra_scheduler.learning_cli train --help
.venv/bin/python -m spectra_scheduler.learning_cli evaluate --help
```
