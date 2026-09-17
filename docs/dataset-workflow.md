# Dataset workflow

The dataset extra adds HDF5 ingestion and offline clustering dependencies without
changing the simulator's inputs:

```bash
uv sync --extra dataset --extra dev
```

`dataset_io` reads completed `.h5` files from one explicitly selected receiver mode
and split. It never traverses `archive` or download-cache directories. Hugging Face
`.incomplete` files are excluded. A final filename alone is not proof of integrity:
the reader validates the HDF5 schema and reads every selected file in chunks.

Columns are resolved from `metadata/feature_names`, not assumed by position. The
canonical order and dataset-card units are arrival time (microseconds), frequency
(MHz), pulse width (microseconds), angle (degrees), and amplitude (dB). Amplitude is
not converted to dBm. Simulator steps are not equated to arrival-time microseconds.
Arrival times must be ordered and features finite. Empty and unlabelled files are
supported. Labels of shape `(N,)` and `(N, 1)` stay separate from features; transmitter
metadata is never deserialised into the inference path.

Full-file statistics use chunk-combined means and variances. Optional sampling
selects uniform row indices without replacement, independently of labels, and
restores source order. A seed gives the same sample regardless of chunk size.
Memory is proportional to batch size, sample size and distinct file-local labels,
not to the total pulse collection. Report metadata grows with the number of selected
files. Files changing during the read are rejected.

Recorded scan data supports association evaluation, not direct counterfactual
evaluation of a different receiver schedule. Dataset labels are arbitrary within
each file; they are not global emitter classes. Keep the provided train, validation
and test splits separate and reserve test files for frozen configurations.

Schema and units follow the [official dataset card](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset)
and [companion loader](https://github.com/alan-turing-institute/turing-deinterleaving-challenge/blob/master/src/turing_deinterleaving_challenge/data/structure.py).

## Commands

Run from the repository root after installing the extras above:

```bash
# Full validation pass on every completed scan training file.
.venv/bin/spectra-dataset inspect --max-files 0 --workers 2 \
  --output reports/generated/tsrd-scan-train-audit.json

# Bounded offline clustering experiment; training split is the default.
.venv/bin/spectra-dataset evaluate --max-files 10 --sample-rows 10000 \
  --features signature --workers 2 \
  --output reports/generated/tsrd-scan-signature.json

# Compare raw PDWs using the exact same sampled pulse population.
.venv/bin/spectra-dataset evaluate --max-files 10 --sample-rows 10000 \
  --features raw --workers 2 \
  --output reports/generated/tsrd-scan-raw.json
```

`python -m spectra_scheduler.dataset_cli` is the equivalent module entry point.
Choose `--mode scan|stare`, `--split train|val|test`, `--seed`, `--batch-rows`,
`--min-cluster-size` and `--min-samples` explicitly for other experiments. The
defaults select the first ten completed filenames in lexicographic order, not a
representative random sample of the dataset. `--max-files 0` selects the entire
chosen split. Files added after discovery wait until the next invocation.

## Baseline and evaluation contract

This is per-file, transductive HDBSCAN clustering, not a trained scheduler or an
emitter classifier. It does not require knowing the number of emitters. The `raw`
baseline uses all five native columns unchanged. The `signature` baseline excludes
absolute arrival time, robust-scales frequency, log1p pulse width and amplitude,
and represents arrival angle by sine/cosine to handle circular wraparound. Its
scaler is fitted on the unlabelled sample for that file; no class labels guide it.
Negative widths are rejected by the signature transform rather than silently clipped.

Sampling is uniform without replacement over each whole file. The effective seed
is derived from the configured seed and relative filename, so worker count and
file-selection limits do not change a given file's sample. Reports include sample
row counts and SHA-256 fingerprints of sampled indices, features and predictions.
These are reproducibility fingerprints, not full-file checksums against the Hub.

HDBSCAN runs only on the bounded sample, with a KD-tree backend and one numerical
thread per file worker. `--sample-rows` is an explicit accuracy/resource trade-off:
subsampling changes density and can omit rare emitters. Do not present these scores
as full-file or published-leaderboard results. This uses scikit-learn's HDBSCAN;
its `min_samples` convention differs from the separate `hdbscan` package.

Each file is scored independently. Summary metrics are unweighted file means,
never pooled emitter IDs. Reports distinguish total validated pulses from sampled
scored pulses. Empty files, missing labels and corrupt inputs are visible, and no
invalid file is silently replaced. A run with file errors exits with status 1 but
still saves results for the valid files. An evaluation with no scored files also
exits with status 1. JSON reports are replaced atomically; input HDF5 files cannot
be used as report paths.

Metrics include homogeneity, completeness, V-measure and pairwise precision/recall/F1.
Pair scores use cluster contingency counts rather than quadratic pulse-pair matrices.
Precision/recall/F1 are zero when there are no relevant positive pairs. Noise is
not dropped from evaluation:

- `metrics` treats HDBSCAN's noise label as one cluster, matching ordinary label
  scoring conventions.
- `noise_singleton_metrics` treats every rejected pulse as its own cluster.
- `noise_fraction` and `predicted_clusters` expose rejection and clustering behaviour.

By default reports contain no wall-clock timestamps or runtime fields. `--profile`
adds elapsed read/total durations, read throughput and process peak RSS. Peak RSS is
the lifetime high-water mark of a worker (including imports and earlier files), not
memory uniquely attributable to that file or total memory across all workers. More
workers multiply memory demand; start with two. Inspecting files does not import
the clustering implementation, keeping audit memory smaller.

Reference: [scikit-learn HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html).

## Integration evidence

The full downloaded scan training split was streamed: 2,500 files, 233,172,417 pulses,
eight valid empty files, no schema/read errors. With 65,536-row batches and two
workers, the largest inspection-worker peak RSS was about 59.4 MiB. An additional
three stare training files contained 3,609,482 pulses and passed inspection; that
single-process run peaked at about 68.7 MiB. Validation and test splits were not used.

Development comparison: first ten scan training filenames, 10,000 uniformly sampled
pulses per file, seed 0, minimum cluster size 20, minimum samples 10. Both configurations
validated 838,204 source pulses and scored the same 100,000 sampled pulses:

| Feature baseline | Macro V-measure | Macro pair F1 | Mean noise fraction |
|---|---:|---:|---:|
| Raw PDWs | 0.4519 | 0.1815 | 0.0499 |
| Scaled signatures | 0.8435 | 0.8121 | 0.0440 |

These figures use the single-noise-cluster convention. Worker peak RSS was about
150 MiB for these bounded clustering runs. The difference demonstrates the effect
of preprocessing on this training subset, not held-out generalisation, online
scheduling improvement or native-engine performance. Generated detailed reports
remain under the ignored `reports/generated/` directory.
