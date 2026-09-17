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
not to the total downloaded collection. Files changing during the read are rejected.

Recorded scan data supports association evaluation, not direct counterfactual
evaluation of a different receiver schedule. Dataset labels are arbitrary within
each file; they are not global emitter classes. Keep the provided train, validation
and test splits separate and reserve test files for frozen configurations.

Schema and units follow the [official dataset card](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset)
and [companion loader](https://github.com/alan-turing-institute/turing-deinterleaving-challenge/blob/master/src/turing_deinterleaving_challenge/data/structure.py).
