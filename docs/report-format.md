# Experiment report format

An experiment report preserves the inputs and aggregate results needed to repeat or
compare a simulation run. It excludes wall-clock timestamps. Legacy simulator
reports also exclude machine details; timing reports retain runtime settings and
resource measurements separately from deterministic receiver results.

## Common metadata

Legacy simulator reports contain:

- `schema_version`: the report contract version;
- `scenario`: the selected built-in scenario;
- `scenario_definition`: the complete custom definition, or `null` for a built-in;
- `start_seed`: the first deterministic seed;
- `runs`: the number of consecutive seeds evaluated;
- `strategy_results`: one aggregate record per scheduler;
- `track_association`: aggregate association results for the track-aware scheduler.

The worker count is not stored because sequential and parallel execution are required
to produce identical results.

## JSON

JSON preserves the nested report structure. Strategy names map to their aggregate
metrics, while `track_association` is a separate object. Keys are sorted when written
so two equivalent reports have a stable textual layout.

## CSV

CSV uses one row per result. `record_type` is either `strategy` or
`track-association`, and `name` identifies the scheduler. Columns that do not apply to
a row are empty. Common metadata is repeated on every row so filtered files remain
self-describing.

## Compatibility

Readers should reject an unsupported `schema_version` instead of guessing field
meaning. Adding a new optional metric does not require changing existing meanings;
renaming or redefining a field requires a new schema version.

## Trained timing comparisons

The main CLI's `--timing-model` path writes JSON schema version 2 with
`backend: synthetic_timing`. It retains scenario/seed metadata and adds
`checkpoint_sha256`, `policy`, `baseline`, `summary`, per-world `results`,
`resources` and an explicit development-comparison `scope`. Every paired policy
receives the same truth denominator and physical time budget. Capture ratios
with no truth pulses are unavailable, rather than scored as zero.

`summary` includes metric means, contributing and paired group counts, paired
mean differences and bootstrap intervals against the declared listening-dwell
round-robin baseline. CUDA/host peak memory and elapsed profiling durations are
stored under `resources`; no calendar timestamp is added.

The optional `inference_runtime` object records `cuda_graph`, `planner` (`numpy`
or `native`) and `planner_library_sha256` (null for NumPy). These identify runtime
acceleration without changing metric definitions or the selected policy.

Timing CSV uses one row per policy/metric, with columns `policy`, `metric`, `mean`,
`groups`, `paired_groups`, `paired_mean_difference` and `paired_bootstrap_95_interval`.
Use JSON when full per-world results and checkpoint provenance are needed.
Experimental scan-study JSON has its own frozen configuration and results layout;
it is not the main CLI schema.
